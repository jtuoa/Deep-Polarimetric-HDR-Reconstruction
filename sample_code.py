import argparse
import glob
import numpy as np
import os
import torch
import cv2
import opt
from torchvision import transforms
from torch.utils import data

from mylib.image import unnormalize
from mylib.img_io import load_image, writeLDR, writeNPY, writeEXR
from mylib.io import load_ckpt
from mylib.io import print_
from mylib.util import make_dirs, get_saturated_regions 

from network.softconvmask_ import SoftConvNotLearnedMaskUNet


parser = argparse.ArgumentParser(description="Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss")
parser.add_argument('--i0', '-i0', type=str, required=True, help='Input images pol0 directory.')
parser.add_argument('--i45', '-i45', type=str, required=True, help='Input images pol45 directory.')
parser.add_argument('--i90', '-i90', type=str, required=True, help='Input images pol90 directory.')
parser.add_argument('--i135', '-i135', type=str, required=True, help='Input images pol135 directory.')
parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to output directory.')
parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the trained CNN weights.')
parser.add_argument('--cpu', action='store_true')

def print_test_args(args):
    print_("\n\n\t-------------------------------------------------------------------\n", 'm')
    print_("\t  HDR image reconstruction from a single exposure using deep CNNs\n\n", 'm')
    print_("\t  Settings\n", 'm')
    print_("\t  -------------------\n", 'm')
    print_("\t  Input image pol0 directory/file:     %s\n" % args.i0, 'm')
    print_("\t  Output directory:               %s\n" % args.out_dir, 'm')
    print_("\t  CNN weights:                    %s\n" % args.weights, 'm')
    print_("\t-------------------------------------------------------------------\n\n\n", 'm')


#Load dataset
class HDRTestDataset(torch.utils.data.Dataset):
    def __init__(self, I0, I45, I90, I135, image_transform):
        super(HDRTestDataset, self).__init__()
        self.images = self._load_dataset(I0, I45, I90, I135)
        self.dataset_len = len(self.images)
        self.img_transform = image_transform

    def __getitem__(self, index):
        input_dir = self.images[index]

        #get Lc
        image0 = load_image(input_dir[0]).astype('float32') 
        image45 = load_image(input_dir[1]).astype('float32')
        image90 = load_image(input_dir[2]).astype('float32')
        image135 = load_image(input_dir[3]).astype('float32')
        image = self.compute_mean(image0, image45, image90, image135)
        assert np.min(image) >= 0.0 and np.max(image) <= 1.0

        #get Ht (HDR formulation)
        t0 = 769/1e6
        image_hdr = self.compute_xuesong_HDR_color(image0, image45, image90, image135,t0)
        image_hdr = image_hdr / np.max(image_hdr)
        assert np.min(image_hdr) >= 0.0 and np.max(image_hdr) <= 1.0
        image_hdr = torch.from_numpy(image_hdr).permute(2,0,1)
        
        #get dolp
        dop = self.compute_dop_color(image0, image45, image90, image135)
        assert np.min(dop) >= 0.0 and np.max(dop) <= 1.0
        dop = torch.from_numpy(dop).permute(2,0,1)

        # get mask K
        conv_mask = 1 - get_saturated_regions(image)
        assert np.min(conv_mask)>=0.0 and np.max(conv_mask)<=1.0
        conv_mask = torch.from_numpy(conv_mask).permute(2,0,1)

        # apply transform to Lc
        image = self.img_transform(image)

        return image, conv_mask, dop, image_hdr


    def compute_dop_color(self, I0, I45, I90, I135):
        w,h,ch = I0.shape
		
        dop_color = np.zeros_like(I0)
        for i in range(ch):
            S0 = 0.5 * (I0[:,:,i] + I45[:,:,i] + I90[:,:,i] + I135[:,:,i])
            S1 = I0[:,:,i] - I90[:,:,i]
            S2 = I45[:,:,i] - I135[:,:,i]
			
            dop_color[:,:,i] = np.sqrt(S1**2 + S2**2) / (S0+1e-12)
            if not np.max(dop_color[:,:,i]) == 0:
                dop_color[:,:,i] = dop_color[:,:,i] / np.max(dop_color[:,:,i])
			
        assert np.isnan(dop_color).any() == False
        return dop_color
	
    def compute_mean(self, I0, I45, I90, I135):
        I0_norm = I0 / 255.0 
        I45_norm = I45 / 255.0 
        I90_norm = I90 / 255.0 
        I135_norm = I135 / 255.0 

        Imean = I0_norm + I45_norm + I90_norm + I135_norm
				
        return Imean / 4.0
	
    def compute_xuesong_HDR_color(self, I0, I45, I90, I135, t0):
        w,h,ch = I0.shape		
        hdr_color = np.zeros_like(I0)
        for i in range(ch):
        	L0L2 = I0[:,:,i] + I90[:,:,i]
        	L1L3= I45[:,:,i] + I135[:,:,i]
        	L0L2norm = L0L2 / np.max(L0L2) 
        	L1L3norm = L1L3 / np.max(L1L3) 
        	
        	I0_norm = I0[:,:,i] / 255.0 
        	I45_norm = I45[:,:,i] / 255.0 
        	I90_norm = I90[:,:,i] / 255.0 
        	I135_norm = I135[:,:,i] / 255.0 

        	K0 = self.weight_map(L0L2norm)
        	K1 = I0_norm + I90_norm
        	hdr_color[:,:,i] = (K0 * K1) / (K0 + 1e-12)

        	K2 = self.weight_map(L1L3norm)
        	K3 = I45_norm + I135_norm
        	hdr_color[:,:,i] += (K2 * K3) / (K2 + 1e-12)
					
        	hdr_color[:,:,i] = hdr_color[:,:,i] / t0

        return hdr_color		
        	
	
    def weight_map(self, pixel):
        num = -1 * ((pixel - 0.5) ** 2)
        denom = 2 * (0.2 ** 2) 
        return np.exp(num/denom)
        	
		
    def __len__(self):
        return self.dataset_len

    def _load_dataset(self, I0, I45, I90, I135):
        images = []
        ang = [0, 45, 90, 135]
        for i in range(20):
        	images_tmp = []
        	images_tmp.append("{}/{}_in_{}.png".format(I0, i, ang[0]))
        	images_tmp.append("{}/{}_in_{}.png".format(I45, i, ang[1]))
        	images_tmp.append("{}/{}_in_{}.png".format(I90, i, ang[2]))
        	images_tmp.append("{}/{}_in_{}.png".format(I135, i, ang[3]))
        	
        	images.append(images_tmp)

        return images



if __name__ == '__main__':
    args = parser.parse_args()
    args.train = False
    print_test_args(args)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print_('\tUsing device: {}.\n'.format(device))

    # create output directory
    make_dirs(args.out_dir)

    # load test data
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

    dataset = HDRTestDataset(args.i0, args.i45, args.i90, args.i135, img_transform)
    iterator_test_set = data.DataLoader(dataset, batch_size=1)
    print_('\tLoaded {} test images.\n'.format(len(dataset))) 

	#Load weights
    model = SoftConvNotLearnedMaskUNet().to(device)
    model.load_state_dict(torch.load(args.weights))
    print('load trained weights')


    print_("Starting prediction...\n\n")
    model.eval()
    for i, (image, mask_M, dop, image_hdr) in enumerate(iterator_test_set):
        print("Image %d/%d"%(i+1, len(dataset)))
        print_("\t(Saturation: %0.2f%%)\n" % (100.0*(1-mask_M.mean().item())), 'm') 
        print_("\tInference...\n")

        with torch.no_grad():
            mask_N = mask_M + dop #mask M1 for feature masking
            max_mask_N = torch.max(mask_N)          
            mask_N = mask_N / max_mask_N
            assert torch.min(mask_N) >= 0.0 and torch.max(mask_N) <= 1.0
			
            pred_img = model(image.to(device), mask_N.to(device))

        print_("\tdone model inference ...\n")


        image = unnormalize(image).permute(0,2,3,1).numpy()[0,:,:,:] 
        mask_M = mask_M.permute(0,2,3,1).numpy()[0,:,:,:] 
        mask_N = mask_N.permute(0,2,3,1).numpy()[0,:,:,:] 
        dop = dop.permute(0,2,3,1).numpy()[0,:,:,:]
        image_hdr = image_hdr.permute(0,2,3,1).numpy()[0,:,:,:] 
        pred_img = pred_img.cpu().permute(0,2,3,1).numpy()[0,:,:,:] 


        #coefficients 
        beta = dop / max_mask_N.numpy() 
        gamma = 1.0 - (mask_N / max_mask_N.numpy())
        
        #Ht
        t2_norm = (beta / (beta + gamma)) * np.power(image_hdr, 4)
        assert np.min(t2_norm)>=0.0 and np.max(t2_norm)<=1.0
        
        #Hd
        t3_norm = (gamma / (beta + gamma)) * pred_img #t2norm+t3norm
        t3_norm = t3_norm - np.min(t3_norm)
        if np.max(t3_norm) == 0:
        	t3_norm = np.zeros(t3_norm.shape) #avoid div by 0
        else:
        	t3_norm = t3_norm / np.max(t3_norm)
        
        assert np.min(t3_norm)>=0.0 and np.max(t3_norm)<=1.0
        

        #HDR formulation 
        H = t2_norm + t3_norm


        #save EXR images
        writeEXR(H, '{}/im_{}.exr'.format(args.out_dir, i+1)) 
        writeNPY(H, '{}/im_{}.npy'.format(args.out_dir, i+1)) 

        

    print_('done saved results ... \n')
    
    
