import cv2
import os
import numpy as np
import pdb
import torch

#from lib.util import make_dirs
#from lib.util import get_saturated_regions

#make_dirs(args.out_dir) #lib
def make_dirs(make_path):
	if not os.path.exists(make_path):
		os.mkdir(make_path)


def saturated_channel_(im, th):
    return np.minimum(np.maximum(0.0, im - th) / (1 - th), 1)

def get_saturated_regions(im, th=0.95):
    #pdb.set_trace()
    w,h,ch = im.shape

    mask_conv = np.zeros_like(im)
    for i in range(ch):
        mask_conv[:,:,i] = saturated_channel_(im[:,:,i], th)

    return mask_conv#, mask

    
def weight_map(im):
	mu = 0.5 #lower shifts left: 0.3, 0.5, 0.7
	sigma = 0.5 #lower compress: 0.2, 0.5, 1.0
	num = -0.5 * ((im - mu) ** 2)
	denom = (sigma ** 2)
	G = (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(num/denom)
	return G / np.max(G)

def get_saturated_regions_gauss(im):
	#pdb.set_trace()
	w,h,ch = im.shape #[0,1]
	mask_conv = np.zeros_like(im)
	for i in range(ch):
		mask_conv[:,:,i] = weight_map(im[:,:,i])
    
	return mask_conv

def compute_mean_channels(image):
	#pdb.set_trace()
	#same result as below
	'''
	image_R = (image[:,0,:,:] + image[:,3,:,:] + image[:,6,:,:] + image[:,9,:,:])/4.0
	image_G = (image[:,1,:,:] + image[:,4,:,:] + image[:,7,:,:] + image[:,10,:,:])/4.0
	image_B = (image[:,2,:,:] + image[:,5,:,:] + image[:,8,:,:] + image[:,11,:,:])/4.0
	
	image_R = torch.unsqueeze(image_R, 0)
	image_G = torch.unsqueeze(image_G, 0)
	image_B = torch.unsqueeze(image_B, 0)
	
	return torch.cat([image_R, image_G, image_B], dim=1)
	'''
	#mean across pol images
	return (image[:,:3,:,:]+image[:,3:6,:,:]+image[:,6:9,:,:]+image[:,9:12,:,:])/ 4.0  
 
	

def compute_mean_channels_train(image):
	#pdb.set_trace()
	image0 = (image[0,:3,:,:] + image[0,3:6,:,:] + image[0,6:9,:,:] + image[0,9:12,:,:]) / 4.0 #batch0 mean across pol images
	image1 = (image[1,:3,:,:] + image[1,3:6,:,:] + image[1,6:9,:,:] + image[1,9:12,:,:]) / 4.0
	image2 = (image[2,:3,:,:] + image[2,3:6,:,:] + image[2,6:9,:,:] + image[2,9:12,:,:]) / 4.0
	image3 = (image[3,:3,:,:] + image[3,3:6,:,:] + image[3,6:9,:,:] + image[3,9:12,:,:]) / 4.0
	
	image0 = torch.unsqueeze(image0, 0)
	image1 = torch.unsqueeze(image1, 0)
	image2 = torch.unsqueeze(image2, 0)
	image3 = torch.unsqueeze(image3, 0)
	
	image = torch.cat([image0, image1, image2, image3], dim=0)
	
	return image		
			
	    
'''
def get_saturated_regions(image_in):
	#input: image 3 channel
	#return: M wxhxc (inverse of Fig. 2)
	W, H, C = image_in.shape[0], image_in.shape[1], image_in.shape[2]
	alpha = 0.96 #threshold 0.96
	image_rgb = []
	for i in range(3):
		image = image_in[:,:,i] / np.max(image_in[:,:,i]) #[0,1]
		image[image <= alpha] = 0.0 
		image = image.flatten().tolist()
		image = [1.0 - i if i > alpha else i for i in image]
		image = np.array(image).reshape([W, H]).astype('float32')
		image_rgb.append(image)

	image_rgb = np.array(image_rgb).transpose(1,2,0)
	assert image_rgb.shape == (W,H,C)
	return image_rgb

#Eilersten HDRCNN eqn2
def get_saturated_regions2(image_in):		
	#pdb.set_trace()
	W, H, C = image_in.shape[0], image_in.shape[1], image_in.shape[2]
	tau = 0.96
	image_in = (image_in / np.max(image_in)).astype('float32')
	max_c = np.max(image_in, axis=2)
	max_c = max_c - tau
	out = np.maximum(max_c, 0) #ensure no neg
	assert np.min(out) >= 0.0 and np.max(out) <= 1.0
	
	#denom = 1 - tau
	
	#out = numer / denom 
	#out = np.clip(out, 0.0, 1.0) #rounding issue will output 1.0000006
	#assert np.min(out) >= 0.0 and np.max(out) <= 1.0
			
	out3 = []
	out3.append(out)
	out3.append(out)
	out3.append(out)
	
	out3 = np.array(out3).transpose(1,2,0)
	assert out3.shape == (W,H,C)

	return out3
'''	
