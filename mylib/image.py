import torch
import opt
import pdb

#from lib.image import unnormalize
#image = unnormalize(image).permute(0,2,3,1).numpy()[0,:,:,:] #lib

def unnormalize(x):
	#input: LDR input tensor norm by self.transform [neg, pos] val can be > 1
	#return: image [0,1]
	#pdb.set_trace()
	x = x.transpose(1,3)
	x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
	x = x.transpose(1,3)
	return x

def unstandardize(image):
	#return: image [0,1]
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]
	
	mean = torch.tensor(MEAN)[None,:,None,None]
	std = torch.tensor(STD)[None,:,None,None]
	
	image = (image * std) + mean
	
	image = image - torch.min(image)
	image = image / torch.max(image)
	
	return image
