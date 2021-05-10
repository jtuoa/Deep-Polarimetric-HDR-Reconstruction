import cv2
import os
import numpy as np
import torch

def make_dirs(make_path):
	if not os.path.exists(make_path):
		os.mkdir(make_path)


def saturated_channel_(im, th):
    return np.minimum(np.maximum(0.0, im - th) / (1 - th), 1)

def get_saturated_regions(im, th=0.95):
    w,h,ch = im.shape

    mask_conv = np.zeros_like(im)
    for i in range(ch):
        mask_conv[:,:,i] = saturated_channel_(im[:,:,i], th)

    return mask_conv
    

def compute_mean_channels(image):
	return (image[:,:3,:,:]+image[:,3:6,:,:]+image[:,6:9,:,:]+image[:,9:12,:,:])/ 4.0  
 
			
			

