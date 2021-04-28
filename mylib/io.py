import cv2
import os
import numpy as np
import torch
import pdb
import argparse, os, sys
import torch.nn as nn

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

def load_ckpt(ckpt_name, models, optimizers=None):
    #pdb.set_trace()
    #fine-tuning: load model weights only
    #load_ckpt
    #optimizer = optim.Adam
    # ... train code
    
    #resuming: load model weights and optimizer
    #laod_ckpt
    # ... train code
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    epoch = ckpt_dict['n_iter'] if 'n_iter' in ckpt_dict else 0
    step = ckpt_dict['step'] if 'step' in ckpt_dict else 0

    return step,  epoch
    
    	
#from lib.io import load_ckpt
#load_ckpt(args.weights, [('model', model)]) #lib
def MY_load_ckpt(weights_path, model_param):
	#input: weights_path, model
	#return: model w/ weights loaded
	print('see lib io.py')
	pdb.set_trace()
	
	#https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c
	#print("Model layers")
	#for param_tensor in model_param[0][1].state_dict():
		#print(param_tensor, "\t", model_param[0][1].state_dict()[param_tensor].size())
	

	name = model_param[0][0] #name
	net = model_param[0][1] #structure
	ckpt = torch.load(weights_path)[name] #access key in state_dict
	net.load_state_dict(ckpt)
	net.eval()
	
	#Err: missing keys
	'''
	import torch.nn as nn
	net = nn.DataParallel(net)
	net.load_state_dict(
		torch.load(weights_path)
	)
	'''
	
	return net


