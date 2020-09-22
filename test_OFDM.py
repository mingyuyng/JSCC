# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import models.channel as chan
import shutil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



# Extract the options
opt = TestOptions().parse()

# For testing  the neural networks, manually edit/add options below
opt.gan_mode = 'none'       # 'wgangp', 'lsgan', 'vanilla', 'none'

opt.n_downsample= 2           # Downsample times 
opt.n_blocks = 2              # Numebr of residual blocks
opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder

# Set the input dataset
opt.dataset_mode = 'CIFAR10'   # Current dataset:  CIFAR10, CelebA

# Set up the training procedure
opt.batchSize = 1           # batch size

opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
opt.norm_EG = 'batch'

if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=2)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './data/celeba/CelebA_test'
    opt.load_size = 80
    opt.crop_size = 64
    opt.size = 64
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')



##################################################################################################
opt.C_channel = 12           # The output channel number of encoder (Important: it controls the rate)
opt.SNR = 20
opt.is_clip = True
opt.CR = 1.2 
opt.is_feedback = False

# IMPLICIT: connect everything directly to the decoder networks
# PLAIN: use the concept of channel estimation and equalization to guide the neural networks
# RESIDUAL1: our usual residual connection
# RESIDUAL2: a small modification to RESIDUAL1 
opt.feedforward = 'RESIDUAL1'   
##############################################################################################################

if not opt.is_clip:
    opt.CR = 0

########################################  OFDM setting  ###########################################

size_after_compress = (opt.size//(opt.n_downsample**2))**2 * (opt.C_channel//2)

opt.N = opt.batchSize                       # Batch size
opt.P = 1                                   # Number of symbols
opt.M = 64                                  # Number of subcarriers per symbol
opt.K = 16                                  # Length of CP
opt.L = 8                                   # Number of paths
opt.decay = 4
opt.S = size_after_compress//opt.M          # Number of packets

opt.is_cfo = False
opt.is_trick = True
opt.is_cfo_random = False
opt.max_ang = 1.7
opt.ang = 1.7

opt.N_pilot = 2   # Number of pilots for chanenl estimation

opt.CE = 'LMMSE'  # Channel Estimation Method
opt.EQ = 'MMSE'   # Equalization Method
opt.pilot = 'QPSK'


if opt.CE not in ['LS', 'LMMSE', 'TRUE']:
    raise Exception("Channel estimation method not implemented")

if opt.EQ not in ['ZF', 'MMSE']:
    raise Exception("Equalization method not implemented")

if opt.feedforward not in ['IMPLICIT', 'PLAIN', 'RESIDUAL1', 'RESIDUAL2', 'RESIDUAL3']:
    raise Exception("Forward method not implemented")


# Display setting
opt.checkpoints_dir = './Checkpoints/'+ opt.dataset_mode + '_OFDM'
opt.name = opt.gan_mode + '_C' + str(opt.C_channel) + '_' + opt.feedforward + '_feed_' + str(opt.is_feedback) + '_clip_' + str(opt.CR) + '_SNR_' + str(opt.SNR)


output_path = './Images/' +  opt.dataset_mode + '_OFDM/' + opt.name

# opt.is_clip = False
# Choose the neural network model
opt.model = 'StoGANOFDM'


opt.num_test = 10000
opt.how_many_channel = 5
opt.N = opt.how_many_channel
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

model.eval()

if os.path.exists(output_path) == False:
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)


PSNR_list = []
SSIM_list = []
H_err_list = []
x_err_list = []
for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode == 'CIFAR10':
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_input(input.repeat(opt.how_many_channel,1,1,1)) 
    model.forward()
    fake = model.fake
    
    #H_err, x_err = model.MSE_calculation()
   #H_err, x_err = 0, 0
    #H_err_list.append(torch.mean(H_err).item())
    #x_err_list.append(torch.mean(x_err).item())
    #print('CE: %.4f, EQ: %.4f' % (torch.mean(H_err).item(), torch.mean(x_err).item()))
    

    #fake = fake - torch.mean(fake,(-2,-1),True) + torch.mean(input.cuda(), (-2,-1),True)
    # Get the int8 generated images
    img_gen_numpy = fake.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8) 

    origin_numpy = input.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)
    

    diff = np.mean((np.float64(img_gen_int8)-np.float64(origin_int8))**2, (1,2,3))

    PSNR = 10*np.log10((255**2)/diff)        
    PSNR_list.append(np.mean(PSNR))

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False) # return (N,)
    #ms_ssim_val = ms_ssim(img_gen_tensor,origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False ) #(N,)
    SSIM_list.append(torch.mean(ssim_val))

    # Save the first sampled image
    save_path = output_path + '/' + str(i) + '_PSNR_' + str(PSNR[0]) +'_SSIM_' + str(ssim_val[0])+'.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path = output_path + '/' + str(i) + '.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)
    #print('PSNR: %.3f, SSIM: %.3f' % (PSNR_list[-1], SSIM_list[-1]))
    if i%100 == 0:
        print(i)


print('PSNR: '+str(np.mean(PSNR_list)))
print('SSIM: '+str(np.mean(SSIM_list)))
print('MSE CE: '+str(np.mean(H_err_list)))
print('MSE EQ: '+str(np.mean(x_err_list)))
