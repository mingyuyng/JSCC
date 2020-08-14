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

opt.C_channel = 16             # The output channel number of encoder (Important: it controls the rate)
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

########################################  OFDM setting  ###########################################

size_after_compress = (opt.size//(opt.n_downsample**2))**2 * (opt.C_channel//2)

opt.N = opt.batchSize                       # Batch size
opt.S = 1                                   # Number of symbols
opt.M = 64                                  # Number of subcarriers per symbol
opt.K = 16                                  # Length of CP
opt.L = 8                                   # Number of paths
opt.decay = 4
opt.P = size_after_compress//opt.M          # Number of packets

opt.is_clip = False
opt.PAPR = 5

opt.is_cfo = False
opt.is_trick = True
opt.is_cfo_random = False
opt.max_ang = 1.7
opt.ang = 1.7

opt.is_feedback = False

opt.is_pilot = False
opt.N_pilot = 0   # Set to 0 if opt.is_pilot is false

opt.SNR = 15

opt.CE = 'LMMSE'  # Channel Estimation Method
opt.EQ = 'MMSE+'   # Equalization Method

if opt.CE not in ['LS', 'LMMSE', 'TRUE']:
    raise Exception("Channel estimation method not implemented")

if opt.EQ not in ['ZF', 'MMSE', 'IMPLICIT', 'MMSE+', 'ZF+']:
    raise Exception("Equalization method not implemented")

# Display setting
opt.checkpoints_dir = './Checkpoints/'+ opt.dataset_mode + '_OFDM'
opt.name = opt.gan_mode + '_C' + str(opt.C_channel) + '_' + opt.CE + '_' + opt.EQ + '_feed_' + str(opt.is_feedback) + '_clip_' + str(opt.is_clip) + '_SNR_' + str(opt.SNR)

output_path = './Images/' +  opt.dataset_mode + '_OFDM/' + opt.name


# Choose the neural network model
opt.model = 'StoGANOFDM'


opt.num_test = 2000
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
for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode == 'CIFAR10':
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_encode(input.repeat(opt.how_many_channel,1,1,1))       # unpack data from data loader
    if opt.is_feedback:
        cof, _ = model.channel.channel.sample(opt.how_many_channel, opt.P, opt.M, opt.L)  # Sample multipath channels
    else:
        cof = None

    latent = model.get_encoded(cof)
    
    out_pilot, out_sig, H_true, noise_pwr = model.get_pass_channel(latent, cof)

    # Channel estimation
    if opt.CE == 'LS':
        H_est = chan.LS_channel_est(model.channel.pilot, out_pilot)
    elif opt.CE == 'LMMSE':
        H_est = chan.LMMSE_channel_est(model.channel.pilot, out_pilot, opt.M*noise_pwr)
    elif opt.CE == 'TRUE':
        H_est = H_true.unsqueeze(2)

    # Equalization and decode
    if opt.EQ == 'ZF':
        rx = chan.ZF_equalization(H_est, out_sig)
        fake = model.netG((rx.view(latent.shape)))
    elif opt.EQ == 'MMSE':
        rx = chan.MMSE_equalization(H_est, out_sig, opt.M*noise_pwr)
        #fake = model.netG(rx.view(latent.shape))
        fake = model.netG((rx.view(latent.shape)))
    elif opt.EQ == 'IMPLICIT':
        N, C, H, W = latent.shape
        dec_in = torch.cat((H_est, out_sig), 1).view(N, -1, H, W)
        fake = model.netG(dec_in)
    elif opt.EQ == 'MMSE+':
        rx = chan.MMSE_equalization(H_est, out_sig, opt.M*noise_pwr)
        fake = model.netG(torch.cat((rx.view(latent.shape), H_est.view(latent.shape),out_pilot.view(latent.shape)), 1))
    elif opt.EQ == 'ZF+':
        rx = chan.ZF_equalization(H_est, out_sig)
        fake = model.netG(torch.cat((rx.view(latent.shape), H_est.view(latent.shape),out_pilot.view(latent.shape)), 1))

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
    print(i)

print('PSNR: '+str(np.mean(PSNR_list)))
print('SSIM: '+str(np.mean(SSIM_list)))

