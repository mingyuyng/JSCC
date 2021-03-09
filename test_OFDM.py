# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from data import create_dataset
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

from torch.nn import init
import sys
from torch.autograd import Variable


seed_list = [10086]
PSNR_list_l = []
SSIM_list_l = []
CE_new_list_l = []
CE_old_list_l = []
PAPR_list_l = []

#forward_list = ['IMPLICIT', 'EXPLICIT-CE-EQ', 'EXPLICIT-CE', 'EXPLICIT-RES']
forward_list = ['EXPLICIT-RES']
for se in range(len(forward_list)):

    torch.manual_seed(seed_list[0])
    np.random.seed(seed_list[0])

    # Extract the options
    opt = TestOptions().parse()

    # For testing  the neural networks, manually edit/add options below
    opt.gan_mode = 'none'       # 'wgangp', 'lsgan', 'vanilla', 'none'

    # Set the input dataset
    opt.dataset_mode = 'CIFAR10'   # Current dataset:  CIFAR10, CelebA

    if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
        opt.n_layers_D = 3
        opt.n_downsample = 2          # Downsample times
        opt.n_blocks = 2              # Numebr of residual blocks
        opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
        # Initial learning rate
    elif opt.dataset_mode == 'CelebA':
        opt.n_layers_D = 3
        opt.n_downsample = 3          # Downsample times
        opt.n_blocks = 2              # Numebr of residual blocks
        opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder

    elif opt.dataset_mode == 'OpenImage':
        opt.n_layers_D = 3
        opt.n_downsample = 4          # Downsample times
        opt.n_blocks = 2              # Numebr of residual blocks
        opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder

    # Set up the training procedure
    opt.batchSize = 1           # batch size

    opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
    opt.norm_EG = 'batch'

    if opt.dataset_mode == 'CIFAR10':
        opt.dataroot = './data'
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

    elif opt.dataset_mode == 'CIFAR100':
        opt.dataroot = './data'
        opt.size = 32
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
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
        opt.serial_batches = True
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)
        print('#training images = %d' % dataset_size)
    else:
        raise Exception('Not implemented yet')

    opt.iter_temp = 5000
    ##################################################################################################
    # Set up the training procedure
    opt.C_channel = 12
    opt.SNR = 15

    opt.is_feedback = False
    opt.feedforward = forward_list[se]

    opt.N_pilot = 1         # Number of pilots for chanenl estimation
    opt.CE = 'MMSE'         # Channel Estimation Method
    opt.EQ = 'MMSE'         # Equalization Method
    opt.pilot = 'ZadoffChu'      # QPSK or ZadoffChu

    opt.lam_h = 0.5
    opt.is_hloss = True

    opt.is_clip = False
    opt.CR = 0 if not opt.is_clip else 1.2
    opt.is_regu_PAPR = False
    opt.lam_PAPR = 0.1
    opt.is_random = False
    opt.lam_G = 0.2
    ##############################################################################################################

    ########################################  OFDM setting  ###########################################

    if opt.gan_mode == 'wgangp':
        opt.norm_D = 'instance'   # Use instance normalization when using WGAN.  Available: 'instance', 'batch', 'none'
    else:
        opt.norm_D = 'batch'      # Used batch normalization otherwise

    size_after_compress = (opt.size // (2**opt.n_downsample))**2 * (opt.C_channel // 2)

    opt.N = opt.batchSize                       # Batch size
    opt.P = 1                                   # Number of symbols
    opt.M = 64                                  # Number of subcarriers per symbol
    opt.K = 16                                  # Length of CP
    opt.L = 8                                   # Number of paths
    opt.decay = 4
    opt.S = size_after_compress // opt.M          # Number of packets

    opt.is_cfo = False
    opt.is_trick = True
    opt.is_cfo_random = False
    opt.max_ang = 1.7
    opt.ang = 1.7

    if opt.CE not in ['LS', 'MMSE', 'TRUE']:
        raise Exception("Channel estimation method not implemented")

    if opt.EQ not in ['ZF', 'MMSE']:
        raise Exception("Equalization method not implemented")

    # if opt.feedforward not in ['IMPLICIT', 'EXPLICIT-CE', 'EXPLICIT-CE-EQ', 'EXPLICIT-RES']:
    #    raise Exception("Forward method not implemented")

    # Display setting
    opt.checkpoints_dir = './Checkpoints/' + opt.dataset_mode + '_OFDM'
    opt.name = 'C' + str(opt.C_channel) + '_' + opt.feedforward + '_SNR_' + str(opt.SNR) + '_pilot_' + str(opt.N_pilot) + '_' + str(opt.is_hloss)

    if opt.is_clip:
        opt.name += '_clip_' + str(opt.CR)

    if opt.is_regu_PAPR:
        opt.name += '_PAPRnet_' + str(opt.lam_PAPR)

    if opt.is_random:
        opt.name += '_random'
    if opt.gan_mode in ['lsgan', 'vanilla', 'wgangp']:
        opt.name += f'_{opt.gan_mode}_{opt.lam_G}'

    output_path = './Images/' + opt.dataset_mode + '_OFDM/' + opt.name

    # Choose the neural network model
    opt.model = 'StoGANOFDM'

    opt.num_test = 10000
    opt.how_many_channel = 5
    opt.N = opt.how_many_channel

    opt.is_clip = True
    opt.CR = 0 if not opt.is_clip else 1.4

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
    H_err_prev_list = []
    H_err_list = []
    PAPR_list = []

    PSNR_refine_list = []
    SSIM_refine_list = []

    REFINE = 0

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        start_time = time.time()
        
        if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data['data']

        model.set_input(input.repeat(opt.how_many_channel, 1, 1, 1))
        model.forward()
        fake = model.fake
        PAPR = torch.mean(10 * torch.log10(model.PAPR))
        PAPR_list.append(PAPR.item())

        if opt.feedforward in ['EXPLICIT-RES', 'EXPLICIT-RES-CE', 'EXPLICIT-CE-EQ']:
            H_err_new, H_err_old = model.MSE_calculation()
            H_err_list.append(torch.mean(H_err_new).item())
            H_err_prev_list.append(torch.mean(H_err_old).item())
        else:
            H_err_list.append(0)
            H_err_prev_list.append(0)

        # Get the int8 generated images
        img_gen_numpy = fake.detach().cpu().float().numpy()
        img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        img_gen_int8 = img_gen_numpy.astype(np.uint8)

        origin_numpy = input.detach().cpu().float().numpy()
        origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        origin_int8 = origin_numpy.astype(np.uint8)

        diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

        PSNR = 10 * np.log10((255**2) / diff)
        PSNR_list.append(np.mean(PSNR))

        img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
        origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

        ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(opt.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # return (N,)
        # ms_ssim_val = ms_ssim(img_gen_tensor,origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False ) #(N,)
        SSIM_list.append(torch.mean(ssim_val).item())

        # Save the first sampled image
        save_path = output_path + '/' + str(i) + '_PSNR_' + str(PSNR[0]) + '_SSIM_' + str(ssim_val[0]) + '.png'
        util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

        save_path = output_path + '/' + str(i) + '.png'
        util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)

        if i % 100 == 0:
            print(i)

    PSNR_list_l.append(np.mean(PSNR_list))
    SSIM_list_l.append(np.mean(SSIM_list))
    CE_new_list_l.append(1000 * np.mean(H_err_list))
    CE_old_list_l.append(1000 * np.mean(H_err_prev_list))
    PAPR_list_l.append(np.mean(PAPR_list))

    opt = None

print(f'PSNR: { {*PSNR_list_l} }')
print(f'SSIM: { {*SSIM_list_l} }')
print(f'MSE_old: { {*CE_old_list_l} }')
print(f'MSE_new: { {*CE_new_list_l} }')
print(f'PAPR: { {*PAPR_list_l} }')
