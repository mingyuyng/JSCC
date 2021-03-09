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
from CIFAR_10.models import *

seed_list = [10086]
ACC_list_l = []


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
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)
        print('#training images = %d' % dataset_size)
    else:
        raise Exception('Not implemented yet')

    opt.iter_temp = 5000
    ##################################################################################################
    # Set up the training procedure
    opt.C_channel = 12
    opt.SNR = 5

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
    opt.lam_G = 1
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

    # Load the classification network
    net = ResNet50()
    checkpoint = torch.load('./CIFAR_10/checkpoint/ckpt.pth')

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    net.eval()

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    correct = 0

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        start_time = time.time()

        if opt.dataset_mode == 'CIFAR10' or 'CIFAR100':
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data['data']

        model.set_input(input.repeat(opt.how_many_channel, 1, 1, 1))
        model.forward()
        fake = model.fake

        with torch.no_grad():
            outputs = net(fake)
            _, predicted = outputs.max(1)

        correct += predicted.eq(data[1].cuda()).sum().item()

        if i % 100 == 0:
            print(i)

    accuracy = correct / (opt.num_test * opt.how_many_channel)
    ACC_list_l.append(accuracy)
    opt = None


print(f'ACC: { {*ACC_list_l} }')
