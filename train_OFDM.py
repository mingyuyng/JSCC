# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Extract the options
opt = TrainOptions().parse()

# For testing  the neural networks, manually edit/add options below
opt.gan_mode = 'none'       # 'wgangp', 'lsgan', 'vanilla', 'none'

# Set the input dataset
opt.dataset_mode = 'CelebA'   # Current dataset:  CIFAR10, CelebA


if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
    opt.n_layers_D = 3
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 2          # Downsample times
    opt.n_blocks = 2              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batchsize = 128
    opt.n_epochs = 200            # # of epochs without lr decay
    opt.n_epochs_decay = 200      # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM
    opt.lr = 1e-4                 # Initial learning rate

elif opt.dataset_mode == 'CelebA':
    opt.n_layers_D = 3
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 3          # Downsample times
    opt.n_blocks = 2              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batch_size = 64
    opt.n_epochs = 30             # # of epochs without lr decay
    opt.n_epochs_decay = 30       # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM
    opt.lr = 5e-4

elif opt.dataset_mode == 'OpenImage':
    opt.n_layers_D = 3
    opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)
    opt.n_downsample = 4          # Downsample times
    opt.n_blocks = 2              # Numebr of residual blocks
    opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder
    opt.batch_size = 16
    opt.n_epochs = 30             # # of epochs without lr decay
    opt.n_epochs_decay = 30       # # of epochs with lr decay
    opt.lr_policy = 'linear'      # decay policy.  Availability:  see options/train_options.py
    opt.beta1 = 0.5               # parameter for ADAM 
    opt.lr = 5e-4


############################ Things recommanded to be changed ##########################################
# Set up the training procedure
opt.C_channel = 12
opt.SNR = 20

opt.is_feedback = False
opt.feedforward = 'EXPLICIT-RES-CE'

opt.N_pilot = 2              # Number of pilots for chanenl estimation
opt.CE = 'MMSE'              # Channel Estimation Method
opt.EQ = 'MMSE'              # Equalization Method
opt.pilot = 'ZadoffChu'      # QPSK or ZadoffChu

opt.is_clip = False
opt.CR = 0 if not opt.is_clip else 1
opt.is_regu_PAPR = False
opt.lam_PAPR = 0.3
##############################################################################################################

# Set up the loss function
opt.lambda_L2 = 128       # The weight for L2 loss
opt.is_Feat = False       # Whether to use feature matching loss or not
opt.lambda_feat = 1

if opt.gan_mode == 'wgangp':
    opt.norm_D = 'instance'   # Use instance normalization when using WGAN.  Available: 'instance', 'batch', 'none'
else:
    opt.norm_D = 'batch'      # Used batch normalization otherwise

opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
opt.norm_EG = 'batch'


if opt.dataset_mode == 'CIFAR10':
    opt.dataroot = './data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CIFAR100':
    opt.dataroot = './data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(opt.size, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './data/celeba/CelebA_train'
    opt.load_size = 80
    opt.crop_size = 64
    opt.size = 64
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'OpenImage':
    opt.dataroot = './data/opv6'
    opt.size = 256
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

else:
    raise Exception('Not implemented yet')


########################################  OFDM setting  ###########################################
size_after_compress = (opt.size // (2**opt.n_downsample))**2 * (opt.C_channel // 2)

opt.P = 1                                   # Number of symbols
opt.M = 64                                  # Number of subcarriers per symbol
opt.K = 16                                  # Length of CP
opt.L = 8                                   # Number of paths
opt.decay = 4
opt.S = size_after_compress // opt.M        # Number of packets

opt.is_cfo = False
opt.is_trick = True
opt.is_cfo_random = False
opt.max_ang = 1.7
opt.ang = 1.7

if opt.CE not in ['LS', 'MMSE', 'TRUE', 'IMPLICIT']:
    raise Exception("Channel estimation method not implemented")

if opt.EQ not in ['ZF', 'MMSE', 'IMPLICIT']:
    raise Exception("Equalization method not implemented")

if opt.feedforward not in ['IMPLICIT', 'EXPLICIT-CE-EQ', 'EXPLICIT-RES', 'EXPLICIT-RES-EQ', 'EXPLICIT-RES-CE', 'IMPLICIT_nopilot']:
    raise Exception("Forward method not implemented")

# Display setting
opt.checkpoints_dir = './Checkpoints/' + opt.dataset_mode + '_OFDM'
opt.name = '_C' + str(opt.C_channel) + '_' + opt.feedforward + '_SNR_' + str(opt.SNR)

if opt.is_clip:
    opt.name +=  '_clip_' + str(opt.CR)

if opt.is_regu_PAPR:
    opt.name +=  '_PAPRnet_' + str(opt.lam_PAPR)
   

opt.display_env = opt.dataset_mode + '_OFDM_' + opt.name

# Choose the neural network model
opt.model = 'StoGANOFDM'

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
total_iters = 0                # the total number of training iterations

# Train with the Discriminator
loss_D_list = []
loss_G_list = []
count = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        if opt.dataset_mode in ['CIFAR10', 'CIFAR100']:
            input = data[0]
        elif opt.dataset_mode == 'CelebA': 
            input = data['data']
        elif opt.dataset_mode == 'OpenImage':
            input = data['data']

        model.set_input(input)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        #count += 1
        #model.update_temp(count)

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
        iter_data_time = time.time()

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
