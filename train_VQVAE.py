# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
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


# Extract the options
opt = TrainOptions().parse()

# For testing  the neural networks, manually edit/add options below
opt.gan_mode = 'none'       # 'wgangp', 'lsgan', 'vanilla', 'none'

opt.n_layers_D = 3
opt.label_smooth = 1          # Label smoothing factor (for lsgan and vanilla gan only)

opt.C_channel = 16            # The output channel number of encoder (Important: it controls the rate)
opt.n_downsample= 2           # Downsample times 
opt.n_blocks = 2              # Numebr of residual blocks
opt.first_kernel = 5          # The filter size of the first convolutional layer in encoder

# Set the input dataset
opt.dataset_mode = 'CIFAR10'   # Current dataset:  CIFAR10, CelebA

# Set up the training procedure
opt.batchSize = 64           # batch size
opt.n_epochs = 80           # # of epochs without lr decay
opt.n_epochs_decay = 80     # # of epochs with lr decay
opt.lr = 5e-4                # Initial learning rate
opt.lr_policy = 'linear'     # decay policy.  Availability:  see options/train_options.py
opt.beta1 = 0.5              # parameter for ADAM
opt.beta  = 1
opt.K = 16


opt.is_Feat = False      # Whether to use feature matching loss or not
opt.lambda_feat = 1


##############################################################################################################


if opt.gan_mode == 'wgangp':
    opt.norm_D = 'instance'   # Use instance normalization when using WGAN.  Available: 'instance', 'batch', 'none'
else:
    opt.norm_D = 'batch'      # Used batch normalization otherwise

opt.activation = 'sigmoid'    # The output activation function at the last layer in the decoder
opt.norm_EG = 'batch'

if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=2)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './data/celeba/CelebA_train'
    opt.load_size = 80
    opt.crop_size = 64
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')



# Display setting
opt.checkpoints_dir = './Checkpoints/'+ opt.dataset_mode + '_VQVAE'
opt.name = opt.gan_mode + '_K' + str(opt.K)

opt.display_env =  opt.dataset_mode + '_VQVAE_' + opt.name

# Choose the neural network model
opt.model = 'VQVAE'


model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
total_iters = 0                # the total number of training iterations


################ Train with the Discriminator
loss_D_list = []
loss_G_list = []

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

        if opt.dataset_mode == 'CIFAR10':
            input = data[0]
        elif opt.dataset_mode == 'CelebA':
            input = data['data']

        model.set_input(input)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

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
            torch.save(model.netEM.state_dict(), os.path.join(model.save_dir, 'Embedding.pt'))
        iter_data_time = time.time()
    
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        torch.save(model.netEM.state_dict(), os.path.join(model.save_dir, 'Embedding.pt'))
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()  
    #model.update_SNR(epoch)
import pdb; pdb.set_trace()  # breakpoint 171c2718 //


