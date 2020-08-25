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
import util.inception_score as incep

# Extract the options
opt = TestOptions().parse()

opt.in_channels = 100            # The output channel number of encoder (Important: it controls the rate)

# Set the input dataset
opt.dataset_mode = 'CIFAR10'   # Current dataset:  CIFAR10, CelebA

# Set up the training procedure
opt.batchSize = 64           # batch size
opt.n_epochs = 300           # # of epochs without lr decay
opt.n_epochs_decay = 300     # # of epochs with lr decay
opt.lr = 1e-3                # Initial learning rate
opt.lr_policy = 'linear'     # decay policy.  Availability:  see options/train_options.py
opt.beta1 = 0.5              # parameter for ADAM

# Set up the loss function
opt.is_Feat = False      # Whether to use feature matching loss or not
opt.lambda_feat = 1
opt.gan_mode = 'wgangp'

##############################################################################################################
if opt.gan_mode == 'wgangp':
    opt.norm_D = 'instance'   # Use instance normalization when using WGAN.  Available: 'instance', 'batch', 'none'
else:
    opt.norm_D = 'batch'      # Used batch normalization otherwise

opt.activation_D = 'LeakyReLU'
opt.activation_G = 'ReLU'    # The output activation function at the last layer in the decoder
opt.norm_G = 'batch'


if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=2, drop_last=True)
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
opt.checkpoints_dir = './Checkpoints/'+ opt.dataset_mode + '_GAN'
opt.name = opt.gan_mode

opt.display_env =  opt.dataset_mode + '_GAN_' + opt.name

# Choose the neural network model
opt.model = 'DCGAN'

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

output_path = './Images/'+ opt.dataset_mode + '_GAN/' + opt.name

if os.path.exists(output_path) == False:
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)


######################################################################

N_test = 1000

images = model.sample_images(N_test)

for i in range(N_test):
    save_path = output_path + '/' + str(i) +'.png'
    util.save_image(util.tensor2im(images[i].unsqueeze(0)), save_path, aspect_ratio=1)
     
mean, std = incep.get_inception_score(images, batch_size=32, resize=True)

import pdb; pdb.set_trace()  # breakpoint 76bb3484 //


for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode == 'CIFAR10':
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_encode(input.repeat(opt.how_many_channel,1,1,1))       # unpack data from data loader

    latent = model.get_encoded()
    
    fake = model.get_decoded(latent)
    

    # Get the int8 generated images
    img_gen_numpy = fake.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8) 

    origin_int8 = np.expand_dims(util.tensor2im(input), axis=0)

    diff = np.mean((np.uint64(img_gen_int8)-np.uint64(origin_int8))**2, (1,2,3))
    PSNR = 10*np.log10((255**2)/diff)        
    PSNR_list.append(np.mean(PSNR))
    
    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False) # return (N,)
    #ms_ssim_val = ms_ssim(img_gen_tensor,origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False ) #(N,)
    SSIM_list.append(torch.mean(ssim_val))

    # Save the first sampled image
    save_path = output_path + '/' + str(i) + '_PSNR_' + str(PSNR[0]) +'.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path = output_path + '/' + str(i) + '.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)
    print(i)

print('PSNR: '+str(np.mean(PSNR_list)))
print('SSIM: '+str(np.mean(SSIM_list)))


