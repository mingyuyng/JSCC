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
from cifar10_dcgan.dcgan import Discriminator, Generator
from torch.nn import init
import sys
from torch.autograd import Variable


def complex_multiplication(x1, x2):
    real1 = x1[..., 0]
    imag1 = x1[..., 1]
    real2 = x2[..., 0]
    imag2 = x2[..., 1]

    out_real = real1 * real2 - imag1 * imag2
    out_imag = real1 * imag2 + imag1 * real2

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)), -1)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# Extract the options
opt = TestOptions().parse()

# For testing  the neural networks, manually edit/add options below
opt.gan_mode = 'none'       # 'wgangp', 'lsgan', 'vanilla', 'none'


# Set the input dataset
opt.dataset_mode = 'CelebA'   # Current dataset:  CIFAR10, CelebA

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

if opt.dataset_mode in 'CIFAR10':
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

elif opt.dataset_mode in 'CIFAR100':
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
opt.SNR =0

opt.is_feedback = False
opt.feedforward = 'EXPLICIT-CE-EQ'

opt.N_pilot = 2         # Number of pilots for chanenl estimation
opt.CE = 'MMSE'         # Channel Estimation Method
opt.EQ = 'MMSE'         # Equalization Method
opt.pilot = 'ZadoffChu'      # QPSK or ZadoffChu

opt.is_clip = False
opt.CR = 0 if not opt.is_clip else 1
opt.is_regu_PAPR = False
opt.lam_PAPR = 0.3
##############################################################################################################


########################################  OFDM setting  ###########################################

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
opt.name = '_C' + str(opt.C_channel) + '_' + opt.feedforward + '_SNR_' + str(opt.SNR)

if opt.is_clip:
    opt.name +=  '_clip_' + str(opt.CR)

if opt.is_regu_PAPR:
    opt.name +=  '_PAPRnet_' + str(opt.lam_PAPR)

output_path = './Images/' + opt.dataset_mode + '_OFDM/' + opt.name


# Choose the neural network model
opt.model = 'StoGANOFDM'

opt.num_test = 10000
opt.how_many_channel = 1
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

PSNR_refine_list = []
SSIM_refine_list = []

REFINE = 0

for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode == 'CIFAR10':
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_input(input.repeat(opt.how_many_channel, 1, 1, 1))
    model.forward()
    fake = model.fake

    #H_err_new, H_err_old = model.MSE_calculation()
    #H_err, x_err = 0, 0
    # H_err_list.append(torch.mean(H_err_new).item())
    # H_err_prev_list.append(torch.mean(H_err_old).item())
    # print(H_err_list)
    # print(x_err_list)
    #print('CE: %.4f, EQ: %.4f' % (torch.mean(H_err).item(), torch.mean(x_err).item()))
    #sio.savemat('H_estimations' + str(i) + '.mat', {'H_true': model.H_true.detach().cpu().numpy(), 'H_est': model.H_est.detach().cpu().numpy(), 'H_est_new': model.H_est_new.detach().cpu().numpy()})
    # import pdb; pdb.set_trace()  # breakpoint 2088e5e3 //

    #fake = fake - torch.mean(fake,(-2,-1),True) + torch.mean(input.cuda(), (-2,-1),True)
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
    SSIM_list.append(torch.mean(ssim_val))

    # Save the first sampled image
    save_path = output_path + '/' + str(i) + '_PSNR_' + str(PSNR[0]) + '_SSIM_' + str(ssim_val[0]) + '.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path = output_path + '/' + str(i) + '.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)

    #print('PSNR: %.3f, SSIM: %.3f' % (PSNR_list[-1], SSIM_list[-1]))

    #if REFINE == 1:
    #    lam = 1
    #    lr = 0.005
    #    in_channels = 100
    #    iteration = 500

    #    latent = Variable(torch.randn(1, in_channels, 1, 1).cuda(), requires_grad=False)

    #    psnr_list = []
    #    ssim_list = []

    #    for k in range(opt.how_many_channel):
    #        G = Generator(ngpu=1, nz=in_channels).to(model.device)
    #        init_weights(G, init_type='normal')

    #        optmize_Com = torch.optim.Adam(G.parameters(), lr=lr)
    #        Criterion = torch.nn.MSELoss().cuda()

    #        H = model.H_est[k].unsqueeze(0).detach()
    #        latent_rx = model.out_sig[k].unsqueeze(0)
    #        decode_img = fake[k].unsqueeze(0)

    #        for j in range(iteration):
    '''
                optmize_Com.zero_grad()

                gen_img = G(latent)
                mse_loss_content = Criterion(gen_img, decode_img.detach())

                latent_tx = model.get_encoded(gen_img)
                latent_rx_est = complex_multiplication(H, latent_tx)
                mse_loss_latent = Criterion(latent_rx_est, latent_rx.detach())

                mse_loss = mse_loss_content + lam * mse_loss_latent
                mse_loss.backward()
                optmize_Com.step()
                #print('Fake MSE: %f, latent MSE: %f' % (mse_loss_content.item(), mse_loss_latent.item()))

            fake_new = G(latent)
            # Get the int8 generated images
            img_gen_numpy_new = fake_new.detach().cpu().float().numpy()
            img_gen_numpy_new = (np.transpose(img_gen_numpy_new, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            img_gen_int8_new = img_gen_numpy_new.astype(np.uint8)

            diff = np.mean((np.uint64(img_gen_int8_new) - np.uint64(origin_int8))**2, (1, 2, 3))
            psnr_list.append(10 * np.log10((255**2) / diff))
            #print('Initial PSNR: %.3fdB, Initial SSIM: %.3f' % (PSNR[i,0], SSIM[i,0]))
            #print('The current PSNR is %.3f' % (psnr_list[-1]))

            img_gen_tensor_new = torch.from_numpy(np.transpose(img_gen_int8_new, (0, 3, 1, 2))).float()
            ssim_val = ssim(img_gen_tensor_new, origin_tensor, data_range=255, size_average=False)  # return (N,)
            ssim_list.append(ssim_val)
            #print('The current SSIM is %.3f' % (ssim_val[-1]))

            if k == 0:
                # Save the first sampled image
                save_path = output_path + '/' + str(i) + '_LAM_' + str(lam) + '.png'
                util.save_image(util.tensor2im(fake_new), save_path, aspect_ratio=1)

            PSNR_refine_list.append(np.mean(psnr_list))
            SSIM_refine_list.append(np.mean(ssim_list))

            print('Before: PSNR: %.3f, SSIM: %.3f' % (PSNR_list[-1], SSIM_list[-1]))
            print('After: PSNR: %.3f, SSIM: %.3f' % (PSNR_refine_list[-1], SSIM_refine_list[-1]))
    '''

    if i % 100 == 0:
        print(i)


print('PSNR: ' + str(np.mean(PSNR_list)))
print('SSIM: ' + str(np.mean(SSIM_list)))
print('MSE new: ' + str(np.mean(H_err_list)))
print('MSE old: ' + str(np.mean(H_err_prev_list)))

print('PSNR after: ' + str(np.mean(PSNR_refine_list)))
print('SSIM after: ' + str(np.mean(SSIM_refine_list)))
