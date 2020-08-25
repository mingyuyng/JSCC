# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks



class DCGANModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_Feat', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
       
        self.model_names = ['G', 'D']

        # define networks (both generator and discriminator)
        self.netG = networks.define_DC_G(in_channels=opt.in_channels, channels=3, activation=opt.activation_G, norm=opt.norm_G, gpu_ids=self.gpu_ids)

        self.netD = networks.define_DC_D(channels=3, activation=opt.activation_D, norm=opt.norm_D, gpu_ids=self.gpu_ids)


        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, 1.0, 0.0).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()

            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            
        self.opt = opt


    def name(self):
        return 'DCGAN_Model'

    def set_input(self, image):
        self.real_B = image.clone().to(self.device)

    def forward(self):
        self.fake = self.netG(torch.randn(self.opt.batchSize, self.opt.in_channels, 1, 1).to(self.device))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        
        self.fake = self.netG(torch.randn(self.opt.batchSize, self.opt.in_channels, 1, 1).to(self.device))
        pred_fake = self.netD(self.fake.detach())        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_data = self.real_B
        pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.opt.gan_mode == 'wgangp':
            penalty, grad = networks.cal_gradient_penalty(self.netD, real_data, self.fake.detach(), self.device, type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
            self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.fake = self.netG(torch.randn(self.opt.batchSize, self.opt.in_channels, 1, 1).to(self.device))
        pred_fake = self.netD(self.fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)        
        self.loss_G_Feat = 0 

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Feat
        self.loss_G.backward()

    def optimize_parameters(self):
        #self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def sample_images(self, N):
        latent = torch.randn(N, self.opt.in_channels, 1, 1).to(self.device)
        return self.netG(latent)

    def sample_images_(self, latent):
        return self.netG(latent.to(self.device))