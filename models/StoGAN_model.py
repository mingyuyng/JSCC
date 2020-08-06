# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks



class StoGANModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L2', 'G_Feat', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none':
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']

        # define networks (both generator and discriminator)
        self.netE = networks.define_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel)

        self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)

        if self.opt.gan_mode != 'none':  # define a discriminator; 
        
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, 
                                        opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids)
        

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt.label_smooth, 1-opt.label_smooth).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

            

        self.normalize = networks.Normalize()
        self.opt = opt

        if opt.channel == 'awgn':
            self.channel = networks.awgn_channel(opt)
        elif opt.channel == 'bsc':
            self.channel = networks.bsc_channel(opt)

    def name(self):
        return 'StoGAN_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = latent.to(self.device)

    def set_img_path(self, path):
        self.image_paths = path
        
    def forward(self):

        # Generate latent vector
        latent = self.netE(self.real_A)
        
        if self.opt.channel == 'awgn':   # AWGN channel
            self.latent = self.normalize(latent, 1)
        elif self.opt.channel == 'bsc':   # BSC channel
            self.latent = torch.sigmoid(latent)

        # 2. Pass the channel
        latent_input = self.channel(self.latent)

        # 3. Reconstruction
        self.fake = self.netG(latent_input)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        
        _, pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_data = self.real_B
        _, pred_real = self.netD(real_data)
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

        if self.opt.gan_mode != 'none':
            feat_fake, pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            if self.opt.is_Feat:
                feat_real, pred_real = self.netD(self.real_B)
                self.loss_G_Feat = 0
                for j in range(len(feat_real)):
                    self.loss_G_Feat += self.criterionFeat(feat_real[j].detach(), feat_fake[j]) * self.opt.lambda_feat
            else:
                self.loss_G_Feat = 0     
        
        else:
            self.loss_G_GAN = 0
            self.loss_G_Feat = 0 

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan_mode != 'none':
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
        # update G
        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    def get_encoded(self):
    
        return self.netE(self.real_A)

    def get_decoded(self, latent, SNR=None):
        
        if SNR is None:
            self.fake = self.netG(latent)
        else:
            sigma = 1/np.sqrt(10**(0.1*SNR))
            noise = sigma * torch.randn_like(latent)
            if self.is_Normalize:
                latent_input = self.normalize(latent+noise, 1)
            else:
                latent_input = latent+noise
            self.fake = self.netG(latent_input)

        return self.fake

    def get_losses(self):

        self.loss_G_L1 = self.criterionL1(self.fake, self.real_B) * self.opt.lambda_L1
        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2

        return self.loss_G_L1, self.loss_G_L2

    def update_SNR(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        
        if epoch < self.opt.n_epochs:
            self.sigma += self.sigma_step
        print('Noise pwr = %.7f' % self.sigma)
#    def forward(self, label, image, infer=False, ADMM=False, SNR=None):
#        # Encode Inputs
#        input_label, real_image = self.encode_input(label, image)

        # Fake Generation
#        input_concat = input_label
#        Compressed_p = self.netE.forward(input_concat)
        
#        latent_norm = self.normalize(Compressed_p, 1)

#        if SNR is None:
#            fake_image = self.netG.forward(latent_norm)
#        else:
#            sigma = 1/np.sqrt(10**(0.1*SNR))
#            noise = sigma * torch.randn_like(latent_norm)
#            latent_input = self.normalize(latent_norm+noise, 1)
#            fake_image = self.netG.forward(latent_input)

        # Fake Detection and Loss
#        pred_fake_pool = self.discriminate(fake_image, use_pool=True)
#        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
#        pred_real = self.discriminate(real_image)
#        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
 #       pred_fake = self.netD.forward(fake_image)
#        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
#        loss_G_GAN_Feat = 0
#        if not self.opt.no_ganFeat_loss:
#            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
#            D_weights = 1.0 / self.opt.num_D
#            for i in range(self.opt.num_D):
#                for j in range(len(pred_fake[i]) - 1):
#                    loss_G_GAN_Feat += D_weights * feat_weights * \
#                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
#        loss_G_VGG = 0
#        if not self.opt.no_vgg_loss:
#            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
#        loss_mse = 0
#        if not self.opt.no_mse_loss:
#            loss_mse = self.criteraion_mse(fake_image, real_image) * self.opt.lambda_mse
#        # Only return the fake_B image if necessary to save BW
#        if ADMM == False:
#            return [[loss_G_GAN, loss_G_GAN_Feat, loss_mse, loss_G_VGG, loss_D_real, loss_D_fake], None if not infer else fake_image]
#        else:
#            return [[loss_G_GAN, loss_G_GAN_Feat, loss_mse, loss_G_VGG, loss_D_real, loss_D_fake], None if not infer else fake_image, Compressed_p]

    