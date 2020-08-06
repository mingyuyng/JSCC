# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .Audio_VGG_Extractor import Audio_VGGLoss


class JSCCModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']
 
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        self.model_names = ['E', 'G']
        

        # define networks (both generator and discriminator)
        self.netE = networks.define_JSCC_E(C_channel=opt.C_channel, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.netG =  networks.define_JSCC_G(C_channel=opt.C_channel, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.SNR = opt.SNR
            
            self.sigma = 1/np.sqrt(10**(0.1*self.SNR)) 

        self.normalize = networks.Normalize()
        self.criterionL2 = torch.nn.MSELoss()

        self.sample = opt.sample


    def name(self):
        return 'JSCC_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device).repeat(self.sample,1,1,1)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = self.normalize(latent.to(self.device),1)

    def set_img_path(self, path):
        self.image_paths = path
        
    def forward(self):

        # Generate latent vector
        latent = self.netE(self.real_A*0.5+0.5)        
        # Normalization
        latent = self.normalize(latent, 1)

        if self.SNR is None:
            self.fake = self.netG(latent)
        else:
            latent = latent.repeat(self.sample,1,1,1)
            noise = self.sigma * torch.randn_like(latent)
            latent_input = latent+noise
            self.fake = self.netG(latent_input)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B)
        self.loss_G_L1 = 0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    def get_encoded(self):
    
        return self.normalize(self.netE(self.real_A*0.5+0.5), 1)

    def get_decoded(self, latent, SNR=None):
        
        if SNR is None:
            self.fake = self.netG(latent)
        else:
            sigma = 1/np.sqrt(10**(0.1*SNR))
            noise = sigma * torch.randn_like(latent)
            latent_input = latent+noise
            self.fake = self.netG(latent_input)

        return self.fake

    def get_losses(self):

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2

        return self.loss_G_L2


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

    