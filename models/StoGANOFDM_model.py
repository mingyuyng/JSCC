# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import channel
import scipy.io as sio

class StoGANOFDMModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L2', 'G_Feat', 'G_PAPR', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none':
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']

        if self.opt.feedforward in ['EXPLICIT-RES']:
             self.model_names += ['R1', 'R2']
        
        if self.opt.feedforward in ['EXPLICIT-CE-EQ', 'EXPLICIT-RES']:
            C_decode = opt.C_channel
        elif self.opt.feedforward == 'IMPLICIT':
            C_decode = opt.C_channel + self.opt.N_pilot*self.opt.P*2 + self.opt.P*2
        elif self.opt.feedforward == 'EXPLICIT-CE':
            C_decode = opt.C_channel + self.opt.P*2
        
        if self.opt.is_feedback:
            add_C = self.opt.P*2
        else:
            add_C = 0
        
        # define networks (both generator and discriminator)
        self.netE = networks.define_OFDM_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, first_add_C=add_C)

        self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=C_decode, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)

        #if self.isTrain and self.is_GAN:  # define a discriminator; 
        if self.opt.gan_mode != 'none':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, 
                                          opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.opt.feedforward in ['EXPLICIT-RES']:
            self.netR1 = networks.define_RES(dim=(self.opt.N_pilot*self.opt.P+1)*2, dim_out=self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            self.netR2 = networks.define_RES(dim=(self.opt.S+1)*self.opt.P*2, dim_out=self.opt.S*self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        
        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())

            if self.opt.feedforward in ['EXPLICIT-RES']:
                params+=list(self.netR1.parameters()) + list(self.netR2.parameters())
            
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        self.opt = opt
        self.normalize = networks.Normalize()
        # Initialize the pilots and OFDM channel
        #self.pilot = torch.load('util/pilot.pt')
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)
        
        #self.cof, _ = self.channel.sample(opt.N)
        
    def name(self):
        return 'StoGANOFDM_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = self.normalize(latent.to(self.device),1)

    def set_img_path(self, path):
        self.image_paths = path
        
    def forward(self):

        N = self.real_A.shape[0]
        # Generate latent vector
        if self.opt.is_feedback:
            cof, _ = self.channel.sample(N)
            H = self.channel.get_channel_estimation(self.opt.CE, self.opt.SNR, N, cof)
            if torch.cuda.is_available():
                H = H.to(self.device)               
            latent = self.netE(self.real_A, H)
        else:
            cof = None
            latent = self.netE(self.real_A)
            
        self.tx = latent.view(self.opt.N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)

        # Normalization is contained in the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR1, self.PAPR2 = self.channel(self.tx, SNR=self.opt.SNR, cof=cof)
        N, C, H, W = latent.shape

        if self.opt.feedforward == 'IMPLICIT':
            r1 = self.channel.pilot.repeat(N,1,1,1,1)
            r2 = out_pilot
            r3 = out_sig
            dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            self.fake = self.netG(dec_in)
        else:
            # Channel estimation
            if self.opt.CE == 'LS':
                self.H_est = channel.LS_channel_est(self.channel.pilot, out_pilot)
            elif self.opt.CE == 'LMMSE':
                self.H_est = channel.LMMSE_channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)
            elif self.opt.CE == 'TRUE':
                self.H_est = H_true.unsqueeze(2).to(self.device)
            else:
                raise NotImplementedError('The channel estimation method [%s] is not implemented' % CE)
                        
            N, C, H, W = latent.shape

            # The first residual connection if required
            if self.opt.feedforward in ['EXPLICIT-RES']:
                r1_input = torch.cat((self.channel.pilot.repeat(N,1,1,1,1), out_pilot), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
                r1_output = self.netR1(r1_input).view(N, self.opt.P, 1, 2, self.opt.M).permute(0,1,2,4,3)
            
            if self.opt.feedforward in ['EXPLICIT-CE-EQ', 'EXPLICIT-CE']:
                self.eq_in = self.H_est
            elif self.opt.feedforward == 'EXPLICIT-RES':
                self.eq_in = self.H_est + r1_output
                r2_input = self.eq_in  
            
            if self.opt.feedforward == 'EXPLICIT-CE':
                dec_in = torch.cat((self.eq_in, out_sig), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
                self.fake = self.netG(dec_in)
            else:
                # Equalization
                if self.opt.EQ == 'ZF':
                    self.rx = channel.ZF_equalization(self.eq_in, out_sig)
                elif self.opt.EQ == 'MMSE':
                    self.rx = channel.MMSE_equalization(self.eq_in, out_sig, self.opt.M*noise_pwr)
                elif self.opt.EQ == 'None':
                    self.rx = None
                else:
                    raise NotImplementedError('The equalization method [%s] is not implemented' % CE)
            
                # The second residual connection if required
                if self.opt.feedforward in ['EXPLICIT-RES']:
                    r2_input = torch.cat((r2_input, out_sig), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
                    r2_output = self.netR2(r2_input).view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
                    self.rx += r2_output
            
                self.fake = self.netG(self.rx.permute(0,1,2,4,3).contiguous().view(latent.shape))
         

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

            if self.is_Feat:
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
        self.loss_G_PAPR = torch.mean(self.PAPR2)

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

    def get_channel(self):
        cof, _ = self.channel.sample()
        return cof
    

    def MSE_calculation(self):
        H_err = torch.mean((self.H_est-self.H_true.unsqueeze(2).cuda())**2, (-2,-1))*2
        x_err = torch.mean((self.rx-channel.Normalize(self.tx, 1)[0])**2, (-3,-2,-1))*2
    
        return H_err, x_err


    def get_encoded(self, cof):
        pass

    def get_pass_channel(self, latent, cof):
        pass

    def get_decoded(self, latent):
        pass

