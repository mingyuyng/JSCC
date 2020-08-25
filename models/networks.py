# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import exp
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')
###############################################################################
# Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())        
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



class Flatten(nn.Module):
  def forward(self, x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Normalize(nn.Module):
  def forward(self, x, power):
    N = x.shape[0]
    pwr = torch.mean(x**2, (1,2,3), True)

    return np.sqrt(power)*x/torch.sqrt(pwr)  


def define_E(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_OFDM_E(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, first_add_C=0):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder_OFDM(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, first_add_C=first_add_C)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_JSCC_E(C_channel, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = JSCC_encoder(C_channel)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[], first_kernel=7, activation='sigmoid'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect", first_kernel=first_kernel, activation_=activation)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_JSCC_G(C_channel, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = JSCC_decoder(C_channel)    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'none']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##############################################################################
# Encoder
##############################################################################
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """   
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        
        model = [nn.ReflectionPad2d((first_kernel-1)//2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  
            model += [ResnetBlock(min(ngf * mult,max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]


        self.model = nn.Sequential(*model)
        self.projection = nn.Conv2d(min(ngf * mult,max_ngf), C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)
        
    def forward(self, input):
        z =  self.model(input)
        return  self.projection(z)

##############################################################################
# Encoder for OFDM
##############################################################################
class Encoder_OFDM(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, first_add_C=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """   
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder_OFDM, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        
        model = [nn.ReflectionPad2d((first_kernel-1)//2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        self.model_down = nn.Sequential(*model)
        model= []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  
            model += [ResnetBlock(min(ngf * mult,max_ngf)+first_add_C, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        
        self.model_res = nn.Sequential(*model)

        self.projection = nn.Conv2d(min(ngf * mult,max_ngf)+first_add_C, C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)
        
    def forward(self, input, H=None):

        z =  self.model_down(input)
        if H is not None:
            N,C,HH,WW = z.shape            
            z = torch.cat((z,H.contiguous().permute(0,1,2,4,3).view(N, -1, HH,WW)), 1)
        z = self.model_res(z)
        return  self.projection(z)

class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Generator, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult //2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((first_kernel-1)//2), nn.Conv2d(ngf, output_nc, kernel_size=first_kernel, padding=0)]

        if activation_ == 'tanh':
            model +=[nn.Tanh()]
        elif activation_ == 'sigmoid':
            model +=[nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.activation_=='tanh':
            return self.model(input)
        elif self.activation_=='sigmoid':
            return 2*self.model(input)-1


# Scaler soft/hard quantization

class quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
    def forward(self, x, Q_type="None"):
        if Q_type=="Soft":            
            W_stack = torch.stack([x for _ in range(len(self.center))],dim=-1)
            W_index = torch.argmin(torch.abs(W_stack-self.center),dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(-1.0*self.Temp*(W_stack-self.center)**2,dim=-1)
            W_soft = torch.einsum('ijklm,m->ijkl',[smx,self.center])
            with torch.no_grad():
                w_bias = (W_hard - W_soft)
            return w_bias + W_soft
        elif Q_type=='None':
            return x
        elif Q_type == 'Hard':
            W_stack = torch.stack([x for _ in range(len(self.center))], dim=-1)
            W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
            W_hard = self.center[W_index]
            return W_hard
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)




class quantizer_channel(nn.Module):
    def __init__(self, center, Temp, bpe, ber):
        super(quantizer_channel, self).__init__()
        
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp

        self.bpe = bpe
        self.ber = ber

        bitmap = np.array([[0],[1]])
        for i in range(bpe-1):
            l = bitmap.shape[0]
            zero = np.zeros((l,1))
            one = np.ones((l,1))
            upper = np.concatenate((zero, bitmap), axis=1)
            lower = np.concatenate((one, bitmap), axis=1)
            bitmap = np.concatenate((upper, lower), axis=0)
        
        self.map = torch.from_numpy(bitmap).float()

    def forward(self, x, Q_type="None"):
        

        W_stack = torch.stack([x for _ in range(len(self.center))],dim=-1)
        W_index = torch.argmin(torch.abs(W_stack-self.center),dim=-1)
        W_hard = self.center[W_index]
        if Q_type == 'Soft':
            smx = torch.softmax(-1.0*self.Temp*(W_stack-self.center)**2,dim=-1)
            W_soft = torch.einsum('ijklm,m->ijkl',[smx,self.center])

        # Quantized vectors to binary expression
        code_list = self.map[W_index].cuda()

        # Add channel noise
        noise = torch.bernoulli(torch.ones(code_list.shape)*self.ber).cuda()
        code_noisy = (code_list + noise) % 2

        # Re-map to vectors
        index = torch.zeros_like(x)

        for i in range(code_noisy.shape[-1]):
            index += 2**i * code_noisy[:, :, :, :, self.bpe-1-i]

        W_noisy = self.center[index.long()]

        if Q_type == 'Soft':
            with torch.no_grad():
                W_bias = (W_noisy - W_soft)

            return W_soft + W_bias
        elif Q_type == 'Hard':
            with torch.no_grad():
                W_bias = (W_noisy - x)

            return x + W_bias,  index
        else:
            return x
       
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)

#############################################################################################################
class bsc_channel(nn.Module):
    def __init__(self, opt):
        super(bsc_channel, self).__init__()
        self.opt=opt
        self.Temp = self.opt.temp
    def forward(self, x):

        # 1. Generating the probability for bernoulli distribution
        if self.opt.enc_type =='prob':
            pass
        elif self.opt.enc_type == 'hard':
            index = torch.zeros_like(x)
            index[x>0.5] = 1
            with torch.no_grad():
                bias = index - x
            x = x + bias
        elif self.opt.enc_type == 'soft':
            x = torch.sigmoid((x**2-(x-1)**2)/self.Temp)
        elif  self.opt.enc_type == 'soft_hard':
            x = torch.sigmoid((x**2-(x-1)**2)/self.Temp)
            index = torch.zeros_like(x)
            index[x>0.5] = 1
            with torch.no_grad():
                bias = index - x
            x = x + bias
        
        out_prob = self.opt.ber + x - 2*self.opt.ber*x

        # 2. Sample the bernoulli distribution and generate decoder input
        if self.opt.sample_type == 'st':
            cha_out = torch.bernoulli(out_prob.detach())
            with torch.no_grad():
                bias = cha_out - out_prob
            dec_in = out_prob + bias

        elif self.opt.sample_type == 'gumbel_softmax':
            probs = clamp_probs(out_prob)
            uniforms = clamp_probs(torch.rand_like(out_prob))
            logits = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p())/self.Temp
            dec_in = torch.sigmoid(logits)
        elif self.opt.sample_type == 'gumbel_softmax_hard':
            probs = clamp_probs(out_prob)
            uniforms = clamp_probs(torch.rand_like(out_prob))
            logits = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p())/self.Temp
            dec_in = torch.sigmoid(logits)
            index = torch.zeros_like(x)
            index[dec_in>0.5] = 1
            with torch.no_grad():
                bias = index - dec_in
            dec_in = dec_in + bias

        return dec_in
       
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    

class awgn_channel(nn.Module):
    def __init__(self, opt):
        super(awgn_channel, self).__init__()
        self.opt=opt
        self.sigma = 10**(-opt.SNR/20)

    def forward(self, x):  
          
        noise = self.sigma * torch.randn_like(x)
        dec_in = x+noise
        return dec_in

##################################################################################################################################


class vector_quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(vector_quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
    def forward(self, x,Q_type='None'):
        x_ = x.view(x.shape[0],-1,4)
        if Q_type=="Soft":
            W_stack = torch.stack([x_ for _ in range(len(self.center))],dim=-1)
            E = torch.norm(W_stack - self.center.transpose(1,0),2,dim=-2)
            W_index = torch.argmin(E,dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(-1.0*self.Temp*E**2,dim=-1)
            W_soft = torch.einsum('ijk,km->ijm',[smx,self.center])
            with torch.no_grad():
                w_bias = (W_hard - W_soft)
            output = w_bias + W_soft
        elif Q_type=='None':
            output = x_
        elif Q_type == 'Hard':
            W_stack = torch.stack([x_ for _ in range(len(self.center))], dim=-1)
            E = torch.norm(W_stack - self.center.transpose(1, 0), 2, dim=-2)
            W_index = torch.argmin(E, dim=-1)
            W_hard = self.center[W_index]
            output =  W_hard
            import pdb; pdb.set_trace()  # breakpoint 2a2b226a //

        return output.view(x.shape), W_index.view(x.shape)
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map
        
        
        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers+1):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model'+str(self.n_layers+1))
        out = model(res[-1])

        return res[1:], out
        



class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.upsample =  lambda x: torch.nn.functional.interpolate(x, scale_factor=2)
    def forward(self, x):
        return self.upsample(x)
class upsample_pad(nn.Module):
    def __init__(self):
        super(upsample_pad, self).__init__()
    def forward(self,x):
        out = torch.zeros(x.shape[0],x.shape[1],2*x.shape[2],2*x.shape[3],device = x.device,dtype = x.dtype)
        out[:,:,0::2,:][:,:,:,0::2]=x
        return out



class JSCC_encoder(nn.Module):
    def __init__(self, n_channel=8):
           
        assert(n_channel>=0)
        super(JSCC_encoder, self).__init__()

        activation = nn.PReLU()
        
        sequence = [nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2), 
                 activation,
                 nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1), 
                 activation,
                 nn.Conv2d(32, n_channel, kernel_size=5, padding=2, stride=1), 
                 activation]
        
        self.model = nn.Sequential(*sequence)
        self.norm = Normalize()

    def forward(self, input):
        z =  self.model(input)
        return  self.norm(z, 1)


class JSCC_decoder(nn.Module):
    def __init__(self, n_channel=8):
           
        assert(n_channel>=0)
        super(JSCC_decoder, self).__init__()

        activation = nn.PReLU()
        
        sequence = [nn.Conv2d(n_channel, 32, kernel_size=5, padding=2, stride=1), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1), 
                 activation,
                 nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), 
                 activation,
                 nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1), 
                 nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return  2*self.model(input)-1



class MSSIM(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 window_size: int=11,
                 size_average:bool = True,
                 is_normalize:bool = True,
                 is_SSIM:bool = False) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)
        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average
        self.is_normalize = is_normalize
        self.is_SSIM = is_SSIM

    def gaussian_window(self, window_size:int, sigma: float) -> Tensor:
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1: Tensor,
             img2: Tensor,
             window_size: int,
             in_channel: int,
             size_average: bool) -> Tensor:

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = img1.max() - img1.min()
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if self.is_normalize:
            mssim = (mssim + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights
  
        output = torch.prod(pow1[:-1] * pow2[-1])

        if self.is_SSIM: 
            return 1 - output, mssim[0]
        else:
            return 1-output


class End_classifier(nn.Module):
    def __init__(self, n_channel=8):
           
        assert(n_channel>=0)
        super(End_classifier, self).__init__()

        activation = nn.ReLU()
        
        sequence = [nn.Conv2d(n_channel, 32, kernel_size=3, padding=1, stride=1), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), 
                 activation,
                 nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
                 activation,
                 Flatten(),
                 nn.Linear(32*8*8, 256), 
                 activation,
                 nn.Linear(256, 128), 
                 activation,
                 nn.Linear(128, 64), 
                 activation,
                 nn.Linear(64, 10)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return  self.model(input)



class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False,one_D_conv=False, one_D_conv_size=63):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D-1):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)
        netD = NLayerDiscriminator(input_nc,ndf,n_layers,norm_layer,use_sigmoid,getIntermFeat,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size)
        if getIntermFeat:
            for j in range(n_layers+2):
                setattr(self, 'scale' + str(num_D-1) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        else:
            setattr(self,'layer'+str(num_D-1),netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.one_D_conv = one_D_conv
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D-1):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        if self.getIntermFeat:
            model = [getattr(self, 'scale' + str(num_D - 1) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
        else:
            model = getattr(self, 'layer' + str(num_D - 1))
        if self.one_D_conv:
            result.append(self.singleD_forward(model, input))
        else:
            result.append(self.singleD_forward(model,input_downsampled))
        return result



class DC_Generator(torch.nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, act='ReLU'):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if act == 'ReLU':
            activation = nn.ReLU(True)
        elif act == 'LeakyReLU':
            activation = nn.LeakyReLU(0.2, inplace=True)

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(1024),
            activation,

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            activation,

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            activation,

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class DC_Discriminator(torch.nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d, act='ReLU'):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if act == 'ReLU':
            activation = nn.ReLU(True)
        elif act == 'LeakyReLU':
            activation = nn.LeakyReLU(0.2, inplace=True)

        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            activation,

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            activation,

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(1024),
            activation)
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


def define_DC_G(in_channels=100, channels=3, activation='ReLU', norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = DC_Generator(in_channels=in_channels, channels=channels, norm_layer=norm_layer, act=activation)    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_DC_D(channels=3, activation='ReLU', norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = DC_Discriminator(channels=channels, norm_layer=norm_layer, act=activation) 
    return init_net(net, init_type, init_gain, gpu_ids)