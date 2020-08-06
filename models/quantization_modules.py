import torch
import torch.nn as nn
import math
import numpy as np

def binary_tensor(x):
  return x.sign()

def quantization_tensor(x,num_bits):
  qmin = 0.
  qmax = 2.**num_bits - 1.
  min_val, max_val = x.min(), x.max()

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale

  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)
  q_x = zero_point + x / scale
  q_x.clamp_(qmin, qmax).round_()
  q_x = q_x.round().byte()
  q_x = scale * (q_x.float() - zero_point)
  return q_x

def fixedpointquantization(x, N, M):
    q_x = x * 2.**M
    qmin = - 2.**N
    qmax = 2.**N -1
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x / 2**M
    return q_x
## fixed point quantization
class FixedConv2d(nn.Conv2d):
  def __init__(self,*kargs,**kwargs):
    super(FixedConv2d,self).__init__(*kargs, **kwargs)

  def forward(self,input):
    self.weight.data = fixedpointquantization(self.weight.data,9,8)
    out = nn.functional.conv2d(input,self.weight,None,self.stride,self.padding, self.dilation, self.groups)
    #out = fixedpointquantization(out,17,16)
    return out

class FixedLinear(nn.Linear):
  def __init__(self,*kargs,**kwargs):
    super(FixedLinear,self).__init__(*kargs,**kwargs)
  
  def forward(self,input):
    self.weight.data = fixedpointquantization(self.weight.data,9,8)
    out = nn.functional.linear(input,self.weight)
    #out = fixedpointquantization(out,17,16)
    return out
##normal quantization
class QuantConv2d(nn.Conv2d):
  def __init__(self,*kargs,**kwargs):
    super(QuantConv2d,self).__init__(*kargs, **kwargs)

  def forward(self,input):
    self.weight.data = quantization_tensor(self.weight.data,8)
    out = nn.functional.conv2d(input,self.weight,None,self.stride,self.padding, self.dilation, self.groups)
    #out = fixedpointquantization(out,17,16)
    return out

class QuantLinear(nn.Linear):
  def __init__(self,*kargs,**kwargs):
    super(QuantLinear,self).__init__(*kargs,**kwargs)
  
  def forward(self,input):
    self.weight.data = quantization_tensor(self.weight.data,8)
    out = nn.functional.linear(input,self.weight)
    #out = fixedpointquantization(out,17,16)
    return out
##binary
class BinaryConv2d(nn.Conv2d):
  def __init__(self,*kargs,**kwargs):
    super(BinaryConv2d,self).__init__(*kargs, **kwargs)

  def forward(self,input):
    self.weight.data = binary_tensor(self.weight.data)
    out = nn.functional.conv2d(input,self.weight,None,self.stride,self.padding, self.dilation, self.groups)
    #out = fixedpointquantization(out,17,16)
    return out

class BinaryLinear(nn.Linear):
  def __init__(self,*kargs,**kwargs):
    super(BinaryLinear,self).__init__(*kargs,**kwargs)
  
  def forward(self,input):
    self.weight.data = binary_tensor(self.weight.data)
    out = nn.functional.linear(input,self.weight)
    #out = fixedpointquantization(out,17,16)
    return out
