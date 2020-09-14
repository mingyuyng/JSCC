
import numpy as np
from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import scipy.io as sio
from mod import QAM
from ldpc import LDPC
import sys
import time
import math
from channel import *
import types
from scipy import special

##########################################################################################################
# OFDM experiments with LDPC included
# Each packet carriers one LDPC codeword

PI = 3.1415926


# Set up the OFDM channel
################################################################################
#
# Parameters:
#    N: batch size
#    P: number of packets (Now is set to 1)
#    S: number of information symbols per packet (there is also ONE symbol for pilots)
#    M: number of subcarriers per symbol. Default is 64. 
#    K: length of cyclic prefix. Default is 16
#    L: number of taps for the multipath channel
#    
#    is_cfo:   whether to include CFO effect
#    is_trick: whether to include the phase uncertainty
#    is_cfo_random:    whether the cfo is constant or uniform distributed
#    max_ang:  maximum degree/sample if cfo is set to be random
#    ang:  degree/sample if cfo is set to be constant
#
#    is_pilot: if we introduce additional pilots for phase correction (not used yet)
#    N_pilot: number of additional pilots for phase correction (not used yet)
#    
# Each packet contains S symbols:
#
#    Packet#1: |-CP(K)-|---Pilot(M)---|-CP(K)-|---symbol1(M)---|-CP(K)-|---symbol2(M)---|...   
#    Packet#2: |-CP(K)-|---Pilot(M)---|-CP(K)-|---symbol1(M)---|-CP(K)-|---symbol2(M)---|...   
#    Packet#3: |-CP(K)-|---Pilot(M)---|-CP(K)-|---symbol1(M)---|-CP(K)-|---symbol2(M)---|...  
# 
# Additional pilots might be added
# 
# Note: the last dimention is 2 because we are representing the real part and imaginary part separately
#
################################################################################

opt = types.SimpleNamespace()
#opt.N = 10         # Batch size
opt.P = 1          # Number of packets  (Keep this as 1 for now)
opt.S = 12          # Number of symbols
opt.M = 64         # Number of subcarriers per symbol
opt.K = 16         # Length of CP
opt.L = 8          # Number of paths
opt.decay = 4

opt.is_clip = False    # Whether to clip the OFDM signal or not
opt.CR = 1             # Clipping Ratio

alpha = 0.7846
sigma = 0.0365

opt.is_cfo = False     # Whether to add CFO to the OFDM signal (not used for the experiment yet)
opt.is_trick = True
opt.is_cfo_random = False
opt.max_ang = 1.7
opt.ang = 1.7

opt.N_pilot = 2           # Number of pilots for channel estimation
opt.pilot = 'QPSK'   # QPSK or ZadoffChu

opt.gpu_ids = ['1']    # GPU setting
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

CE = 'TRUE'         # Channel Estimation Method
EQ = 'MMSE'          # Equalization Method
CHANNEL_CODE = 'LDPC'   # Channel Coding 

if CE not in ['LS', 'LMMSE', 'TRUE']:
    raise Exception("Channel estimation method not implemented")

if EQ not in ['ZF', 'MMSE']:
    raise Exception("Equalization method not implemented")

if CHANNEL_CODE not in ['LDPC', 'NONE']:
    raise Exception("Channel coding method not implemented")

# Set up modulation scheme (2 -> QPSK, 4 -> 16QAM, 8-> 64QAM)
N_bit = 1
qam = QAM(Ave_Energy=1, B=N_bit)

# Calculate the number of target ldpc codeword length
N_syms = opt.P*opt.S*opt.M 
N_bits = N_syms * N_bit
a, b = 2, 3
rate = a/b
if rate == 1/2:
    d_v, d_c = 2, 4
elif rate == 1/3:
    d_v, d_c = 2, 3
elif rate == 2/3:
    d_v, d_c = 2, 6

# Set up channel code
if CHANNEL_CODE == 'LDPC':
    k = math.ceil(N_bits/b)*a
    n = int(k/rate)
    if n % d_c != 0:
        n = (n//d_c+1)*d_c
        k = n//b*a
    ldpc = LDPC(d_v, d_c, k, maxiter=50)
elif CHANNEL_CODE == 'NONE':
    k = math.ceil(N_bits/b)*a
    n = k
print(k)
print(n)
# Generate information bits
N_test = 100
tx_bits = np.random.randint(2, size=(N_test, k))

if CHANNEL_CODE == 'LDPC':
    # Map the information bits to channel codes
    tx_bits_c = np.zeros((N_test, n))
    for i in range(N_test):
        tx_bits_c[i,:] = ldpc.enc(tx_bits[i,:])
elif CHANNEL_CODE == 'NONE':
    tx_bits_c = tx_bits

# Map the binary data to complex symbols
N_trans = math.ceil(N_test*n/N_bits)
remain_bits = N_trans*N_bits - N_test*n
dummy = np.zeros(remain_bits)

tx_bits_tmp = np.concatenate((tx_bits_c.flatten(),dummy),axis=0)
tx_syms_complex = qam.Modulation(tx_bits_tmp).reshape(N_trans, opt.P, opt.S, opt.M)
tx_syms_real = torch.from_numpy(tx_syms_complex.real).float().unsqueeze(-1).to(device)
tx_syms_imag = torch.from_numpy(tx_syms_complex.imag).float().unsqueeze(-1).to(device)
tx_syms = torch.cat((tx_syms_real, tx_syms_imag), dim=-1)

opt.N = N_trans
# Create the OFDM channel
ofdm_channel = OFDM_channel(opt, device)
SNR_list = np.arange(5,10,5)

print('Total number of bits tested: %d' % (N_test*k))
print('Channel Estimation: ' + CE)
print('Equalization: ' + EQ)
print('Channel Coding: ' + CHANNEL_CODE)


#########################################################################################
#Test the BER for each SNR value

for idx in range(SNR_list.shape[0]):

    SNR = SNR_list[idx]
    print('Processing SNR %d dB.......' % (SNR))

    # Pass the OFDM channel 
    out_pilot, out_sig, H_true, noise_pwr, _, _ = ofdm_channel(tx_syms, SNR=SNR, norm=False)

    # Channel Estimation
    if CE == 'LS':
        H_est = LS_channel_est(ofdm_channel.pilot, out_pilot)
    elif CE == 'LMMSE':
        H_est = LMMSE_channel_est(ofdm_channel.pilot, out_pilot, opt.M*noise_pwr)
    elif CE == 'TRUE':
        H_est = H_true.unsqueeze(2).to(device)

    print("Channel estimation MSE: %f" % (torch.sum(abs(H_est.squeeze(2)-H_true.to(device))**2).item() /opt.N))

    # Equalization
    if EQ == 'ZF':
        rx_sym = ZF_equalization(H_est, out_sig).detach().cpu().numpy()
    elif EQ == 'MMSE':
        rx_sym = MMSE_equalization(H_est, out_sig, opt.M*noise_pwr).detach().cpu().numpy()
    
    rx_sym = rx_sym[...,0] + rx_sym[...,1] * 1j
    rx_sym = rx_sym.flatten()[:N_test*n//N_bit+1]
    
    H_est = H_est.repeat(1,1,opt.S,1,1).detach().cpu().numpy()
    H_est = H_est[...,0] + H_est[...,1] * 1j
    H_est = H_est.flatten()[:N_test*n//N_bit+1]

    out_sig = out_sig.detach().cpu().numpy()
    out_sig = out_sig[...,0] + out_sig[...,1] * 1j
    out_sig = out_sig.flatten()[:N_test*n//N_bit+1]

    noise_pwr = noise_pwr.repeat(1,opt.P,opt.S,opt.M).detach().cpu().numpy()
    noise_pwr = noise_pwr.flatten()[:N_test*n//N_bit+1]
    
    LLR = qam.LLR_OFDM(out_sig, H_est, opt.M*noise_pwr)[:N_test*n].reshape(N_test, n)
    LLR = np.clip(LLR, -10, 10)
    
    # Decoding and demodulation
    rx_bits = np.zeros((N_test, k))
    if CHANNEL_CODE == 'LDPC':
       
        for i in range(N_test):
            rx_bits[i,:] = ldpc.dec(LLR[i])
            if i % 100 == 0:
                print('DECODED: %d' % (i))
    elif CHANNEL_CODE == 'NONE':
        rx_bits[LLR<0] = 1
    
    BER = np.sum(abs(rx_bits-tx_bits))/(N_test*k)
    print("BER: %f" % (BER))
    
    correct_blocks = np.sum(np.sum(abs(rx_bits - tx_bits), 1)==0)
    BlockER = correct_blocks/N_test
    print("Block Error Rate: %f" % (1-BlockER))





