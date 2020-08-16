
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

##########################################################################################################
# OFDM experiments with LDPC included
# Each packet carriers one LDPC codeword


# Set up modulation scheme (2 -> QPSK, 4 -> 16QAM, 8-> 64QAM)
N_bit = 2
qam = QAM(Ave_Energy=1, B=N_bit)

# Set up the OFDM channel
################################################################################
#
# Parameters:
#    N: batch size
#    P: number of packets
#    S: number of information symbols per packet (there is also ONE symbol for pilots)
#    M: number of subcarriers per symbol. Default is 64. 
#    K: length of cyclic prefix. Default is 16
#    L: number of taps for the multipath channel
#
#    is_clip:  whether to clip or not
#    PAPR:     the PAPR requirement
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
opt.N = 200        # Batch size
opt.P = 1          # Number of packets
opt.S = 4          # Number of symbols
opt.M = 64         # Number of subcarriers per symbol
opt.K = 16         # Length of CP
opt.L = 8          # Number of paths
opt.decay = 4

opt.is_clip = False
opt.PAPR = 5

opt.is_cfo = False
opt.is_trick = True
opt.is_cfo_random = False
opt.max_ang = 1.7
opt.ang = 1.7

opt.N_pilot = 2   # Set to 0 if opt.is_pilot is false

opt.gpu_ids = ['0']

CE = 'LMMSE'  # Channel Estimation Method
EQ = 'MMSE'   # Equalization Method
CHANNEL_CODE = 'LDPC'   # Channel Coding 

if CE not in ['LS', 'LMMSE', 'TRUE']:
    raise Exception("Channel estimation method not implemented")

if EQ not in ['ZF', 'MMSE']:
    raise Exception("Equalization method not implemented")

if CHANNEL_CODE not in ['LDPC', 'NONE']:
    raise Exception("Channel coding method not implemented")

# Calculate the number of target ldpc codeword length  
n = opt.P*opt.S*opt.M*N_bit

# Set up channel code
if CHANNEL_CODE == 'LDPC':
    d_v = 2
    d_c = 4
    rate = 1-(d_v/d_c)
    k = round(n*rate)
    ldpc = LDPC(d_v, d_c, k, maxiter=50)
elif CHANNEL_CODE == 'NONE':
    k = n

# Load the generated pilot
# pilot = torch.load('pilot.pt')

# Create the OFDM channel

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

pilot = torch.load('../util/pilot.pt')
ofdm_channel = OFDM_channel(opt, device, pilot=pilot)
SNR_list = np.arange(0,35,5)



print('Total number of bits tested: %d' % (opt.N*k))
print('Channel Estimation: ' + CE)
print('Equalization: ' + EQ)
print('Channel Coding: ' + CHANNEL_CODE)

for idx in range(SNR_list.shape[0]):

    SNR = SNR_list[idx]
    print('Processing SNR %d dB.......' % (SNR))

    # Generate the bits to be transmitted
    tx_bits = np.random.randint(2, size=k*opt.N)

    if CHANNEL_CODE == 'LDPC':
        tx_list = []    
        for i in range(opt.N):
            tx_bits_tmp = tx_bits[i*k:(i+1)*k]
            tx_c = ldpc.enc(tx_bits_tmp)
            tx_sym = np.expand_dims(qam.Modulation(tx_c), axis=1)
            tx_tmp = np.concatenate((tx_sym.real, tx_sym.imag), axis=1)
            tx_tmp = tx_tmp.reshape(1, opt.P, opt.S, opt.M, N_bit)
            tx_list.append(tx_tmp)
        tx = np.vstack(tx_list)
    elif CHANNEL_CODE == 'NONE':
        tx_sym = np.expand_dims(qam.Modulation(tx_bits), axis=1)
        tx_tmp = np.concatenate((tx_sym.real, tx_sym.imag), axis=1)
        tx = tx_tmp.reshape(opt.N, opt.P, opt.S, opt.M, N_bit)

    tx = torch.from_numpy(tx).float().to(device)
    
    # Pass the OFDM channel 
    out_pilot, out_sig, H_true, noise_pwr = ofdm_channel(tx, SNR=SNR)
    
    H_est, rx = OFDM_receiver(CE, EQ, out_sig, out_pilot, ofdm_channel.pilot, opt.M*noise_pwr, H_true=H_true)

    # Channel Estimation
    #if CE == 'LS':
    #    H_est = LS_channel_est(ofdm_channel.pilot, out_pilot)
    #elif CE == 'LMMSE':
    #    H_est = LMMSE_channel_est(ofdm_channel.pilot, out_pilot, opt.M*noise_pwr)
    #elif CE == 'TRUE':
    #    H_est = H_true.unsqueeze(2)

    print("Channel estimation MSE: %f" % (torch.sum(abs(H_est.squeeze(2)-H_true.to(device))**2).item() /opt.N))

    # Equalization
    #if EQ == 'ZF':
    #    rx_sym = ZF_equalization(H_est, out_sig).numpy()
    #elif EQ == 'MMSE':
    #    rx_sym = MMSE_equalization(H_est, out_sig, opt.M*noise_pwr).numpy()
    
    rx_sym = rx.detach().cpu().numpy()
    rx_sym= rx_sym[...,0] + rx_sym[...,1] * 1j
    

    # Decoding and demodulation
    if CHANNEL_CODE == 'LDPC':
        
        rx_list = []
        for i in range(opt.N):
            LLR = qam.LLR(rx_sym[i].flatten(), 0, simple = True)
            LLR[LLR>10] = 10
            LLR[LLR<-10] = -10
            rx_bits_tmp = ldpc.dec(LLR)
            rx_list.append(rx_bits_tmp)
        rx_bits = np.hstack(rx_list)
    elif CHANNEL_CODE == 'NONE':
        rx_bits = qam.Demodulation(rx_sym.flatten())

    BER = np.sum(rx_bits!=tx_bits)/(opt.N*k)
    print("BER: %f" % (BER))




