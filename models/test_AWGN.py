
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

            # Clipping Ratio
# Set up modulation scheme (2 -> QPSK, 4 -> 16QAM, 8-> 64QAM)
N_bit = 2
qam = QAM(Ave_Energy=1, B=N_bit)


# Calculate the number of target ldpc codeword length
N_syms = opt.P*opt.S*opt.M 
N_bits = N_syms * N_bit
a, b = 1, 2
rate = a/b
if rate == 1/2:
    d_v, d_c = 2, 4
elif rate == 1/3:
    d_v, d_c = 2, 3
elif rate == 2/3:
    d_v, d_c = 2, 6

CHANNEL_CODE = 'LDPC'
# Set up channel code
if CHANNEL_CODE == 'LDPC':
    k = math.ceil(N_bits/b)*a
    n = int(k/rate)
    if n % d_c != 0:
        n = (n//d_c+1)*d_c
        k = n//b*a
    ldpc = LDPC(d_v, d_c, k, maxiter=200)
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
tx_syms_complex = qam.Modulation(tx_bits_tmp)

SNR_list = np.arange(-0.25,4,0.25)

#########################################################################################
#Test the BER for each SNR value

for idx in range(SNR_list.shape[0]):

    SNR = SNR_list[idx]
    print('Processing SNR %.3f dB.......' % (SNR))
    
    noise_pwr = 10**(-SNR*0.1)
    noise = np.sqrt(noise_pwr/2)* (np.random.randn(*tx_syms_complex.shape) + 1j*np.random.randn(*tx_syms_complex.shape))

    rx_syms_complex = tx_syms_complex + noise

    LLR = qam.LLR_AWGN(rx_syms_complex, noise_pwr)[:N_test*n].reshape(N_test, n)
    LLR = np.clip(LLR, -10, 10)

    # Decoding and demodulation
    rx_bits = np.zeros((N_test, k))
    #llr = np.transpose(LLR)
    if CHANNEL_CODE == 'LDPC':
        t_start = time.time()
        for i in range(N_test):
            rx_bits[i,:] = ldpc.dec(LLR[i])
            #rx_bits = ldpc.dec(llr[:,:6])
            if i % 100 == 0:
                print('DECODED: %d' % (i))
        #print('time: %.3f' % (time.time()-t_start))

    elif CHANNEL_CODE == 'NONE':
        rx_bits[LLR<0] = 1
    
    BER = np.sum(abs(rx_bits-tx_bits))/(N_test*k)
    print("BER: %f" % (BER))
    





