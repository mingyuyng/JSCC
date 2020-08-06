
import numpy as np
from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import scipy.io as sio
from mod import QAM
from ldpc import LDPC
from polar import Polar
import sys
import time
import math


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


PI = 3.1415926

d_v = 2
d_c = 4
K = 4096
N = 8192

# Initialize LDPC
ldpc = LDPC(d_v, d_c, K)
Num_bit = 2

is_channel_code = True
SNR_list = np.arange(1,33,2)
err_list = []

qam = QAM(Ave_Energy=1, B=Num_bit)

for n in range(SNR_list.shape[0]):

    SNR = SNR_list[n]
    sigma = 10 ** (-SNR / 20) /np.sqrt(2)
    Num_test = 10
    err = 0

    for i in range(Num_test):
        tx = np.random.randint(2, size=K)

        if is_channel_code:
            tx_c = ldpc.enc(tx)
            tx_sym = qam.Modulation(tx_c)
        else:
            tx_sym = qam.Modulation(tx)
            
        noise = sigma * (np.random.randn(tx_sym.shape[0]) + 1j*np.random.randn(tx_sym.shape[0]))
        rx_sym = tx_sym + noise

        if is_channel_code:
            LLR = qam.LLR(rx_sym, sigma)
            # LLR clipping
            LLR[LLR>5] = 5
            LLR[LLR<-5] = -5
            rx = ldpc.dec(LLR)
        else:
            rx = qam.Demodulation(rx_sym)

        err += abs(rx - tx).sum()
        progress_bar(i, Num_test)

    err_list.append(err/(Num_test*K))
    print('Finished %ddB, with BER %.3f' % (SNR, err/(Num_test*K)))


    
