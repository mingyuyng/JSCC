import numpy as np
from scipy.linalg import dft
import os
import torch
from polarcodes import *
import matplotlib.pyplot as plt


class Quantizer():

    def __init__(self, bits=4, center=None):

        
        self.map = self.generate_map(bits)

        self.num_bits = bits
        level = 2**bits
        
        if center is None:
            center_tmp = np.linspace(0, 1, num=level+1)
            self.center = center_tmp[:-1] + 1 / (2*level)
        else:
            self.center = center

        Construct(self.polar, self.Design_SNR)

        print(self.polar, "\n\n")

    def generate_map(self, bits):

        ma = np.array([[0],[1]])
        for i in range(bits-1):
            l = ma.shape[0]
            zero = np.zeros((l,1))
            one = np.ones((l,1))
            upper = np.concatenate((zero, ma), axis=1)
            lower = np.concatenate((one, ma), axis=1)
            ma = np.concatenate((upper, lower), axis=0)

        return ma

    def quan2binary(self, index):

        index_flat = index.flatten()

        code_list = self.map[index_flat]

        return code_list.flatten()

    def binary2quan(self, bits):

        l = bits.shape[0]
        bits_reshaped = np.reshape(bits, (l // self.num_bits, self.num_bits))

        index = np.zeros(bits_reshaped.shape[0])
        for i in range(self.num_bits):
            index += 2**i * bits_reshaped[:, self.num_bits-1-i]


        quantized_vec = self.center[index.astype(int)]

        return quantized_vec

    def quantize(self, vector):

        input_size = vector.shape
        vector = vector.flatten()
        W_stack = np.hstack([np.expand_dims(vector, axis=1) for _ in range(self.center.shape[0])])
        W_index = np.argmin(abs(W_stack - self.center), axis=1)
        W_hard = self.center[W_index]
        return W_hard.reshape(input_size), W_index.reshape(input_size)
