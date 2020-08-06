
import numpy as np
from scipy.linalg import dft
import os
import torch


class HDM():
    def __init__(self, perm=None, W=4, V=4, D=256, K=4, Modulation='QPSK'):

        self.Modulation = Modulation
        self.W = W
        self.V = V
        self.D = D
        self.K = K
        self.M = D // K

        if self.Modulation == 'QPSK':
            self.symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
            self.Q = 4

        self.rate = self.V * self.K * (np.log2(self.Q) + np.log2(self.M)) / self.D

        if perm is None:
            self.perm = [np.random.permutation(self.D) for i in range(W * V)]
            self.perm_inv = [np.argsort(self.perm[i]) for i in range(W * V)]

        self.perm = np.vstack(self.perm)
        self.perm_inv = np.vstack(self.perm_inv)

        self.DFT = dft(self.D)
        self.dictionary = [self.DFT[self.perm[i]] for i in range(W * V)]
        self.dictionary = np.hstack(self.dictionary)

    def sample(self, N, is_noise=False, noise_type='Uniform', noise_power=0.5):
        '''
        Sample N valid HDM vectors
        output size:
            out2: (N, 2D)
            out1: (N, WV, 2D)
        '''
        index = np.random.randint(self.M, size=(N, self.W * self.V, self.K))
        sym = np.random.randint(self.Q, size=(N, self.W * self.V, self.K))
        for k in range(self.K):
            index[:, :, k] += self.M * k

        # Generate noise
        if is_noise is True:
            if noise_type == 'Uniform':
                noise_real = np.random.rand(N, self.W * self.V, self.K) - 0.5
                noise_imag = np.random.rand(N, self.W * self.V, self.K) - 0.5
                noise = noise_power * 2 * (noise_real + noise_imag * 1j)

        # Option 1: generate convertible shape
        hdm_out = np.zeros((N, self.W * self.V, self.D), dtype=complex)
        for i in range(N):
            for v in range(self.W * self.V):
                sub_atom = self.dictionary[:, index[i, v, :] + self.D * v]
                sub_sym = self.symbols[sym[i, v, :]]
                if is_noise is True:
                    sub_sym += noise[i, v]
                hdm_out[i, v] = sub_atom.dot(sub_sym)

        hdm_real = hdm_out.real
        hdm_imag = hdm_out.imag
        out1 = torch.from_numpy(np.concatenate((hdm_real, hdm_imag), axis=2)).float()

        out2 = torch.sum(out1, 1)

        return out1, out2, index, sym

    def decode(self, w):
        '''
        input size: (N, WV, sqrt(2D), sqrt(2D))
        '''
        # Reshape

        N, WV, qD, _ = w.shape
        w = w.view(N, WV, -1)  # (N, WV, 2D)
        w = w.view(N, WV, 2, qD * qD // 2).permute(0, 1, 3, 2)  # (N, WV, D, 2)

        perm_inv = torch.from_numpy(self.perm_inv).long()  # (WV, D)

        out = torch.zeros_like(w)
        for v in range(WV):
            out[:, v, :, :] = w[:, v, perm_inv[v], :]

        out = torch.ifft(out, 1)

        return out

    def proj(self, w):
        '''
        input size: (N, WV, sqrt(2D), sqrt(2D))
        '''
        # Reshape

        perm = torch.from_numpy(self.perm).long()  # (WV, D)

        N, WV, qD, _ = w.shape
        latent = self.decode(w)   # (N, WV, D, 2)
        ref = torch.zeros_like(latent)
        ref[latent < 0] = -1
        ref[latent >= 0] = 1
        diff = latent - ref
        diff = diff[:, :, :, 0]**2 + diff[:, :, :, 1]**2
        for i in range(N):
            for j in range(WV):
                _, index = torch.sort(diff[i, j, :])
                ref[i, j, index[2:], :] = 0
                latent[i, j, :, :] = torch.fft(ref[i, j, :, :], 1)
                latent[i, j, :, :] = latent[i, j, perm[j], :]

        return torch.cat((latent[:, :, :, 0], latent[:, :, :, 1]), dim=2).view(N, WV, qD, qD)

    def suffle(self, w):
        N, WV, qD, _ = w.shape
        w = w.view(N, WV, -1)  # (N, WV, 2D)
        w = w.view(N, WV, 2, qD * qD // 2).permute(0, 1, 3, 2)  # (N, WV, D, 2)

        perm_inv = torch.from_numpy(self.perm_inv).long()  # (WV, D)

        out = torch.zeros_like(w)
        for v in range(WV):
            out[:, v, :, :] = w[:, v, perm_inv[v], :]
        return out
    def rate(self):
        return self.rate

    def params(self):
        print('HDM dimension(D): ' + str(self.D))
        print('HDM groups(K): ' + str(self.K))
        print('HDM hyper-dimention(V): ' + str(self.V))
        print('HDM hyper-dimention(W):' + str(self.W))
        print('HDM rate:' + str(self.rate))

    def set_perm(self, perm):
        self.perm = perm
        self.DFT = dft(self.D)
        self.dictionary = [self.DFT[self.perm[i]] for i in range(self.W * self.V)]
        self.dictionary = np.hstack(self.dictionary)
        for i in range(self.W * self.V):
            self.perm_inv[i] = np.argsort(self.perm[i])

    def save_perm(self, path):
        np.savetxt(path, self.perm, fmt='%d')

    def load_perm(self, path):
        perm = np.loadtxt(path, dtype=int)
        self.set_perm(perm)

    def show_perm(self):
        print(self.perm)

    def demodulate(self, input):
        pass
