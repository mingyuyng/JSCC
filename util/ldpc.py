
from pyldpc import make_ldpc, decode, get_message, utils
import matplotlib.pyplot as plt
import numpy as np
from mod import QAM
import time

class LDPC():
    def __init__(self, d_v, d_c, K):

        self.design_rate = 1-d_v/d_c             # Compute the designed rate
        self.N = int(K/self.design_rate)            # Compute the target output dimension

        self.H, self.G = make_ldpc(self.N, d_v, d_c, systematic=True, sparse=True)

        self.k = self.G.shape[1]
        self.K = K
        self.true_rate = self.k / self.N

        self.dumb = np.zeros(self.k-K)

        print('LDPC code built, with a design rate of %.3f and a true rate of %.3f' % (self.design_rate, self.true_rate))

    def enc(self, v):

        x = np.concatenate((v, self.dumb), axis=0)
        return utils.binaryproduct(self.G, x)


    def dec(self, LLR):

        d = get_message(self.G, decode(self.H, LLR, maxiter=200))

        return d[:self.K]




if __name__ == "__main__":

    

    # Initialize LDPC object
    d_v = 3
    d_c = 4
    K = 2048
    ldpc = LDPC(d_v, d_c, K)

    # Initialize QAM object
    QPSK = QAM(Ave_Energy=1, B=2)

    # Initialize AWGN channel
    snr = 0
    sigma = 10 ** (-snr / 20) /np.sqrt(2)


    # Generte transmit bits
    v = np.random.randint(2, size=K)

    # LDPC encode
    x = ldpc.enc(v)

    # Modulation
    x_mod = QPSK.Modulation(x)
    
    # Pass the noisy channel
    noise = sigma * (np.random.randn(x_mod.shape[0]) + 1j*np.random.randn(x_mod.shape[0]))
    y_mod = x_mod + noise

    # LLR calculation
    LLR = QPSK.LLR(y_mod, sigma)

    # LLR clipping
    #LLR[LLR>5] = 5
    #LLR[LLR<-5] = -5

    #for i in range(5):
    iter_start_time = time.time()
    y = ldpc.dec(LLR)
    print(time.time() - iter_start_time)

    print('Bit error (Soft): %d' % (abs(y - v).sum()))

    y_est = QPSK.Demodulation(y_mod)
    LLR = np.zeros(y_est.shape)
    LLR[y_est == 0] = 5
    LLR[y_est == 1] = -5
    y = ldpc.dec(LLR)

    print('Bit error (Hard): %d' % (abs(y - v).sum()))

    import pdb; pdb.set_trace()  # breakpoint 2335695b //
