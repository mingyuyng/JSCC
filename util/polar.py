import numpy as np
from polarcodes import *
from mod import QAM


class Polar():
    def __init__(self, N, K, Design_SNR=0):

        self.N = N
        self.K = K
        self.Design_SNR = Design_SNR

        self.rate = K / N

        self.polar = PolarCode(N, K)
        self.polar.construction_type = 'bb'

        Construct(self.polar, self.Design_SNR)

        print(self.polar, "\n\n")

   
    def encode(self, message):

        self.polar.set_message(message)
        Encode(self.polar)
        return self.polar.u.copy()

   
    def decode(self, LLR):

        self.polar.likelihoods = LLR
        Decode(self.polar)
        return self.polar.message_received.copy()

    def rate(self):
        return self.rate


if __name__ == "__main__":

    # Initialize Polar object
    K = 512
    N = 1024

    snr = 0
    sigma = 10 ** (-snr / 20) /np.sqrt(2)

    polar = Polar(N, K, Design_SNR=snr)
   
    # Initialize QAM object
    QPSK = QAM(Ave_Energy=1, B=2)

    # Generte transmit bits
    v = np.random.randint(2, size=K)

    # LDPC encode
    x = polar.encode(v)

    # Modulation
    x_mod = QPSK.Modulation(x)
    
    # Pass the noisy channel
    noise = sigma * (np.random.randn(x_mod.shape[0]) + 1j*np.random.randn(x_mod.shape[0]))
    y_mod = x_mod + noise

    # LLR calculation
    LLR = QPSK.LLR(y_mod, sigma)

    # LLR clipping
    LLR[LLR>5] = 5
    LLR[LLR<-5] = -5

    y = polar.decode(LLR)

    print('Bit error (Soft): %d' % (abs(y - v).sum()))

    y_est = QPSK.Demodulation(y_mod)
    LLR = np.zeros(y_est.shape)
    LLR[y_est == 0] = 5
    LLR[y_est == 1] = -5
    y = polar.decode(LLR)

    print('Bit error (Hard): %d' % (abs(y - v).sum()))

    import pdb; pdb.set_trace()  # breakpoint 2335695b //

