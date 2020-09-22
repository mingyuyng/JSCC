import numpy as np
from scipy.linalg import dft
import os
import matplotlib.pyplot as plt


class QAM():
    def __init__(self, Ave_Energy=1, B=2):
        '''
        Gray mapping for QPSK (B=2)

        | b0 |  I  | b1 |  Q  |
        | 0  | -1  | 0  | -1  |
        | 1  |  1  | 1  |  1  |

        Gray mapping for 16-QAM (B=4)

        | b0b1 |  I  | b2b3 |  Q  |
        |  00  | -3  |  00  | -3  |
        |  01  | -1  |  01  | -1  |
        |  11  |  1  |  11  |  1  |
        |  10  |  3  |  10  |  3  |

        Gray mapping for 64-QAM (B=6)

        | b0b1b2 |  I  | b3b4b5 |  Q  |
        |  000   | -7  |  000   | -7  |
        |  001   | -5  |  001   | -5  |
        |  011   | -3  |  011   | -3  |
        |  010   | -1  |  010   | -1  |
        |  110   |  1  |  110   |  1  |
        |  111   |  3  |  111   |  3  |
        |  101   |  5  |  101   |  5  |
        |  100   |  7  |  100   |  7  |
        '''

        self.index = np.arange(2**(B//2))
        
        if B > 1:
            if B == 2:
                self.map = np.array([-1, 1])
                self.map2 = np.array([[0], [1]])
                self.unit = np.sqrt(Ave_Energy/2)
            elif B == 4:
                self.map = np.array([-3, -1, 3, 1])
                self.map2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                self.unit = np.sqrt(Ave_Energy/10)
            elif B == 6:
                self.map = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
                self.map2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
                self.unit = np.sqrt(Ave_Energy/42)

            self.inv_map_1 = np.zeros((B//2, 2**(B//2-1)))
            self.inv_map_0 = np.zeros((B//2, 2**(B//2-1)))

            tmp = self.index.copy()
            for i in range(B//2):
                self.inv_map_1[i] = np.where(tmp>=2**(B//2-1-i))[0] 
                self.inv_map_0[i] = np.where(tmp<2**(B//2-1-i))[0] 
                tmp[tmp>=2**(B//2-1-i)] -= 2**(B//2-1-i)

            self.constellation = (np.expand_dims(self.map, axis=1) + np.expand_dims(self.map, axis=0)*1j)*self.unit        
            self.bound = self.unit*(np.arange(-2**(B//2)+2, 2**(B//2), 2))
        else:
            self.constellation = np.array([-1, 1])
        
        self.B = B

    def LLR(self, y, sigma):

        M = y.shape[0]
        LLR = []
        for m in range(M):
            sym = y[m]
            
            f = np.vectorize(np.int)
            LLR_real = []
            LLR_imag = []

            for i in range(self.B//2):

                pos_0 = self.inv_map_0[i]
                pos_1 = self.inv_map_1[i]
                sym_0 = self.map[f(pos_0)]*self.unit
                sym_1 = self.map[f(pos_1)]*self.unit

                prob_0_real = np.sum(np.exp(-(sym.real - sym_0)**2/(sigma**2)))
                prob_1_real = np.sum(np.exp(-(sym.real - sym_1)**2/(sigma**2)))

                prob_0_imag = np.sum(np.exp(-(sym.imag - sym_0)**2/(sigma**2)))
                prob_1_imag = np.sum(np.exp(-(sym.imag - sym_1)**2/(sigma**2)))

                ratio_real = np.log(prob_0_real) - np.log(prob_1_real)
                ratio_imag = np.log(prob_0_imag) - np.log(prob_1_imag)

                LLR_real.append(ratio_real)
                LLR_imag.append(ratio_imag)

            LLR.append(np.stack(LLR_real))
            LLR.append(np.stack(LLR_imag))

        return np.hstack(LLR)
    
    def LLR_dist(self, y):
        # Calculate the LLR based on Euclidean distance
        # Used when there is clipping effect
        # y: received symbols
        M = y.shape[0]
        LLR = []
        for m in range(M):
            sym = y[m]
            
            f = np.vectorize(np.int)
            LLR_real = []
            LLR_imag = []

            for i in range(self.B//2):

                pos_0 = self.inv_map_0[i]
                pos_1 = self.inv_map_1[i]
                sym_0 = self.map[f(pos_0)]*self.unit
                sym_1 = self.map[f(pos_1)]*self.unit

                # Euclidean Distance (exponential)
                dist_0_real = np.min((sym.real - sym_0)**2)
                dist_1_real = np.min((sym.real - sym_1)**2)

                dist_0_imag = np.min((sym.imag - sym_0)**2)
                dist_1_imag = np.min((sym.imag - sym_1)**2)


                prob_1_real = 1/(1+np.exp(2*dist_1_real-2*dist_0_real))
                prob_0_real = 1 - prob_1_real

                prob_1_imag = 1/(1+np.exp(2*dist_1_imag-2*dist_0_imag))
                prob_0_imag= 1 - prob_1_imag

                ratio_real = np.log(prob_0_real) - np.log(prob_1_real)
                ratio_imag = np.log(prob_0_imag) - np.log(prob_1_imag)


                LLR_real.append(ratio_real)
                LLR_imag.append(ratio_imag)

            LLR.append(np.stack(LLR_real))
            LLR.append(np.stack(LLR_imag))

        return np.hstack(LLR)


    def LLR_OFDM(self, y, H, pwr, eps=1e-50):
        # LLR calculation for OFDM system
        # Used when we have perfect channel knowledge
        # y: received symbols
        # H: estimated channel frequency response

        M = y.shape[0]
        LLR = np.zeros(M*self.B)
        for m in range(M):
            sym = y[m]
            XH = H[m] * self.constellation
            if self.B == 1:  #If it is BPSK
                symbols_0 = XH[0]
                symbols_1 = XH[1]
                prob_0 = np.exp((-abs(sym-symbols_0)**2)/pwr[m])
                prob_1 = np.exp((-abs(sym-symbols_1)**2)/pwr[m])
                LLR[m] = np.log(prob_0+eps) - np.log(prob_1+eps)
            else: # If not BPSK
                for i in range(self.B):
                    ind = self.map2[:, i%(self.B//2)]
                    if i < self.B//2:
                        symbols_0 = XH[ind==0, :]
                        symbols_1 = XH[ind==1, :]
                    else:
                        symbols_0 = XH[:, ind==0]
                        symbols_1 = XH[:, ind==1]               
                    prob_0 = np.sum(np.exp((-abs(sym-symbols_0)**2)/pwr[m]))
                    prob_1 = np.sum(np.exp((-abs(sym-symbols_1)**2)/pwr[m]))
                    LLR[m*self.B+i] = np.log(prob_0+eps) - np.log(prob_1+eps)
        return LLR

    def LLR_OFDM_clip(self, y, H, pwr, alpha, sigma, eps=1e-50):
        # LLR calculation for OFDM system
        # Used when we have perfect channel knowledge
        # y: received symbols
        # H: estimated channel frequency response

        M = y.shape[0]
        LLR = np.zeros(M*self.B)
        for m in range(M):
            if self.B == 1:  #If it is BPSK
                sym = y[m]/H[m]
                XH = alpha * self.constellation
                symbols_0 = XH[0]
                symbols_1 = XH[1]                
                prob_0 = np.exp((-(sym.real-symbols_0.real)**2)/(pwr[m]/(abs(H[m])**2)+2*sigma))
                prob_1 = np.exp((-(sym.real-symbols_1.real)**2)/(pwr[m]/(abs(H[m])**2)+2*sigma))
                LLR[m] = np.log(prob_0+eps) - np.log(prob_1+eps)
            else:  # If not BPSK
                sym = y[m]
                XH = alpha * H[m] * self.constellation
                for i in range(self.B):
                    ind = self.map2[:, i%(self.B//2)]
                    if i < self.B//2:
                        symbols_0 = XH[ind==0, :]
                        symbols_1 = XH[ind==1, :]
                    else:
                        symbols_0 = XH[:, ind==0]
                        symbols_1 = XH[:, ind==1]      
                    prob_0 = np.sum(np.exp((-abs(sym-symbols_0)**2)/(pwr[m]+(abs(H[m])**2)*sigma)))
                    prob_1 = np.sum(np.exp((-abs(sym-symbols_1)**2)/(pwr[m]+(abs(H[m])**2)*sigma)))
                    LLR[m*self.B+i] = np.log(prob_0+eps) - np.log(prob_1+eps)
        return LLR

    def Modulation(self, x):
        '''
        input: Nx1
        '''
        if self.B != 1:
            tx = x.reshape(x.shape[0]//self.B, self.B)
            tx_I = tx[:, :self.B//2]
            tx_Q = tx[:, self.B//2:]

            index_I = np.zeros(tx_I.shape[0])
            index_Q = np.zeros(tx_Q.shape[0])

            for i in range(self.B//2):
                index_I += 2**(self.B//2-i-1)*tx_I[:,i]
                index_Q += 2**(self.B//2-i-1)*tx_Q[:,i]

            f = np.vectorize(np.int)
            tx_sym = self.unit*(self.map[f(index_I)] + self.map[f(index_Q)]*1j)

            return tx_sym
        else:
            return 2*x-1


    def Demodulation(self, y):
        '''
        input: N/Bx1
        '''

        M = y.shape[0]
        code_list = []
        for m in range(M):
            sym = y[m]

            index_real = np.where(self.bound>sym.real)[0]
            if index_real.shape[0] == 0:
                pos_real = 2**(self.B//2)-1
            else:
                pos_real = np.min(index_real)

            code_list.append(self.map2[int(pos_real)])

            index_imag = np.where(self.bound>sym.imag)[0]
            if index_imag.shape[0] == 0:
                pos_imag = 2**(self.B//2)-1
            else:
                pos_imag = np.min(index_imag)

            code_list.append(self.map2[int(pos_imag)])

        return np.hstack(code_list)




if __name__ == "__main__":

    qam = QAM(Ave_Energy=1, B=4)

    x = np.random.randint(2, size=240)
    tx = qam.Modulation(x)

    SNR = 10
    sigma = 1/np.sqrt(2*10**(0.1*SNR))
    noise = sigma * (np.random.randn(tx.shape[0]) + 1j*np.random.randn(tx.shape[0]))

    rx = qam.Demodulation(tx+noise)

    LLR = qam.LLR(tx+noise, sigma)
    LLR[LLR>5] = 5
    LLR[LLR<-5] = -5
    import pdb; pdb.set_trace()  # breakpoint 8e4afe0e //

    
