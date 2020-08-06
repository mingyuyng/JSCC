
import numpy as np
from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import scipy.io as sio
from util.mod import QAM

PI = 3.1415926

def Normalize(x, pwr=1):
    # Normalize the packet power
    # Input size:  ...xMx2
    power = torch.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/torch.sqrt(power)
    return alpha*x, alpha

class Clip(nn.Module): 
    def __init__(self, A=1):
        super(Clip, self).__init__()
        self.A = np.sqrt(10**(0.1*A)/64)
    def forward(self, x):
        # Input size:  ...xMx2
        with torch.no_grad():
            amp = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2)
            scale = self.A/amp
            scale[scale>1] = 1
        return x*scale.unsqueeze(-1)

class Add_CP(nn.Module): 
    def __init__(self, length=16):
        super(Add_CP, self).__init__()
        self.length = length
    def forward(self, x):
        # Input size:  NxPxSxMx2
        return torch.cat((x[:,:,:,-self.length:,:], x), dim=3)

class RM_CP(nn.Module): 
    def __init__(self, length=16):
        super(RM_CP, self).__init__()
        self.length = length
    def forward(self, x):
        # Input size:  NxPxSxMx2
        return x[:,:,:,self.length:, :]

class Add_CFO(nn.Module): 
    def __init__(self, S=4, M=64, K=16, L=8, ang=1):
        super(Add_CFO, self).__init__()
        self.ang = ang
        self.S = S
        self.M = M
        self.K = K
        self.L = L
    def forward(self, input, isTrick, isRandom):
        # Input size:  NxPxSx(M+K)x2
        N = input.shape[0]
        if isTrick:
            index = torch.arange(-self.K, self.M).float()
            if isRandom:
                angs = torch.rand(N)*2*self.ang-self.ang
            else:
                angs = torch.ones(N)*self.ang 
            angs_all = torch.ger(angs, index).repeat((1,self.S+1)).view(N, self.S+1, self.M+self.K)    # Nx(S+1)x(M+K)
        else:
            index = torch.arange(0, (self.S+1)*(self.M+self.K)).float()
            if isRandom:
                angs = torch.rand(N)*2*self.ang-self.ang
            else:
                angs = torch.ones(N)*self.ang
            angs_all = torch.ger(angs, index).view(N, self.S+1, self.M+self.K)    # Nx(S+1)x(M+K)

        # Randomly generate the CFO for each batch

        real = torch.cos(angs_all/360*2*PI).unsqueeze(1).unsqueeze(4)   # Nx1xSx(M+K)x1 
        imag = torch.sin(angs_all/360*2*PI).unsqueeze(1).unsqueeze(4)   # Nx1xSx(M+K)x1

        real_in = input[:,:,:,:,0].unsqueeze(4)    # NxPx(Sx(M+K))x1 
        imag_in = input[:,:,:,:,1].unsqueeze(4)    # NxPx(Sx(M+K))x1

        real_out = real*real_in - imag*imag_in
        imag_out = real*imag_in + imag*real_in

        return torch.cat((real_out, imag_out), dim=4) 


class Add_Pilot(nn.Module):
    def __init__(self):
        pass
    def forward(self, x, pilot):
        signal = torch.cat((pilot, x), 2)

        return signal


class Channel(nn.Module):
    def __init__(self, S=5, M=64, K=16, decay=8, L=16):
        super(Channel, self).__init__()

        # Assign the power delay spectrum
        self.decay = decay
        self.length = L
        self.M = M
        MK = M+K
        # Generate unit power profile
        power = torch.exp(-torch.arange(self.length).float()/self.decay).unsqueeze(0).unsqueeze(0).unsqueeze(3)  # 1x1xLx1
        self.power = power/torch.sum(power)

        # Generate the index for batch convolution
        self.index = toeplitz(np.arange(S*MK-1, 2*S*MK+L-2), np.arange(S*MK-1,-1,-1))

    def forward(self, input, cof=None):
        # Input size:   NxPx(Sx(M+K))x2
        # Output size:  NxPx(L+Sx(M+K)-1)x2

        # Generate Channel Matrix
        N, P, SMK, _ = input.shape
        L = self.length

        if cof is None:
            cof = torch.sqrt(self.power/2) * torch.randn(N, P, L, 2)       # NxPxLx2

        cof_true = torch.cat((cof, torch.zeros((N,P,self.M-L,2))), 2)  # NxPxLx2
        H_true = torch.fft(cof_true, 1)  # NxPxLx2

        cof = torch.cat((torch.zeros((N,P,SMK-1,2)),cof,torch.zeros((N,P,SMK-1,2))), 2)   # NxPx(2xSMK+L-2)x2,   zero-padding

        channel = cof[:,:,self.index,:].cuda()                       #  NxPx(L+SMK-1)xSMKx2
        H_real = channel[:,:,:,:,0].view(N*P, L+SMK-1, SMK)   # (NxP)x(L+SMK-1)xSMK
        H_imag = channel[:,:,:,:,1].view(N*P, L+SMK-1, SMK)   # (NxP)x(L+SMK-1)xSMK
        
        signal_real = input[:,:,:,0].view(N*P, SMK, 1)       # (NxP)x(Sx(M+K))x1
        signal_imag = input[:,:,:,1].view(N*P, SMK, 1)       # (NxP)x(Sx(M+K))x1

        output_real = torch.bmm(H_real, signal_real) - torch.bmm(H_imag, signal_imag)   # (NxP)x(L+SMK-1)x1
        output_imag = torch.bmm(H_real, signal_imag) + torch.bmm(H_imag, signal_real)   # (NxP)x(L+SMK-1)x1

        output = torch.cat((output_real, output_imag), -1)   # (NxP)x(L+SMK-1)x2

        return output.view(N,P,L+SMK-1,2), H_true


class OFDM_channel(nn.Module):
    def __init__(self, N, P, S, M, K, L, SNR=20, pwr=1, A=None, Ang=None, pilot=None):
        super(OFDM_channel, self).__init__()
        self.N = N    # Batch size
        self.P = P    # Number of Packets
        self.S = S    # Number of symbols per packet
        self.M = M    # Number of sub-carriers per symbol
        self.K = K    # CP length
        self.L = L    # Number of paths for multipath fading channel
        self.A = A    # Clipping threshold
        self.Ang = Ang    # CFO angle/sample

        # Setup the add & remove CP layers
        self.add_cp = Add_CP(length=K)
        self.rm_cp = RM_CP(length=K)

        # Setup the channel layer
        self.channel = Channel(S=S+1, M=M, K=K, decay=L//2, L=L)

        if A is not None:
            self.clip = Clip(A)
        if Ang is not None:
            self.cfo = Add_CFO(S=S, M=M, K=K, L=L, ang=Ang)

        # Generate the pilot signal
        if pilot is None:
            pilot = torch.randn(N,P,1,M,2)
        
        pilot, _ = Normalize(pilot, pwr=pwr)
        pilot =  torch.ifft(pilot, 1)      #NxPx1xMx2  => NxPx1xMx2

        #if A is not None:
        #    pilot = self.clip(pilot)       #NxPx1xMx2  => NxPx1xMx2

        self.pilot = self.add_cp(pilot)         #NxPx1x(M+K)x2  => NxPx1x(M+K)x2
        self.SNR = SNR
        self.pwr = pwr

    def forward(self, x, SNR=None, cof=None, isTest=False, isTrick=False, isRandom=True):

        with torch.no_grad():
            if isTest:
                pwr = torch.mean(x**2, (-2,-1), True)     # NxPxSx1x1
                pwr = torch.cat((0.5*torch.ones(self.N, self.P, 1, 1, 1), pwr),2)

        # IFFT:                    NxPxSxMx2  => NxPxSxMx2
        x = torch.ifft(x, 1)

        # Add Cyclic Prefix:       NxPxSxMx2  => NxPxSx(M+K)x2
        x = self.add_cp(x)

        # Add pilot:               NxPxSx(M+K)x2  => NxPx(S+1)x(M+K)x2
        x = torch.cat((self.pilot, x), 2)    

        # Reshape: 
        x = x.view(self.N, self.P, (self.S+1)*(self.M+self.K), 2)

        # Normalize 
        #x, alpha = Normalize(x, pwr=1)

        # Clipping (Optional):     NxPx(S+1)(M+K)x2  => NxPx(S+1)(M+K)x2
        if self.A is not None:
            x = self.clip(x)

        # Pass the Channel:        NxPx(S+1)(M+K)x2  =>  NxPx((S+1)(M+K)+L-1)x2
        y, H_true = self.channel(x, cof)
        
        # Peak Detection: (Perfect)    NxPx((S+1)(M+K)+L-1)x2  =>  NxPx(S+1)x(M+K)x2
        output = y[:,:,:(self.S+1)*(self.M+self.K),:].view(self.N, self.P, self.S+1, self.M+self.K, 2)


        if isTest:
            if SNR is not None:
                noise_pwr = pwr/(self.M*10**(0.1*SNR))
            else:
                noise_pwr = pwr/(self.M*10**(0.1*self.SNR))
        else:
            if SNR is not None:
                noise_pwr = self.pwr/(self.M*10**(0.1*SNR))  # Input pwr is 1 in default
            else:
                noise_pwr = self.pwr/(self.M*10**(0.1*self.SNR))

        noise = np.sqrt(noise_pwr/2)*torch.randn_like(output)

        output_ny = output + noise  #NxPx(S+1)x(M+K)x2

        # Add CFO:                  NxPx(S+1)x(M+K)x2 => NxPx(S+1)x(M+K)x2
        if self.Ang is not None:
            output_ny = self.cfo(output_ny, isTrick, isRandom)

        # Peak Detection: (Perfect)    NxPx((S+1)(M+K)+L-1)x2  =>  NxPx(S+1)(M+K)x2 
        
        y_pilot = output_ny[:,:,0,:,:].unsqueeze(2)         # NxPx1x(M+K)x2
        y_sig = output_ny[:,:,1:,:,:]                       # NxPxSx(M+K)x2

        # Remove Cyclic Prefix":   
        info_pilot = self.rm_cp(y_pilot)    # NxPx1xMx2
        info_sig = self.rm_cp(y_sig)        # NxPxSxMx2

        # FFT:                     
        info_pilot = torch.fft(info_pilot, 1)
        info_sig = torch.fft(info_sig, 1)

        return info_pilot, info_sig, H_true, noise_pwr

def complex_division(no, de):
    a = no[:,:,:,:,0]
    b = no[:,:,:,:,1]
    c = de[:,:,:,:,0]
    d = de[:,:,:,:,1]

    out_real = (a*c+b*d)/(c**2+d**2)
    out_imag = (b*c-a*d)/(c**2+d**2)

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_multiplication(x1, x2):
    real1 = x1[:,:,:,:,0]
    imag1 = x1[:,:,:,:,1]
    real2 = x2[:,:,:,:,0]
    imag2 = x2[:,:,:,:,1]

    out_real = real1*real2 - imag1*imag2
    out_imag = real1*imag2 + imag1*real2

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_conjugate(x):
    out_real = x[:,:,:,:,0]
    out_imag = -x[:,:,:,:,1]
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_amp(x):
    real = x[:,:,:,:,0]
    imag = x[:,:,:,:,1]
    return torch.sqrt(real**2 + imag**2).unsqueeze(-1)

def ZadoffChu(order, length, index=0):
    cf = length % 2
    n = np.arange(length)
    arg = np.pi*order*n*(n+cf+2*index)/length
    zado = np.exp(-1j*arg)
    zado_real = torch.from_numpy(zado.real).unsqueeze(-1).float()
    zado_imag = torch.from_numpy(zado.imag).unsqueeze(-1).float()
    return torch.cat((zado_real, zado_imag), 1)

def ZF_equalization(H_est, Y):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    return complex_division(Y, H_est)

def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    # Noise_pwr: NxPx1x1  
    no = complex_multiplication(Y, complex_conjugate(H_est))
    de = complex_amp(H_est)**2 + noise_pwr
    return no/de

def LS_channel_est(pilot_tx, pilot_rx):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPx1xMx2
    return complex_division(pilot_rx, pilot_tx)

def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPx1xMx2
    return complex_multiplication(pilot_rx, complex_conjugate(pilot_tx))/(1+noise_pwr)

if __name__ == '__main__':

    N = 3
    P = 3
    S = 4
    M = 64
    K = 16
    L = 8
    
    is_channel_code = 0
    SNR_list = np.arange(1,33,2)
    err_list = []
    error_Hest1_list = []
    error_Hest2_list = []
    error_LS_ZF_list = []
    error_LMMSE_MMSE_list = []
    error_MMSE_list = []
    error_1_list = []
    error_2_list = []
    error_3_list = []
    for n in range(SNR_list.shape[0]):
        if not is_channel_code:
            Num_test = 100
            Num_bit = 2
            SNR = SNR_list[n]
            qam = QAM(Ave_Energy=1, B=Num_bit)
            pilot = ZadoffChu(order=1, length=M)
            pilot, _ = Normalize(pilot.unsqueeze(0))
            pilot_rep = pilot.repeat((N,P,1,1)).unsqueeze(2)
            ofdm_channel = OFDM_channel(N, P, S, M, K, L, SNR=SNR, pilot=pilot_rep)

            error1 = 0
            error2 = 0
            error3 = 0
            error4 = 0

            error_Hest1 = []
            error_Hest2 = []
            error_LS_ZF = []
            error_LMMSE_MMSE = []
            error_MMSE = []
            for i in range(Num_test):

                tx_bits = np.random.randint(2, size=N*P*S*M*Num_bit)
                tx_sym = qam.Modulation(tx_bits)
                tx_real = torch.from_numpy(tx_sym.real).unsqueeze(1)
                tx_imag = torch.from_numpy(tx_sym.imag).unsqueeze(1)
                tx = torch.cat((tx_real, tx_imag), 1).view(N, P, S, M, 2).type(torch.FloatTensor)
    
                out_pilot, out_sig, H_true, noise_pwr = ofdm_channel(tx, isTest=True, isTrick=True, isRandom=True)

                H_est1 = LS_channel_est(pilot_rep, out_pilot)
                rx_sym1 = ZF_equalization(H_est1, out_sig).numpy()

                H_est2 = LMMSE_channel_est(pilot_rep, out_pilot, M*noise_pwr[:,:,0,:,:].unsqueeze(2))
                rx_sym2 = MMSE_equalization(H_est2, out_sig, M*noise_pwr[:,:,1:,:,:]).numpy()
                #rx_sym2 = ZF_equalization(H_est2, out_sig).detach().cpu().numpy()
                #rx_sym3 = ZF_equalization(H_true.unsqueeze(2), out_sig).numpy()

                rx_sym3 = MMSE_equalization(H_true.unsqueeze(2), out_sig, M*noise_pwr[:,:,1:,:,:]).numpy()

                error_Hest1.append(2*torch.mean((H_est1-H_true.unsqueeze(2))**2).item())
                error_Hest2.append(2*torch.mean((H_est2-H_true.unsqueeze(2))**2).item())
                error_LS_ZF.append(2*np.mean((rx_sym1-tx.numpy())**2))
                error_LMMSE_MMSE.append(2*np.mean((rx_sym2-tx.numpy())**2))
                error_MMSE.append(2*np.mean((rx_sym3-tx.numpy())**2))

                #sio.savemat('example_31dB_CFO_1.73.mat', {'H_est1': H_est1.numpy(), 'H_est2': H_est2.numpy(), \
                #    'H_true': H_true.numpy(), 'rx_sym1': rx_sym1, 'rx_sym2': rx_sym2, \
                #    'rx_sym3': rx_sym3, 'tx': tx.numpy()})
                
                rx_sym1 = rx_sym1[:,:,:,:,0] + rx_sym1[:,:,:,:,1]*1j
                rx_bits1 = qam.Demodulation(rx_sym1.flatten())

                rx_sym2 = rx_sym2[:,:,:,:,0] + rx_sym2[:,:,:,:,1]*1j
                rx_bits2 = qam.Demodulation(rx_sym2.flatten())

                rx_sym3 = rx_sym3[:,:,:,:,0] + rx_sym3[:,:,:,:,1]*1j
                rx_bits3 = qam.Demodulation(rx_sym3.flatten())

                error1 += np.sum(rx_bits1!=tx_bits)
                error2 += np.sum(rx_bits2!=tx_bits)
                error3 += np.sum(rx_bits3!=tx_bits)

                #print(i)

            error_Hest1_list.append(np.mean(error_Hest1))
            error_Hest2_list.append(np.mean(error_Hest2))
            error_LS_ZF_list.append(np.mean(error_LS_ZF))
            error_LMMSE_MMSE_list.append(np.mean(error_LMMSE_MMSE))
            error_MMSE_list.append(np.mean(error_MMSE))
            error_1_list.append(error1/(Num_test*N*P*S*M*Num_bit))
            error_2_list.append(error2/(Num_test*N*P*S*M*Num_bit))
            error_3_list.append(error3/(Num_test*N*P*S*M*Num_bit))

            print('Err Hest1: ' + str(np.mean(error_Hest1)))
            print('Err Hest2: ' + str(np.mean(error_Hest2)))  
            print('Err LS+ZF: ' + str(np.mean(error_LS_ZF)))  
            print('Err LMMSE+MMSE: ' + str(np.mean(error_LMMSE_MMSE)))  
            print('Err MMSE: ' + str(np.mean(error_MMSE)))

            print('SNR: ' + str(SNR) + ' Err1: ' + str(error1/(Num_test*N*P*S*M*Num_bit)) + \
                ' Err2: ' + str(error2/(Num_test*N*P*S*M*Num_bit)) + \
                ' Err3: ' + str(error3/(Num_test*N*P*S*M*Num_bit)) )

    sio.savemat('results_Clip_2dB.mat', {'error_Hest1_list': np.stack(error_Hest1_list), \
                'error_Hest2_list': np.stack(error_Hest2_list), \
                'error_LS_ZF_list': np.stack(error_LS_ZF_list), \
                'error_LMMSE_MMSE_list': np.stack(error_LMMSE_MMSE_list), \
                'error_MMSE_list': np.stack(error_MMSE_list), \
                'error_1_list': np.stack(error_1_list), \
                'error_2_list': np.stack(error_2_list), \
                'error_3_list': np.stack(error_3_list)})
    import pdb; pdb.set_trace()  # breakpoint deff0f8d //



    polar = Polar(N=rate*dim, K=dim, Design_SNR=SNR, bits=NUM_bits, center=np.squeeze(centers))

    pilot = ZadoffChu(order=1, length=M)
    pilot, _ = Normalize(pilot.unsqueeze(0))

    pilot_rep = pilot.repeat((N,P,1,1)).unsqueeze(2)

    ofdm_channel = OFDM_channel(N, P, S, M, K, L, SNR=10, pilot=pilot_rep)

    out_pilot, out_sig, H_true = ofdm_channel(tx)

    #h_est = complex_division(out_pilot, pilot)

    sio.savemat('info', {'pilot': pilot.numpy(), 'tx': tx.numpy(), 'H_true': H_true.numpy(), 'out_pilot': out_pilot.numpy(), 'out_sig': out_sig.numpy()})
    import pdb; pdb.set_trace()  # breakpoint 042ac99f //

    
    sio.savemat('h_est', {'h': h_est.numpy(), 'input': input.numpy(), 'pilot': pilot.numpy(), 'out_pilot': out_pilot.numpy(), 'out_sig': out_sig.numpy()})
    sig_est = complex_division(out_sig, h_est)
    import pdb; pdb.set_trace()  # breakpoint e9171aae //


    
