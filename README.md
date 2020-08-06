# JSCC

Run **train.py** for BSC or AWGN channel

Run **train_OFDM.py** for multipath channel based on OFDM system

So far, can only work with **CIFAR-10** dataset. Currently working to include **CelebA**

Three GANs available:  **vanilla GAN, LSGAN, WGAN**

**BSC channel**:

* Soft and hard Gunbel Softmax relaxed Bernoulli distribution

**OFDM system**:

* LS, LMMSE channel estimation
* ZF, MMSE, implicit equalization
* A version of feedback of CSI included

## visdom compatible 

* ssh port forwarding from server to local machine `ssh -N -f -L localhost:8998:localhost:8998 username@server` 
* Then view the training dynamics in `localhost:8998` with local browser

Use **nohup** to run multiple threads

The basic coding framework is based on pix2pix [Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
