# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:39:39 2022

@author: bjark
"""

import numpy as np
import matplotlib.pyplot as plt
from receiver import Receiver
from transmitter import Transmitter
from target import Target


def add_noise(signal, SNR_dB):
    samples = len(signal)
    SNR = 10.0**(SNR_dB/10.0)
    s_var = np.var(signal)
    W_var = s_var/SNR
    
    v_var = np.sqrt(W_var/2)
    
    v = np.random.normal(0, 1, size=(2, samples))
    W = v_var * v[0,:] + 1j * v_var * v[1,:]
    
    print(10.0*np.log10(s_var/W_var), 10.0*np.log10(s_var/np.var(W)))
    return sig + W    
    
samples_tot = 100
desired_dB = [30, 20, 10, 0]
t = np.linspace(0, 2*np.pi, samples_tot)
sig = np.cos(t) + 1j * np.sin(t)

fig, axs = plt.subplots(len(desired_dB), 2, sharex=True, sharey=False, figsize=(6.4, 4.8))
for i in range(len(desired_dB)):
    sig_n = add_noise(sig, desired_dB[i])
    axs[i, 0].plot(sig.real, label='Without noise')
    axs[i, 0].plot(sig_n.real, label='With noise')
    axs[i, 1].plot(sig.imag, label='Without noise')
    axs[i, 1].plot(sig_n.imag, label='With noise')
    axs[i, 0].set_title(f'Real part ({desired_dB[i]} dB SNR)')
    axs[i, 1].set_title(f'Imaginary part ({desired_dB[i]} dB SNR)')
fig.supxlabel('Samples')
fig.supylabel('Signal amplitude')
fig.legend(['No noise', 'With noise'], bbox_to_anchor=(1,0.5), loc="center left", borderaxespad=0)
fig.tight_layout()

plt.savefig('signal_noise.pdf', dpi=200)