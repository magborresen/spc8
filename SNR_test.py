# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:42:27 2022

@author: bjark
"""
import numpy as np
import matplotlib.pyplot as plt
from signal_model import Signal
import time
import math
import scipy.stats as stats

from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal

def add_noise(observations : list, SNR_dB=1.0):
    """ Generate k input vectors for the system.
            Right now this function sets all acceleration to 0!

            Args:
                k_tot (int): Observations

            Returns:
                acc (list): Acceleration for each observation
    """
    obs = np.array(observations)
    SNR = 10.0**((SNR_dB)/10.0) # Desired linar SNR
    for i in range(len(obs[0])):
        s_power = np.var(obs[:,i]) # Power of signal
        n_power = s_power/SNR # Desired power of noise
        # Generate noise:
        noise = np.random.multivariate_normal((0,0), (n_power/2)*np.eye(2), len(obs))
        noise = noise.view(np.complex128) # From [A, B] to [A + 1j*B]
        #print('Original:\n',noise,'Please:',np.exp(1j*10)*noise)
        obs[:,i] = [sum(x)[0] for x in zip(obs[:,i], noise)]  
    return obs.tolist()

def plot_distributions(observations, SNR_dB=30.0):
    # First, make a realization to get the info:
    obs = np.array(observations)
    SNR = 10.0**((SNR_dB)/10.0)
    s_power = np.var(obs[:,0])
    n_power = s_power/SNR
    #n_power = math.sqrt(n_power)
    # Then we can realize the noise
    noise = np.random.multivariate_normal((0,0), (n_power/2)*np.eye(2), len(obs))
    mu = 0
    re = np.sort(noise[:,0])
    im = np.sort(noise[:,1])
    
    # Plotting 1D pdf:
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    fig.suptitle('Probability density function of multivariate normal distribution')
    axs[0].plot(re, stats.norm.pdf(re, mu, n_power))
    axs[0].set_title('Real part')
    axs[1].plot(re, stats.norm.pdf(im, mu, n_power))
    axs[1].set_title('Imaginary part')
    plt.tight_layout()
    plt.xlabel(r'x')
    plt.ylabel(r'$f_x(x)$')
    axs[0].grid()
    axs[1].grid()
    plt.show()
    
    # Plotting 2D pdf:
    X, Y = np.meshgrid(re, im)
    pos = np.dstack((X, Y))
    mu = np.array([0, 0])
    cov = (n_power/2)*np.eye(2)
    rv = multivariate_normal(mu, cov)
    print(np.var(noise), np.var(obs))
    Z = rv.pdf(pos)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('Real', rotation=-10)
    ax.set_ylabel('Imaginary')
    ax.set_zlabel('Probability density', rotation=65)
    plt.show()

if __name__ == '__main__':
    SNR_dB = 30.0
    
    t0 = time.time()
    sig = Signal(10000, [2000, 2000], 4e-4, 10, 10) #tx,rx
    td = time.time() - t0
    print('Signal:       ', td, 's')
    
    #sig.plot_region()
    
    t0 = time.time()
    obs = sig.observe()
    td = time.time() - t0
    print('Observation:  ', td, 's')
    
    t0 = time.time()
    obs_n = add_noise(obs, SNR_dB)
    td = time.time() - t0
    print('Adding noise: ', td, 's')
    
    obs = np.array(obs)
    obs_n = np.array(obs_n)    
    s_power = np.var(obs[:,1])
    n_power = np.var(obs[:,1] - obs_n[:,1])
    print('Desired SNR:  ', SNR_dB)
    print('Resulting SNR:', 10.0*np.log10(s_power/(n_power)))
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Comparison of observations for receiver 1')
    axs[0].plot(range(len(obs[:,0].real)), obs[:,0].real, label='Ogirignal')
    axs[0].plot(range(len(obs_n[:,0].real)), obs_n[:,0].real, label='With noise')
    axs[0].set_title('Real part')
    axs[1].plot(range(len(obs[:,0].imag)), obs[:,0].imag, label='Ogirignal')
    axs[1].plot(range(len(obs_n[:,0].imag)), obs_n[:,0].imag, label='With noise')
    axs[1].set_title('Imaginary part')
    plt.ylabel('Signal amplitude')
    plt.xlabel('Observation')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plot_distributions(obs, SNR_dB)