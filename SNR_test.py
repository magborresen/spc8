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

def add_nice(obs, SNR_dB=30.0):
    SNR = 10.0**((SNR_dB)/10.0) # Desired linar SNR
    rec = len(obs)
    obs_n = []
    res_dB = []
    for r in range(rec):
        s_power = np.var(obs[r])
        n_power = s_power/SNR
        print('Receiver', r, (np.linalg.norm(obs[r])**2)/len(obs[r]))
        #print('Receiver', r, 's_power', (sum(abs(obs[r]))**2)/len(2*obs[r]+1))
        noise = np.random.multivariate_normal((0,0), (n_power/2)*np.eye(2), len(obs[r]))
        noise = noise.view(np.complex128)
        obs_n.append([sum(x)[0] for x in zip(obs[r], noise)])
        #print('Resulting SNR:', 10.0*np.log10(s_power/(np.var(noise))), 'dB')
        dB = 10.0*np.log10(s_power/(np.var(noise)))
        res_dB.append(dB)
    return np.array(obs_n), res_dB

def plot_distributions_nice(observations, SNR_dB=10.0):
    SNR = 10.0**((SNR_dB)/10.0)
    s_power = np.var(obs[0])
    n_power = s_power/SNR
    noise = np.random.multivariate_normal((0,0), (n_power/2)*np.eye(2), len(obs[0][0]))
    noice = noise.view(np.complex128)
    
    print(len(noice))
    print('Resulting SNR:', 10.0*np.log10(s_power/(np.var(noice))), 'dB')

    re = np.sort(noise[:,0])
    im = np.sort(noise[:,1])
    
    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(4,3))
    fig.add_subplot(111, frame_on=False)
    #plt.tick_params(labelcolor="none", bottom=False, left=False)
    #fig.suptitle('Multivariate distribution')
    axs[0].plot(re, stats.norm.pdf(re, 0, n_power))
    axs[0].set_title('Real part')
    axs[1].plot(re, stats.norm.pdf(im, 0, n_power))
    axs[1].set_title('Imaginary part')
    plt.tight_layout()
    #plt.xlabel(r'$W_k$')
    #plt.ylabel('Density')
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Density')
    #plt.ylabel(r'$W_k$')
    axs[0].set_xlabel(r'$X_k$')
    axs[1].set_xlabel(r'$Y_k$')
    axs[0].grid()
    axs[1].grid()
    plt.savefig('multivariate.pdf', dpi=300)
    plt.show()
    
    X, Y = np.meshgrid(re, im)
    pos = np.dstack((X, Y))
    mu = np.array([0, 0])
    cov = (n_power/2)*np.eye(2)
    rv = multivariate_normal(mu, cov)
    print(np.var(noise), np.var(obs))
    Z = rv.pdf(pos)
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('Real', rotation=-10)
    ax.set_ylabel('Imaginary')
    ax.set_zlabel('Probability density', rotation=65)
    plt.savefig('complex.pdf', dpi=300)
    plt.show()
    
    
    return None


if __name__ == '__main__':
    obs_k = 10
    sig = Signal(obs_k, [2000, 2000], 4e-4, 10, 10) #tx,rx
    obs = []
    obs_n = []
    for i in range(obs_k):
        obs.append(sig.observe_y(i, 0))
        obs_n.append(add_nice(obs[i]))
        
    s_power = np.var(obs[0][0])
    n_power = np.var(obs_n[0][0] - obs[0][0])
    #print('Resulting SNR:', 10.0*np.log10(s_power/(n_power)))
    
    # fig, axs = plt.subplots(2)
    # fig.suptitle('First 100 samples for 1 receiver observation')
    # axs[0].plot(range(len(obs[0][0][0:100].real)), obs[0][0][0:100].real, label='No noise')
    # axs[0].plot(range(len(obs_n[0][0][0:100].real)), obs_n[0][0][0:100].real, label='With noise')
    # axs[0].set_title('Real part')
    # axs[1].plot(range(len(obs[0][0][0:100].imag)), obs[0][0][0:100].imag, label='No noise')
    # axs[1].plot(range(len(obs_n[0][0][0:100].imag)), obs_n[0][0][0:100].imag, label='With noise')
    # axs[1].set_title('Imaginary part')
    # plt.ylabel('Signal amplitude')
    # plt.xlabel('Observation')
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    
    plot_distributions_nice(obs[0], SNR_dB=30.0)
    
    dBs = [5, 15, 30]
    dB_results = []
    actual_dB = []
    for dB in dBs:
        print('Running for', dB, 'dB')
        res, res_dB = add_nice(obs[0], dB)
        dB_results.append(res)
        actual_dB.append(res_dB[0])
    
    pnts = 10
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle('First 10 samples for 1 receiver observation')
    axs[0].plot(range(1,1+len(obs[0][0][0:pnts].real)), obs[0][0][0:pnts].real, label='Original signal', color="blue", alpha=8/10)
    axs[0].plot([], [], ' ', label="Desired dB : Actual dB")
    axs[0].plot(range(1,1+len(dB_results[0][0][0:pnts].real)), dB_results[0][0][0:pnts].real, label=f'5 dB : {actual_dB[0]:.2f} dB', color="red", alpha=8/10)
    axs[0].plot(range(1,1+len(dB_results[1][0][0:pnts].real)), dB_results[1][0][0:pnts].real, label=f'15 dB : {actual_dB[1]:.2f} dB', color="green", alpha=8/10)
    axs[0].plot(range(1,1+len(dB_results[2][0][0:pnts].real)), dB_results[2][0][0:pnts].real, label=f'30 dB : {actual_dB[2]:.2f} dB', color="orange", alpha=8/10)
    axs[0].set_title('Real part')
    #axs[0].legend()
    
    
    axs[0].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    
    axs[1].plot(range(1,1+len(obs[0][0][0:pnts].imag)), obs[0][0][0:pnts].imag, color="blue")
    axs[1].plot(range(1,1+len(dB_results[0][0][0:pnts].imag)), dB_results[0][0][0:pnts].imag, color="red", alpha=8/10)
    axs[1].plot(range(1,1+len(dB_results[1][0][0:pnts].imag)), dB_results[1][0][0:pnts].imag, color="green", alpha=8/10)
    axs[1].plot(range(1,1+len(dB_results[2][0][0:pnts].imag)), dB_results[2][0][0:pnts].imag, color="orange", alpha=8/10)
    axs[1].set_title('Imaginary part')
    #axs[1].legend()
    
    plt.ylabel(' ')
    plt.xlabel('Sample number')
    fig.text(0.04, 0.5, 'Signal amplitude', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig('signal.pdf', dpi=300)
    plt.show()
    