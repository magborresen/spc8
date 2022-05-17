import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target
from simulator import Simulator
from scipy.signal import butter, sosfilt

def plot_alpha(rho_list):
    plots_dist = []
    plots_alpha = []
    for _, rho in enumerate(rho_list):
        k_obs = 137
        tx = Transmitter(channels=2, chirps=2, tx_power=30)
        rx = Receiver(channels=2)
        radar = Radar(tx, rx, "tdm", 2000)
        target = Target(radar.t_obs + radar.k_space, velocity=16.6)
        plot_dist = []
        plot_alpha = []
        cnt_k = 0
        radar.rho = rho
        while cnt_k < k_obs-1:
            theta = target.generate_states(300, method='linear_away')
            for k in range(theta.shape[0]):
                dists = [np.sqrt((radar.rx_pos[0,rx_n] - theta[k][0])**2 + (radar.rx_pos[1,rx_n] - theta[k][1])**2) for rx_n in range(radar.n_channels)]
                if (np.array(dists) > 2500).any() or (np.array(dists) < 0).any() or cnt_k > k_obs-1:
                    break
                s_k, y_k, alpha = radar.observation(k, theta[k], add_noise=True)
                plot_alpha.append(alpha[0])
                plot_dist.append(np.max(dists))
                cnt_k += 1
        plots_dist.append(plot_dist)
        plots_alpha.append(plot_alpha)
    
    plots_dist = np.array(plots_dist)    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)
    for i in range(len(rho_list)):
        ax.plot(plots_dist[i], plots_alpha[i], label=f'\u03C1 = {rho_list[i]}')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlim([0, 2400])
    plt.legend()
    plt.xlabel('Range [m]')
    plt.ylabel('\u03B1')
    fig.suptitle('Attenuation over range')
    fig.tight_layout()
    plt.savefig('attenuation_vs_range.pdf', dpi=200)

def plot_sigma(power_list):
    plots_dist = []
    plots_noise = []
    for _, power in enumerate(power_list):
        k_obs = 137
        tx = Transmitter(channels=2, chirps=2, tx_power=power)
        rx = Receiver(channels=2)
        radar = Radar(tx, rx, "tdm", 2000)
        target = Target(radar.t_obs + radar.k_space, velocity=16.6)
        plot_dist = []
        plot_noise = []
        cnt_k = 0
        while cnt_k < k_obs-1:
            theta = target.generate_states(300, method='linear_away')
            theta_est = []
            for k in range(theta.shape[0]):
                dists = [np.sqrt((radar.rx_pos[0,rx_n] - theta[k][0])**2 + (radar.rx_pos[1,rx_n] - theta[k][1])**2) for rx_n in range(radar.n_channels)]
                if (np.array(dists) > 2500).any() or (np.array(dists) < 0).any() or cnt_k > k_obs-1:
                    break
                radar.observation(k, theta[k], add_noise=True)
                plot_dist.append(np.max(dists))
                plot_noise.append(radar.receiver.sigma_noise)
                cnt_k += 1
        plots_dist.append(plot_dist)
        plots_noise.append(plot_noise)
    plots_dist = np.array(plots_dist)
    plots_noise = np.array(plots_noise)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)
    for i in range(len(power_list)):
        ax.plot(plots_dist[i], plots_noise[i], label=f'TX power = {power_list[i]} dBm')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlim([0, 2400])
    plt.legend()
    plt.xlabel('Range [m]')
    plt.ylabel(f'$\sigma_w^2$')
    fig.suptitle('Noise variance over range')
    fig.tight_layout()
    plt.savefig('variance_vs_range.pdf', dpi=200)

def plot_tx_fft():
    tx = Transmitter(channels=1, chirps=1, tx_power=30)
    rx = Receiver(channels=1)
    radar = Radar(tx, rx, "tdm", 2000)
    
    tx_sig = radar.transmitter.tx_tdm(radar.t_vec)[0]
    tx_sig = tx_sig[np.nonzero(tx_sig)]
    # plt.plot(tx_sig.real)
    fft_sig = np.fft.fft(tx_sig)
    N = len(fft_sig)
    T = N/(radar.receiver.f_sample*10)
    n = np.arange(N)
    freq = n/T
    # dBm = 10*np.log10()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    axs.plot(freq/1e6, np.abs(fft_sig))
    # axs.plot(freq[:N//2], 2.0/N * np.abs(fft_sig[:N//2]))
    

if __name__ == '__main__':
    # plot_alpha([0.5, 0.75, 1])
    # plot_sigma([20, 30, 50])
    plot_tx_fft()