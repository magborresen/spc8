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

def monte_carlo(k_obs):
    # Create objects
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=2)
    radar = Radar(tx, rx, "tdm", 2000, snr=50)
    target = Target(radar.t_obs + radar.k_space, velocity=16.6)
    pf = ParticleFilter(radar.t_obs + radar.k_space, rx.channels, n_particles=5, region=2000)

    # Setup lists to save data
    theta_error = []
    range_error = []
    range_RMSE = []
    theta_RMSE = []
    cnt_k = 0
    # Start simulating
    while cnt_k < k_obs-1:
        print(f'\nk={cnt_k}/{k_obs-1}')
        theta = target.generate_states(200, method='random')
        # radar.plot_region(theta)#, title=f'Simulation {cnt_k+1}')
        theta_est = []
        for k in range(theta.shape[0]):
            # Calculate distances from all antennas to target
            dists = [np.sqrt((radar.rx_pos[0,rx_n] - theta[k][0])**2 + (radar.rx_pos[1,rx_n] - theta[k][1])**2) for rx_n in range(radar.n_channels)]
            print(f'k={k}/{k_obs-1}, true distance={max(dists)[0]}')
            
            # Stop if target is out of region
            if (np.array(dists) > 2300).any() or (np.array(dists) < 23).any() or cnt_k > k_obs-1:
                print(f'Stopped, k+={k}')
                break
            
            # Generate radar observation
            s_k, y_k, alpha = radar.observation(k, theta[k], add_noise=True)
            
            # Estimate range
            range_est = pf.get_range(y_k, radar.transmitter.slope)
            
            # Range error and RMSE
            range_error.append(np.array([dists[n]-range_est[n] for n in range(radar.n_channels)]))
            range_RMSE.append(np.sqrt(1/(cnt_k+1) * np.sum(np.square(range_error))))
            
            # Estimate target trajectory
            estimate = theta[k] + np.random.normal(0,0.1,4)
            theta_est.append(estimate)
            
            # Target trajectory error and RMSE
            theta_error.append(theta[k] - theta_est[k])
            theta_RMSE.append(np.sqrt(1/(cnt_k+1) * np.sum(np.square(theta_error))))

            cnt_k += 1

    theta_RMSE = np.array(theta_RMSE)
    range_RMSE = np.array(range_RMSE)
    
    print(f'\nRMSE:\n Theta: {theta_RMSE[-1]}\n Range: {range_RMSE[-1]}')
    

def plot_parameters(power_list, power_alpha):
    plots_dist = []
    plots_noise = []
    plots_alpha = []
    for _, power in enumerate(power_list):
        k_obs = 137
        tx = Transmitter(channels=2, chirps=2, tx_power=power)
        rx = Receiver(channels=2)
        radar = Radar(tx, rx, "tdm", 2000, snr=50)
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
                s_k, y_k, alpha = radar.observation(k, theta[k], add_noise=True)
                if power == power_alpha:
                    plots_alpha.append(alpha[0])
                plot_dist.append(np.max(dists))
                plot_noise.append(radar.receiver.sigma_noise)
                cnt_k += 1
        plots_dist.append(plot_dist)
        plots_noise.append(plot_noise)
    
    # Plot noise variance over range
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
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)
    idx = power_list.index(power_alpha)
    plt.plot(plots_dist[idx], np.array(plots_alpha))
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlim([0, 2400])
    plt.xlabel('Range [m]')
    plt.ylabel('\u03B1')
    fig.suptitle('Attenuation over range')
    fig.tight_layout()
    plt.savefig('attenuation_vs_range.pdf', dpi=200)

if __name__ == '__main__':
    
    # plot_parameters([20, 30, 50], 30)
    monte_carlo(100)