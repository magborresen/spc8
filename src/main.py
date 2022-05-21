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
    tx = Transmitter(channels=2, chirps=10, tx_power=30)
    rx = Receiver(channels=2)
    radar = Radar(tx, rx, "tdm", 2000)
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
            # print(f'k={k}/{k_obs-1}, true distance={max(dists)[0]}', end='\r')

            # Stop if target is out of region
            if (np.array(dists) > 2400).any() or (np.array(dists) < 1).any() or cnt_k > k_obs-1:
                print(f'Stopped, k+={k}')
                break

            # Generate radar observation
            range_est, alpha = radar.observation(k, theta[k], add_noise=True)

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

if __name__ == '__main__':
    monte_carlo(1000)