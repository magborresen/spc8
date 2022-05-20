import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target
from simulator import Simulator

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
                _, alpha = radar.observation(k, theta[k], add_noise=True)
                plot_alpha.append(alpha[0])
                plot_dist.append(np.max(dists))
                cnt_k += 1
        plots_dist.append(plot_dist)
        plots_alpha.append(plot_alpha)

    plots_dist = np.array(plots_dist)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)
    for i, rho in enumerate(rho_list):
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
    for i, pwr in enumerate(power_list):
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

def plot_tx_signals(oversample=20):
    tx = Transmitter(channels=1, chirps=1, tx_power=30)
    rx = Receiver(channels=1)
    radar = Radar(tx, rx, "tdm", 2000, oversample=oversample)

    tx_sig = radar.transmitter.tx_tdm(radar.t_vec)[0]

    # Plot chirp signal
    plot_tx_sig = tx_sig[0:10000]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    tx_samples = np.nonzero(plot_tx_sig)
    plt.plot(radar.t_vec[tx_samples]/1e-6, plot_tx_sig[tx_samples].real)
    plt.grid()
    plt.xlabel('Time [\u03BCs]')
    plt.ylabel('Amplitude')
    fig.suptitle('First 10000 samples of chirp')
    fig.tight_layout()
    plt.savefig('plots/chirp_tx_sig.pdf', dpi=200)

    # Plot instantaneous frequency of chirp
    freq = np.linspace(0, radar.transmitter.bandwidth,
                       int(radar.transmitter.t_chirp * radar.receiver.f_sample))
    t = np.linspace(0, radar.transmitter.t_chirp,
                    int(radar.transmitter.t_chirp * radar.receiver.f_sample))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    plt.plot(t/1e-6, freq/1e6)
    plt.xlabel('Time [\u03BCs]')
    plt.ylabel('Frequency [MHz]')
    fig.suptitle('Instantaneous frequency of chirp')
    plt.grid()
    fig.tight_layout()
    plt.savefig('plots/inst_up_chirp_tx.pdf', dpi=200)

    # Plot chirp FFT of chirp
    fft_sig = np.fft.fft(tx_sig)
    N = len(fft_sig)
    T = N/(radar.receiver.f_sample*oversample)
    n = np.arange(N)
    freq = n/T
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    freq = np.fft.fftfreq(fft_sig.shape[-1], 1/(radar.receiver.f_sample*oversample))
    axs.plot(freq[:N//2]/1e6, np.abs(fft_sig[:N//2]))
    plt.grid()
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('|P(f)|')
    fig.suptitle('FFT of chirped signal')
    fig.tight_layout()
    plt.savefig('plots/tx_fft_plot_chirp.pdf', dpi=200)

    # Plot TX signals for observation
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    radar = Radar(tx, rx, "tdm", 2000, oversample=oversample)
    tx_sig = radar.transmitter.tx_tdm(radar.t_vec)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for idx, m_ch in enumerate(tx_sig):
        axs[idx].plot(radar.t_vec/1e-6, m_ch.real)
        axs[idx].set_title(f"TX channel {idx}")
        axs[idx].grid()
        axs[idx].set_ylabel('Amplitude')
    plt.xlabel("Time [µs]")
    fig.suptitle('TX signals for an observation')
    fig.tight_layout()
    plt.savefig('plots/tx_plot_2_ch.pdf', dpi=200)

def plot_rx_signals():
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=2)
    radar = Radar(tx, rx, "tdm", 2000, oversample=1)
    tx_sig = radar.transmitter.tx_tdm(radar.t_vec)
    theta = np.array([[1000],[1000],[8.49],[8.49]])
    alpha = radar.get_attenuation(theta)
    tau_vec = radar.time_delay(theta, radar.t_vec)
    _, rx_sig = radar.receiver.rx_tdm(tau_vec,
                                    tx_sig,
                                    radar.transmitter.f_carrier,
                                    alpha,
                                    radar.t_vec,
                                    radar.transmitter.t_chirp)
    rx_sig, _ = radar.add_awgn(rx_sig, alpha)

    # Plot RX signals
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for idx, m_ch in enumerate(rx_sig):
        axs[idx].plot(radar.t_vec/1e-6, m_ch.real)
        axs[idx].set_title(f"RX channel {idx}")
        axs[idx].grid()
        axs[idx].set_ylabel('Amplitude')
    plt.xlabel("Time [µs]")
    fig.suptitle('RX signals for an observation')
    fig.tight_layout()
    plt.savefig('plots/rx_plot_2_ch.pdf', dpi=200)

def plot_mixed_signals():
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=2)
    radar = Radar(tx, rx, "tdm", 2000, oversample=1)
    tx_sig = radar.transmitter.tx_tdm(radar.t_vec)
    theta = np.array([[1000],[1000],[8.49],[8.49]])
    alpha = radar.get_attenuation(theta)
    tau_vec = radar.time_delay(theta, radar.t_vec)
    _, rx_sig = radar.receiver.rx_tdm(tau_vec,
                                    tx_sig,
                                    radar.transmitter.f_carrier,
                                    alpha,
                                    radar.t_vec,
                                    radar.transmitter.t_chirp)
    rx_sig, _ = radar.add_awgn(rx_sig, alpha)

    ## MIXER FUNCTION (this version pads with zeros)
    chirp_len = int(radar.receiver.f_sample * radar.transmitter.t_chirp)
    # Chirp start samples
    samples = np.array([[tx + radar.m_channels*chirp
                        for chirp in range(radar.transmitter.chirps)]
                        for tx in range(radar.m_channels)]) * chirp_len
    mix_vec = []
    for n_ch, y in enumerate(rx_sig):
        # Find echo start sample
        y_sig_start = np.argmax(y > 0)
        rx_mix_vec = []
        # for m_ch, x in enumerate(tx_sig):
        #     for chirp in range(self.transmitter.chirps):
        for chirp in range(radar.transmitter.chirps):
            for m_ch, x in enumerate(tx_sig):
                # Determine start/stop samples of chirp
                chirp_start = y_sig_start + samples[m_ch][chirp]
                chirp_stop = chirp_start + chirp_len
                # Mix signal
                mix_sig = np.zeros(x.shape, dtype=np.complex128)
                mix_sig[chirp_start:chirp_stop] = y[chirp_start:chirp_stop]
                mix_sig = np.conjugate(mix_sig) * x
                rx_mix_vec.append(mix_sig)
        mix_vec.append(rx_mix_vec)
    mix_vec = np.flip(np.array(mix_vec), axis=0)

    # Plot mixed signals
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for n_ch in range(mix_vec.shape[0]):
        for chirp in range(mix_vec.shape[1]):
            axs[n_ch].plot(radar.t_vec/1e-6, mix_vec[n_ch][chirp].real, label=f'$n_{chirp}$')
        axs[n_ch].set_title(f"RX channel: {n_ch}")
        axs[n_ch].grid()
        axs[n_ch].set_ylabel('Amplitude')
        axs[n_ch].legend(loc='center right')
    plt.xlabel("Time [µs]")
    fig.suptitle('Mixed signals for an observation')
    fig.tight_layout()
    plt.savefig('plots/mixed_plot_2_ch.pdf', dpi=200)

    range_cube, range_est = radar.range_fft_cube(mix_vec)
    # Plot FFT of mixed signals
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    # fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    N = mix_vec.shape[2]
    T = N/(radar.receiver.f_sample * radar.oversample)
    n = np.arange(N)
    freq = n/T
    fft_range = (freq * radar.light_speed /
                (2 * radar.transmitter.bandwidth/radar.transmitter.t_chirp))
    sample = np.array([np.argmax(range_cube[n_ch][chirp])
                for chirp in range(radar.m_channels * radar.transmitter.chirps)
                for n_ch in range(radar.n_channels)])
    offset = 5
    for n_ch in range(mix_vec.shape[0]):
        for chirp in range(mix_vec.shape[1]):
            measure = fft_range[np.argmax(range_cube[n_ch][chirp])]
            axs[n_ch].plot(fft_range, np.abs(range_cube[n_ch][chirp]), label=f'$n_{chirp}={measure} m$')
            # axs2.plot(fft_range[np.min(sample)-offset:np.max(sample+1)+offset],
            #           np.abs(range_cube[n_ch][chirp])[np.min(sample)-offset:np.max(sample+1)+offset],
            #           label=f'$n_{chirp}={measure} m$')
        axs[n_ch].set_title(f"RX channel: {n_ch}")
        axs[n_ch].grid()
        axs[n_ch].set_ylabel('|P(f)|')
        axs[n_ch].legend(loc='center right')
    plt.xlabel("Range [m]")
    fig.suptitle('Range-FFT of mixed signals')
    fig.tight_layout()
    plt.savefig('plots/mixed_fft_plot_2_ch.pdf', dpi=200)

def plot_target(itr = 10):
    target = Target(1, velocity=16)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True)

    for idx in range(itr):
        theta = target.generate_states(300, method='random')
        axs.scatter(theta[:,0], theta[:,1], label=f'Example {idx}', s=1)
    axs.set_xlim(0, 2000)
    axs.set_ylim(0, 2000)
    axs.set_aspect(1)
    plt.ylabel('$x$ [m]')
    plt.xlabel('$y$ [m]')
    plt.savefig('plots/target_examples.pdf', dpi=200)


def plot_likelihood_map(points=2000):
    """
        Plot the distrubtion of the weights given a target location
    """
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=5)
    radar = Radar(tx, rx, "tdm", 2000)
    target_pos = np.array([[500], [500], [0], [10]])
    target_range, _ = radar.observation(0, target_pos)
    particle_pos_x = np.linspace(0, radar.region, points)
    particle_pos_y = np.linspace(0, radar.region, points)
    particle_filter = ParticleFilter(radar.t_obs + radar.k_space, 2)
    likelihood_map = np.zeros((points, points))

    for idx, pos_y in enumerate(particle_pos_y):
        for idy, pos_x in enumerate(particle_pos_x):
            point_range = radar.get_true_dist(np.array([[pos_x], [pos_y]]))
            likelihood = particle_filter.get_likelihood(target_range, point_range)
            likelihood_map[idx][idy] = likelihood

    c_mesh = plt.pcolormesh(particle_pos_x, particle_pos_y, likelihood_map)
    plt.colorbar(c_mesh)
    plt.title(f"Likelihoods for target in {target_pos[0][0]}, {target_pos[1][0]}")
    plt.xlabel("Meters")
    plt.ylabel("Meters")
    plt.savefig("plots/likelihood_map.pdf")
    plt.show()

def plot_distribution_2d(ranges=2400):
    """
        Plot the distrubtion of the weights given a target location
    """
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=2)
    radar = Radar(tx, rx, "tdm", 2000)
    target_pos = np.array([[500], [500], [0], [10]])
    target_range, _ = radar.observation(0, target_pos)
    particle_ranges = np.linspace((0, 0), (radar.region, radar.region), ranges)
    particle_filter = ParticleFilter(radar.t_obs + radar.k_space, 2)
    likelihoods = []

    for _, particle_range in enumerate(particle_ranges):
        likelihoods.append(particle_filter.get_likelihood(target_range, particle_range))
    plt.plot(particle_ranges, likelihoods)
    plt.show()


def plot_distribution_3d(points=200):
    """
        Plot the distrubtion of the weights given a target location
    """
    tx = Transmitter(channels=2, chirps=2, tx_power=30)
    rx = Receiver(channels=5)
    radar = Radar(tx, rx, "tdm", 2000)
    target_pos = np.array([[500], [500], [0], [10]])
    target_range, _ = radar.observation(0, target_pos)
    particle_pos_x = np.linspace(0, radar.region, points)
    particle_pos_y = np.linspace(0, radar.region, points)
    particle_mesh = np.meshgrid(particle_pos_x, particle_pos_y)
    particle_filter = ParticleFilter(radar.t_obs + radar.k_space, 2)

    point_range = radar.get_true_dist(particle_mesh)
    likelihood = particle_filter.get_likelihood(target_range, point_range)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(particle_mesh[0], particle_mesh[1], likelihood, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.title("Map of likelihoods over observation region")
    ax.set_xlabel("Meters")
    ax.set_ylabel("Meters")
    ax.set_zlabel("Likelihood")
    plt.savefig("plots/likelihood_map_3d.pdf")
    plt.show()

if __name__ == '__main__':
    # plot_target()
    # plot_tx_signals()
    # plot_rx_signals()
    # plot_mixed_signals()

    # plot_alpha([0.5, 0.75, 1])
    # plot_sigma([20, 30, 50])
    #plot_likelihood_map(points=200)
    #plot_distribution_2d()
    plot_distribution_3d()