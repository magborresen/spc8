"""
    Radar system object consisting of transmitter(s) and receiver(s)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from receiver import Receiver
from transmitter import Transmitter
from target import Target
from scipy.constants import k as Boltzman

class Radar:
    """
        Radar system class

        Args:
            no value
        Returns:
            no value
    """

    def __init__(self, transmitter, receiver, mp_method, region, oversample=1):
        self.transmitter = transmitter
        self.receiver = receiver
        self.mp_method = mp_method
        self.region = region
        self.snr = None
        self.n_channels = self.receiver.channels
        self.m_channels = self.transmitter.channels
        self.t_rx = 20e-6
        self.t_obs = (self.transmitter.t_chirp * self.m_channels *
                      self.transmitter.chirps + self.transmitter.t_chirp)
        self.oversample = oversample
        self.samples_per_obs = int(self.receiver.f_sample * self.t_obs * self.oversample)
        self.light_speed = 300e6
        self.atmos_loss_factor = 0.02
        self.wavelength = self.light_speed / self.transmitter.f_carrier
        self.tx_pos = None
        self.rx_pos = None
        self.rx_tx_spacing = 0
        self.place_antennas()
        self.max_range = None
        self.min_range = self.transmitter.t_chirp * self.light_speed / 2
        self.range_res = self.light_speed / (2 * self.transmitter.bandwidth)
        self.k_space = 1
        self.t_vec = self.create_time_vector()
        self.rho = 0.5

    def place_antennas(self) -> None:
        """
            Place antennas in 2D region.

            Args:
                no value

            Returns:
                no value
        """
        self.tx_pos = np.array([(self.region / 2 + self.rx_tx_spacing / 2 -
                                  np.linspace(self.wavelength / 2,
                                              self.m_channels * self.wavelength / 2,
                                              self.m_channels)),
                                  np.zeros(self.m_channels)])

        self.rx_pos = np.array([(self.region / 2 + self.rx_tx_spacing / 2 +
                                np.linspace(self.wavelength * 2,
                                            self.n_channels * self.wavelength/2,
                                            self.n_channels)),
                                np.zeros(self.n_channels)])

    def plot_antennas(self) -> None:
        """ Plot antenna postions in 2D space

            Args:
                no value

            Returns:
                no value
        """
        offset = 0.5
        plt.scatter(self.tx_pos[0], self.tx_pos[1], label="TX Antennas")
        plt.scatter(self.rx_pos[0], self.rx_pos[1], label="RX Antennas")
        plt.legend()
        plt.xlim(self.tx_pos[0][-1]-offset, self.rx_pos[0][-1]+offset)
        plt.ylim(-offset, self.tx_pos[1][-1]+offset)
        plt.ylim()
        plt.title("RX/TX positions in the plane")
        plt.xlabel("Position [m]")
        plt.ylabel("Position [m]")
        plt.show()

    def plot_region(self, states, zoom=False):
        """
            Plots the observation region with antenna locations and trajectory

            Args:
                states (np.ndarray): Collection of all states
                zoom (bool): Show only trajectory region if true

            Returns:
                no value
        """
        _, ax = plt.subplots()
        ax.scatter(states[:,0], states[:,1], label="Trajectory")
        ax.scatter(self.tx_pos[0], self.tx_pos[1], label="TX Antennas")
        ax.scatter(self.rx_pos[0], self.rx_pos[1], label="RX Antennas")
        ax.set_aspect(1)
        if zoom:
            ax.set_xlim(min(states[:,0]), max(states[:,0]))
            ax.set_ylim(min(states[:,1]), max(states[:,1]))
        else:
            ax.set_xlim(0, self.region)
            ax.set_ylim(0, self.region)
        plt.title('Observation region with antennas and trajectory')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.legend()
        plt.show()

    def get_true_dist(self, theta):
        """
            Get the true distance of the target to each receiver antenna

            Args:
                theta (np.ndarray): Position and velocity vector of the target

            Returns:
                true_dist (list): List of eucledian distances to each of the receiver antennas
        """
        true_dist = [np.sqrt((self.rx_pos[0,rx_n] - theta[0])**2 +
                             (self.rx_pos[1,rx_n] - theta[1])**2)
                    for rx_n in range(self.n_channels)]

        return np.array(true_dist)

    def get_true_vel(self, theta):
        """
            Get the velocity of the target

            Args:
                theta (np.ndarray): Position and velocity vector of the target

            Returns:
                true_vel (float): True velocity of the target
        """
        true_vel = np.linalg.norm((theta[2], theta[3]))

        return true_vel

    def time_delay(self, theta: np.ndarray, t_vec: np.ndarray) -> list:
        """
            Find time delays from receiver each n, to the target, and back to
            every transmitter. Note that output dimension will be M*N

            Args:
                theta (np.ndarray): Target state at observation k
                t_vec  (np.ndarray): Times for which to calculate the delays

            Returns:
                tau (np.ndarray): Collection of time delay signals
        """

        traj = self.trajectory(t_vec, theta)

        tau = []
        for rx_n in range(self.n_channels):
            for tx_m in range(self.m_channels):
                d_tx = np.sqrt((self.tx_pos[0,tx_m] - traj[0].T)**2 +
                               (self.tx_pos[1,tx_m] - traj[1].T)**2)
                d_rx = np.sqrt((self.rx_pos[0,rx_n] - traj[0].T)**2 +
                               (self.rx_pos[1,rx_n] - traj[1].T)**2)

                tau_kmn = 1 / self.light_speed * (d_tx + d_rx)

                tau.append(tau_kmn)

        return np.array(tau)

    def trajectory(self, t_vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
            Calculate target trajectory within an acquisition period

            Angle A is betweeen the line-of-sight from origin to target, and
            the velocity vector of the drone. los is a unit vector
            representation of the line-of-sight.

            Args:
                t_vec (np.ndarray): time vector
                theta (np.ndarray): Target position

            Returns:
                r_k (np.ndarray): Short time trajectory model based
                                  on original position and velocity.
        """
        # Normalized unitvector for position (line-of-sight
        p_antenna = np.array([[self.region/2], [0]])
        p_offset = p_antenna - theta[:2]
        los = p_offset / np.linalg.norm(p_offset)
        # los = theta[:2] / np.linalg.norm(theta[:2])

        # Target trajectory within acquisition period
        r_k = theta[:2] + (t_vec - t_vec[0]) * ((los[0]*theta[2]) + (los[1]*theta[3]))
        # r_k = theta[:2] + (t_vec - t_vec[0]) * ((los[0]*theta[2]) + (los[1]*theta[3]))

        return r_k

    def get_attenuation(self, theta, target_rcs=0.04):
        """
            Calculate alpha to attenuate the received signal

            The target range used in this function, is equal to the echo range.

            Args:
                theta (np.ndarray): Target position
                target_rcs (float): Radar-cross-section
                rho (float): Effectivity of aperture (0 < rho <= 1)

            Returns:
                alpha (np.ndarray): Receiver attenuations
        """
        aperture = self.wavelength**2 * self.rho
        gain = 4 * np.pi * aperture / self.wavelength**2
        alpha = []
        for rx_n in range(self.n_channels):
            # Get range from receiver to target
            target_range = np.sqrt((self.rx_pos[0,rx_n] - theta[0])**2 +
                                   (self.rx_pos[1,rx_n] - theta[1])**2)

            # Determine atmospheric loss (Richards, p. 43)
            loss_atmospheric = 10**(self.atmos_loss_factor * target_range / 5000)

            # Calculate attenuation (Richards, p. 92)
            attenuation = (gain**2 * self.wavelength**2 * target_rcs /
                          ((4*np.pi)**3 * target_range**4 * loss_atmospheric))
            alpha.append(attenuation)

        return np.array(alpha)

    def add_awgn(self, signals, alpha):
        """
            Calculate and add circular symmetric white gaussian noise to the signal

            Args:
                sig (np.ndarray): The original signal
                sig_var (np.ndarray): Variance of the part of the
                                      observation containing the signal(s).

            Returns:
                sig_noise (np.ndarray): Signal with noise added.
        """
        signals_noise = []
        for signal in signals:
            sig_noise = np.copy(signal)
            # Find where the signal is non zero
            sig_idx = np.abs(signal) > 0
            # Index the original signal
            sig_only = signal[sig_idx]
            # Find the variance (power of the signal)
            sig_var = np.var(sig_only)

            # Determine spectral height of noise
            t_sig = self.m_channels * self.transmitter.chirps * self.transmitter.t_chirp
            fraq = t_sig / Boltzman * self.receiver.temp * self.receiver.noise_figure

            # # Calculate linear SNR
            # lin_snr = 10.0**(self.snr / 10.0)
            lin_snr = np.mean(alpha * fraq)
            self.snr = 10*np.log10(lin_snr)

            # Calculate the variance of the noise
            n_var = sig_var / lin_snr
            # Calculate the noise
            noise = np.sqrt(n_var / 2) * (np.random.normal(size=(sig_only.shape)) +
                                        1j*np.random.normal(size=(sig_only.shape)))

            sig_noise[sig_idx] = signal[sig_idx] + noise
            signals_noise.append(sig_noise)

        return np.array(signals_noise), np.var(noise)


    def create_time_vector(self):
        """
            Create a generic time vector

            Args:
                no value

            Returns:
                t_vec (np.ndarray): Array containing sample times
        """
        t_vec = np.linspace(0, self.t_obs, self.samples_per_obs)

        return t_vec

    def butter_lowpass_filter(self, sigs, cutoff, order=10):
        """
            Create and apply a low-pass Butterworth filter on multiple signals

            Args:
                sigs (np.ndarray): Collection of signals
                cutoff (float): Cutoff frequency for filter
                order (int): Filter order

            Returns:
                sigs_filtered (np.ndarray): Collection of filtered signals
        """
        # Create the filter
        nyq = self.receiver.f_sample * self.oversample
        sos = butter(order, cutoff, fs=nyq, btype='low', analog=False, output='sos')
        sigs_filtered = []
        # Apply the filter
        for _, sig in enumerate(sigs):
            sigs_filtered.append(sosfilt(sos, sig))
        return np.array(sigs_filtered)

    def signal_mixer(self, rx_sig, tx_sig):
        """
            This is a complex signal mixer, that seperates individual chirps
            in the received signals - and mixes them with the respective
            transmitted chirp. Since there will be a total of m_channels * N_c
            chirps, so will the output. Output is indexed by n_channels and
            chirp_index.

            Args:
                rx_sig (np.ndarray): Collection of received signals
                tx_sig (np.ndarray): Collection of transmitted signals

            Returns:
                mix_vec (np.ndarray): Collection of mixed signals
        """
        # Chirp length in samples
        chirp_len = int(self.receiver.f_sample * self.transmitter.t_chirp)
        # Chirp start samples
        samples = np.array([[tx + self.m_channels*chirp
                            for chirp in range(self.transmitter.chirps)]
                            for tx in range(self.m_channels)]) * chirp_len
        mix_vec = []
        for _, y in enumerate(rx_sig):
            # Find echo start sample
            y_sig_start = np.argmax(y > 0)
            rx_mix_vec = []
            # for m_ch, x in enumerate(tx_sig):
            #     for chirp in range(self.transmitter.chirps):
            for chirp in range(self.transmitter.chirps):
                for m_ch, x in enumerate(tx_sig):
                    # Determine start/stop samples of chirp
                    chirp_start = y_sig_start + samples[m_ch][chirp]
                    chirp_stop = chirp_start + chirp_len
                    # Mix signal
                    # mix_sig = np.zeros(x.shape, dtype=np.complex128)
                    mix_sig = y[chirp_start:chirp_stop]
                    mix_sig = np.conjugate(mix_sig) * x[chirp_start:chirp_stop]
                    rx_mix_vec.append(mix_sig)

            mix_vec.append(rx_mix_vec)
        mix_vec = np.flip(np.array(mix_vec), axis=0)
        return mix_vec

    def range_fft_cube(self, mix_vec):
        """
            This is a complex signal mixer, that seperates individual chirps
            in the received signals - and mixes them with the respective
            transmitted chirp. Since there will be a total of m_channels * N_c
            chirps, so will the output. Output is indexed by n_channels and
            chirp_index.

            Args:
                mix_vec (np.ndarray): Collection of mixed signals

            Returns:
                range_cube (np.ndarray): Three-dimensional range cube
        """
        # Prepare low-pass filter for mixed signals
        nyq = self.receiver.f_sample * self.oversample
        sos = butter(10, nyq/2-1, fs=nyq, btype='low', analog=False, output='sos')
        T = self.samples_per_obs/(self.receiver.f_sample * self.oversample)
        n = np.arange(self.samples_per_obs)
        freq = n/T
        sig_range = (freq * self.light_speed /
                    (2 * self.transmitter.bandwidth/self.transmitter.t_chirp))
        range_est = []
        range_cube = []
        for n_ch in range(self.n_channels):
            range_n = 0
            fft_vec = []
            for chirp_idx in range(self.m_channels * self.transmitter.chirps):
                # Find non-zero part of mixed signal
                sig = mix_vec[n_ch][chirp_idx]
                # Low-pass filter mixed signal
                sig_fil = sosfilt(sos, sig)
                # Get range-FFT of mixed signal
                sig_fft = np.fft.fft(sig_fil)
                N = len(sig_fft)
                T = N/(self.receiver.f_sample * self.oversample)
                n = np.arange(N)
                freq = n/T
                # Convert frequency to range
                sig_range = (freq * self.light_speed /
                            (2 * self.transmitter.bandwidth/self.transmitter.t_chirp))
                fft_vec.append(sig_fft)
                range_n += sig_range[np.argmax(sig_fft)]
            range_est.append(range_n / (self.m_channels * self.transmitter.chirps))
            range_cube.append(fft_vec)

        return np.array(range_cube), np.array(range_est)

    def velocity_fft_cube(self, range_cube):
        """
            THIS FUNCTION IS NOT FULLY FUNCTIONAL

            Args:
                range_cube (np.ndarray): Three-dimensional range cube

            Returns:
                none
        """
        samples = [np.argmax(range_cube[n_ch][chirp])
                    for chirp in range(self.m_channels * self.transmitter.chirps)
                    for n_ch in range(self.n_channels)]
        vel_est = np.zeros(range_cube.shape, dtype=np.complex128)
        for n_ch in range(self.n_channels):
            cube = range_cube[n_ch].T
            vel_cube = np.zeros(cube.shape, dtype=np.complex128)
            for idx in range(vel_cube.shape[0]):
                sig_fft = np.fft.fft(cube[idx])
                vel_cube[idx] = sig_fft
            vel_est[n_ch] = vel_cube.T
        velocity_table = 2*np.pi*np.concatenate((np.arange(0, (self.transmitter.chirps*self.m_channels)//2), np.arange(-(self.transmitter.chirps*self.m_channels)//2, 0)[::-1]))*self.transmitter.bandwidth/(self.transmitter.chirps*self.m_channels)
        velocity_table = velocity_table * self.light_speed / (4*np.pi*self.transmitter.f_carrier)
        plt.imshow(np.abs(vel_est[0]))
        plt.imshow(np.abs(vel_est[1]))
        plt.xlim((min(samples),max(samples)+1))
        plt.yticks(range(velocity_table.size), velocity_table)
        plt.gca().set_aspect('equal')

    def observation(self, k_obs, theta, alpha=None, add_noise=False,
                    plot_tx=False, plot_rx=False, plot_tau=False, plot_mixed=False,
                    plot_range_fft=False):
        """
            Create a time vector for a specific observation, generate the Tx
            signal and make the observation.

            Args:
                k_obs (int): Observation to calculate the signals for.
                theta (np.ndarray): Target position and velocity.
                plot_tx (bool): Plot the transmitted signals.
                plot_rx (bool): Plot the received signals.
                plot_tau (bool): Plot the calculated delays over time.

            Returns:
                rx_sig (list): List of tdm rx signal
        """
        if alpha is None:
            alpha = self.get_attenuation(theta)

        # Find the time delay between the tx -> target -> rx
        tau_vec = self.time_delay(theta, self.t_vec)

        # Find the originally transmitted signal (starting at t = 0)
        tx_sig = self.transmitter.tx_tdm(self.t_vec)

        # Create the received signal
        s_sig, rx_sig = self.receiver.rx_tdm(tau_vec,
                                             tx_sig,
                                             self.transmitter.f_carrier,
                                             alpha,
                                             self.t_vec,
                                             self.transmitter.t_chirp)

        if add_noise:
            rx_sig, self.receiver.sigma_noise = self.add_awgn(rx_sig, alpha)

        # Mix signals
        mix_vec = self.signal_mixer(rx_sig, tx_sig)
        # Range-FFT
        range_cube, range_est = self.range_fft_cube(mix_vec)
        # range_true, vel_true = self.get_true_dist(theta)
        # self.velocity_fft_cube(range_cube)
        # self.print_estimates(range_true, range_est, vel_true, np.zeros(self.n_channels))

        # Plotters
        if plot_tx:
            self.plot_sig(tx_sig, f"TX signals for observation {k_obs}")
        if plot_rx:
            self.plot_sig(rx_sig, f"RX signals for observation {k_obs}")
        if plot_tau:
            self.plot_tau(tau_vec)
        if plot_mixed:
            self.plot_sig(mix_vec, f"LPF Mixed signals for observation {k_obs}")
        if plot_range_fft:
            self.plot_range_fft(mix_vec, f"FFT for LPF mixed signals for observation {k_obs}")

        return range_est, alpha

    def print_estimates(self, range_true, range_est, vel_true, vel_est):
        print('v_max =', self.wavelength / (4*self.transmitter.t_chirp))
        print('v_res =', self.wavelength / (4*(self.t_obs - self.transmitter.t_chirp)))
        print(f'\nTarget velocity = {vel_true}')
        for n_ch in range(self.n_channels):
            print(f'Receiver {n_ch}:\n Range true: {range_true[n_ch][0]}\n Range est: {range_est[n_ch]}\n Range error: {range_true[n_ch][0]-range_est[n_ch]}\n Velocity est: {vel_est[n_ch]}\n Velocity error: {vel_true-vel_est[n_ch]}\n')

    def plot_sig(self, sig, title):
        """
            Plot the transmitted signals over time. This plotter will create a
            subplot for each signal, be aware that it does not plot more than
            {maxPlots} subplots in total.

            Args:
                sig (np.ndarray): Collection of signal to plot
                title (str): Title for the plot

            Returns:
                no value
        """
        max_plots = 5
        fig, axs = plt.subplots(nrows=min(self.m_channels, max_plots),
                                ncols=1, figsize=(8, 5), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        for idx, m_ch in enumerate(sig):
            axs[idx].plot(self.t_vec / 1e-6, m_ch.real)
            axs[idx].set_title(f"Channel: {idx}")
            if idx == max_plots-1:
                break
        plt.xlabel("Time [µs]")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_tau(self, tau_vec):
        """
            Plot the calculated delays over time

            Args:
                tau_vec (np.ndarray): Collection of time delays over time

            Returns:
                no value
        """
        for idx, tau in enumerate(tau_vec):
            plt.plot(self.t_vec / 1e-6, self.light_speed * tau / 2, label=f'\u03C4$_{idx}$')
            print(f"Change in \u03C4: {idx}", tau[-1] - tau[0])
        plt.legend()
        plt.xlabel("Time [µs]")
        plt.ylabel("Range [m]")
        plt.title("\u03C4 converted into range over time")
        plt.show()

    def plot_range_fft(self, sig_vec, title=None):
        """
            Find and plot FFT's for inputtet signals. This plotter will create
            a subplot for each signal, be aware that it does not plot more than
            {maxPlots} subplots in total.

            Args:
                tau_vec (np.ndarray): Collection of time delays over time
                title (str): Title for the plot

            Returns:
                no value
        """
        max_plots = 3
        fig, axs = plt.subplots(nrows=min(self.m_channels, max_plots),
                                ncols=1, figsize=(8, 5), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        for idx, sig in enumerate(sig_vec):
            fft_sig = np.fft.fft(sig)
            N = len(fft_sig)
            T = N/(self.receiver.f_sample * self.oversample)
            n = np.arange(N)
            freq = n/T
            fft_range = (freq * self.light_speed /
                        (2 * self.transmitter.bandwidth/self.transmitter.t_chirp))
            axs[idx].plot(fft_range, N * np.abs(fft_sig))
            axs[idx].set_title(f"Channel: {idx}")

            sample = np.argmax(len(fft_sig) * np.abs(fft_sig))
            # print(fft_range[sample])

            if idx == max_plots-1:
                break
        plt.xlabel("Range [m]")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    k = 100
    tx = Transmitter(channels=2, chirps=10)
    rx = Receiver(channels=2)

    radar = Radar(tx, rx, "tdm", 2000)
    target = Target(radar.t_obs + radar.k_space, velocity=-12)
    target_states = target.generate_states(k, 'linearaway')
    radar.observation(20, target_states[20],
                      add_noise=True, plot_tx=False, plot_rx=False,
                      plot_mixed=False, plot_range_fft=False)
    # radar.plot_region(target_states)