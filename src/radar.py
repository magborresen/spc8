"""
    Radar system object consisting of transmitter(s) and receiver(s)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from receiver import Receiver
from transmitter import Transmitter
from target import Target


class Radar:
    """
        Radar system class

        Args:
            no value
        Returns:
            no value
    """

    def __init__(self, transmitter, receiver, mp_method, region, snr=None):
        self.transmitter = transmitter
        self.receiver = receiver
        self.mp_method = mp_method
        self.region = region
        self.snr = snr
        self.n_channels = self.receiver.channels
        self.m_channels = self.transmitter.channels
        self.t_rx = 20e-6
        self.t_obs = (self.transmitter.t_chirp * self.m_channels *
                      self.transmitter.chirps + self.transmitter.t_chirp)
        self.oversample = 1
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
                                              self.m_channels * self.wavelength/2,
                                              self.m_channels)),
                                  np.zeros(self.m_channels)])

        self.rx_pos = np.array([(self.region / 2 + self.rx_tx_spacing / 2 +
                                np.linspace(self.wavelength / 2,
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
                d_tx = np.sqrt((self.tx_pos[0,tx_m] - traj[0].T)**2 + (self.tx_pos[1,tx_m] - traj[1].T)**2)
                d_rx = np.sqrt((self.rx_pos[0,rx_n] - traj[0].T)**2 + (self.rx_pos[1,rx_n] - traj[1].T)**2)
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
        # Normalized unitvector for position (line-of-sight)
        los = theta[:2] / np.linalg.norm(theta[:2])

        # Target trajectory within acquisition period
        r_k = theta[:2] + (t_vec - t_vec[0]) * ((los[0]*theta[2]) + (los[1]*theta[3]))

        return r_k

    def get_attenuation(self, theta, target_rcs=0.04, rho=0.5):
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
        aperture = self.wavelength**2 * rho
        gain = 4 * np.pi * aperture / self.wavelength**2
        alpha = []
        for rx_n in range(self.n_channels):
            # Get range from receiver to target
            target_range = np.sqrt((self.rx_pos[0,rx_n] - theta[0])**2 + (self.rx_pos[1,rx_n] - theta[1])**2)

            # Determine atmospheric loss (Richards, p. 43)
            loss_atmospheric = 10**(self.atmos_loss_factor * target_range / 5000)

            # Calculate attenuation (Richards, p. 92)
            attenuation = (gain**2 * self.wavelength**2 * target_rcs /
                          ((4*np.pi)**3 * target_range**4 * loss_atmospheric))
            alpha.append(attenuation)

        return np.array(alpha)

    def add_awgn(self, signals):
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
            # Calculate linear SNR
            lin_snr = 10.0**(self.snr / 10.0)
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

    def delay_signal(self, sig_vec, tau_vec):
        """
            Delay a list of signals, given a list of time-delays. A entire
            transmitted signal is delayed by the first corresponding tau. It
            is assumed that the target does not move within t_obs.

            Args:
                sig_vec (np.ndarray): Collection of signals
                tau_vec (np.ndarray): Collection of time-delays

            Returns:
                output_vec (np.ndarray): Collection of delayed signals
        """
        output_vec = []
        for idx, sig in enumerate(sig_vec):
            # Get offset in seconds, based on first tau from each transmitter
            offset = tau_vec[idx * self.n_channels][0]

            # Get delay in number of samples
            delay = round(offset / self.t_vec[1])

            # Delay signal by desired number of samples (pad with 0)
            output = np.r_[np.full(delay, 0), sig[:-delay]]

            output_vec.append(output)

        return np.array(output_vec)


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

    def observation(self, k_obs, theta, alpha=None, add_noise=False, plot_tx=False, plot_rx=False, plot_tau=False, plot_mixed=False, plot_fft=False):
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

        # Delay the originally transmitted signal (starting at tau)
        tx_sig_offset = self.delay_signal(tx_sig, tau_vec)

        # Create the received signal
        s_sig, rx_sig = self.receiver.rx_tdm(tau_vec,
                                             tx_sig_offset,
                                             self.transmitter.f_carrier,
                                             alpha)

        if add_noise:
            rx_sig, self.receiver.sigma_noise = self.add_awgn(rx_sig)

        # Mix signals
        mixed_sig = np.conjugate(rx_sig) * sum(tx_sig) # With attenuation
        mixed_s_sig = np.conjugate(s_sig) * sum(tx_sig) # Without attenuation

        # Low-pass filter signals
        lpf_mixed_sig = self.butter_lowpass_filter(mixed_sig, self.receiver.f_sample/2-1) # With attenuation
        lpf_mixed_s_sig = self.butter_lowpass_filter(mixed_s_sig, self.receiver.f_sample/2-1) # Without attenuation

        # Plotters
        if plot_tx:
            self.plot_sig(tx_sig, f"TX signals for observation {k_obs}")
        if plot_rx:
            self.plot_sig(rx_sig, f"RX signals for observation {k_obs}")
        if plot_tau:
            self.plot_tau(tau_vec)
        if plot_mixed:
            self.plot_sig(lpf_mixed_sig, f"LPF Mixed signals for observation {k_obs}")
        if plot_fft:
            self.plot_fft(lpf_mixed_sig, f"FFT for LPF mixed signals for observation {k_obs}")

        return (lpf_mixed_s_sig, lpf_mixed_sig, alpha)

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
        maxPlots = 5
        fig, axs = plt.subplots(nrows=min(self.m_channels, maxPlots),
                                ncols=1, figsize=(8, 5), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        for idx, m_ch in enumerate(sig):
            axs[idx].plot(self.t_vec / 1e-6, m_ch.real)
            axs[idx].set_title(f"Channel: {idx}")
            if idx == maxPlots-1: 
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

    def plot_fft(self, sig_vec, title=None):
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
        maxPlots = 3
        fig, axs = plt.subplots(nrows=min(self.m_channels, maxPlots),
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
            axs[idx].plot(fft_range, 2.0/N * np.abs(fft_sig))
            axs[idx].set_title(f"Channel: {idx}")
            if idx == maxPlots-1: 
                break
        plt.xlabel("Range [m]")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    k = 10
    tx = Transmitter(channels=2, t_chirp=60e-6, chirps=2)
    rx = Receiver(channels=2)

    radar = Radar(tx, rx, "tdm", 2000, snr=1)
    target = Target(radar.t_obs + radar.k_space)
    target_states = target.generate_states(k, 'linear_away')
    radar.observation(1, target_states[1], 
                      add_noise=True, plot_tx=False, plot_rx=True,
                      plot_mixed=False, plot_fft=True)
    
    # Check distance:
    # print([np.sqrt((radar.rx_pos[0,rx_n] - target_states[0][0])**2 + (radar.rx_pos[1,rx_n] - target_states[0][1])**2) for rx_n in range(radar.n_channels)])

    # radar.plot_antennas()
    # radar.plot_region(target_states)