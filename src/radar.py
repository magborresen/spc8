"""
    Radar system object consisting of transmitter(s) and receiver(s)
"""
import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, transmitter, receiver, mp_method, region):
        self.transmitter = transmitter
        self.receiver = receiver
        self.mp_method = mp_method
        self.region = region
        self.n_channels = self.receiver.channels
        self.m_channels = self.transmitter.channels
        self.t_rx = 20e-6
        self.t_obs = self.t_rx*self.n_channels + self.transmitter.t_chirp*self.m_channels
        self.oversample = 1
        self.samples_per_obs = int(self.receiver.f_sample * self.t_obs * self.oversample)
        self.light_speed = 300e6
        self.wavelength = self.light_speed / self.receiver.f_sample
        self.tx_pos = None
        self.rx_pos = None
        self.place_antennas()
        self.max_range = None
        self.min_range = self.transmitter.t_chirp * self.light_speed / 2
        self.range_res = self.light_speed / (2 * self.transmitter.bandwidth)
        self.k_space = 1
        self.t_vec = self.create_time_vector()

    def place_antennas(self):
        """
            Place antennas in 2D region.
        """
        self.tx_pos = np.array([np.linspace(self.wavelength / 2,
                                            self.m_channels * self.wavelength/2,
                                            self.m_channels),
                                np.linspace((self.n_channels + self.m_channels-1) * self.wavelength/2,
                                            self.n_channels * self.wavelength/2,
                                            self.m_channels)])

        self.rx_pos = np.array([np.linspace(self.tx_pos[0][-1] + self.wavelength/2,
                                           (self.n_channels + self.m_channels) * self.wavelength/2,
                                            self.n_channels),
                                np.linspace(self.tx_pos[1][-1] - self.wavelength/2,
                                            0,
                                            self.n_channels)])

    def time_delay(self, theta: np.ndarray, t_vec: np.ndarray) -> list:
        """
            Find time delay from receiver n, to the target, to the m'th transmitters.
            It is assumed that all antennas are located in the origin of the
            coordinate system.

            Args:
                theta (np.ndarray): x and y position of the target
                t_vec  (np.ndarray): Times for which to calculate the delays

            Returns:
                tau (np.ndarray): Signals time delay
        """

        traj = self.trajectory(t_vec, theta)

        tau = []
        for tx_m in range(self.m_channels):
            for rx_n in range(self.n_channels):
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

    def plot_region(self, states, zoom=False):
        """
            Plots the observation region with antenna locations and trajectory

            Args:
                states (np.ndarray): Collection of all states
                zoom (bool): Show only trajectory region if true

            Returns:
                no value
        """
        fig, ax = plt.subplots()
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
            Delay a list of signals, given a list of time-delays

            Args:
                sig_vec (np.ndarray): Collection of signals
                tau_vec (np.ndarray): Collection of time-delays

            Returns:
                output_vec (np.ndarray): Collection of delayed signals
        """
        output_vec = []
        for idx, sig in enumerate(sig_vec):
            # Get offset in seconds, based on first tau in each transmission
            offset = tau_vec[idx * self.n_channels][0]
            
            # Get delay in number of samples
            delay = int(round(offset / self.t_vec[1]))
            
            # Delay signal by desired number of samples
            if delay >= 0:
                output = np.r_[np.full(delay, 0), sig[:-delay]]
            else:
                output = np.r_[sig[-delay:], np.full(-delay, 0)]
                
            output_vec.append(output)
        
        return np.array(output_vec)
    

    def observation(self, k_obs, theta, add_noise=True, plot_tx=False, plot_rx=False, plot_rx_tx=False, plot_tau=False):
        """
            Create a time vector for a specific observation, generate the Tx
            signal and make the observation.

            Args:
                k_obs (int): Observation to calculate the signals for.
                theta (np.ndarray): Target position and velocity.
                plot_tx (bool): Plot the transmitted signals.
                plot_rx (bool): Plot the received signals.
                plot_rx_tx (bool): Plot the received and transmitted
                                   signals in the same figure.
                plot_tau (bool): Plot the calculated delay over time.

            Returns:
                rx_sig (list): List of tdm rx signal
        """
        # Find the time delay between the tx -> target -> rx
        tau_vec = self.time_delay(theta, self.t_vec)

        # Find the originally transmitted signal (starting at t = 0)
        tx_sig_nd = self.transmitter.tx_tdm(self.t_vec)
        
        # Delay the originally transmitted signal (starting at tau_m[0])
        tx_sig = self.delay_signal(tx_sig_nd, tau_vec)
        
        # Create the received signal
        s_sig, rx_sig = self.receiver.rx_tdm(tau_vec, tx_sig, self.transmitter.f_carrier, add_noise=add_noise)

        if plot_tx:
            self.plot_sig(self.t_vec, tx_sig_nd, f"TX signals for observation {k_obs}")

        if plot_rx:
            self.plot_sig(self.t_vec, rx_sig, f"RX signals for observation {k_obs}")

        if plot_rx_tx:
            self.plot_sigs(self.t_vec, tx_sig_nd,
                            self.t_vec, rx_sig,
                            f"TX/RX signals for observation {k_obs}")

        # if plot_tau:
        #     self.plot_tau(self.t_vec, tau)

        return (s_sig, rx_sig)

    def plot_sig(self, t_vec, sig, title):
        """
            Plot the transmitted signals over time

            Args:
                t_vec (np.ndarray): Time based array for x-axis
                sign (np.ndarray): Signal to plot
                title (str): Title for the plot

            Returns:
                no value
        """

        fig, axs = plt.subplots(nrows=self.m_channels, ncols=1, figsize=(8, 5), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        axs.ravel()

        for idx, m_ch in enumerate(sig):
            axs[idx].plot(t_vec / 1e-6, m_ch.real)
            axs[idx].set_title(f"Channel: {idx}")

        plt.xlabel("Time [µs]")
        fig.suptitle(title)
        plt.show()

    def plot_sigs(self, t_vec1, sig1, t_vec2, sig2, title):
        """
            Plot multiple signals over time

            Args:
                t_vec1 (np.ndarray): First time vector
                sig1 (np.ndarray): First signal to plot
                t_vec2 (np.ndarray): Second time vector
                sig2 (np.ndarray): Second signal to plot
                title (str): Title of the figure

            Returns:
                no value
        """

        fig, axs = plt.subplots(nrows=self.m_channels, ncols=1, figsize=(8, 5), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        axs.ravel()

        for idx, m_ch in enumerate(sig1):
            axs[idx].plot(t_vec1 / 1e-6, np.sum(sig1, axis=0).real)
            axs[idx].set_title(f"Channel: {idx}")

        for idx, m_ch in enumerate(sig2):
            axs[idx].plot(t_vec2 / 1e-6, m_ch.real)

        plt.xlabel("Time [µs]")
        fig.suptitle(title)
        plt.show()


    def plot_tau(self, t_vec, tau):
        """
            Plot the calculated delays over time

            Args:
                t_vec (np.ndarray): Time vector over which to plot tau
                tau (np.ndarray): The delay variables over time

            Returns:
                no value
        """

        plt.plot(t_vec / 1e-6, tau)
        plt.xlabel("Time [µs]")
        plt.ylabel("$\tau$")
        plt.title("$\tau$ over time")
        plt.show()

if __name__ == '__main__':
    k = 10
    tx = Transmitter(channels=3)
    rx = Receiver(channels=3)

    radar = Radar(tx, rx, "tdm", 2000)

    target = Target(radar.t_obs + radar.k_space)
    target_states = target.generate_states(k, 'linear_way')
    #radar.plot_region(target_states, True)
    radar.observation(1, target_states[1], plot_rx_tx=False, plot_tx=True, plot_rx=True)
    # s, rx = radar.observation(1, target_states[1], plot_rx_tx=False)
