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

    def __init__(self, transmitter, receiver, observations, mp_method, region):
        self.transmitter = transmitter
        self.receiver = receiver
        self.observations = observations
        self.mp_method = mp_method
        self.region = region
        self.n_channels = self.receiver.channels
        self.m_channels = self.transmitter.channels
        self.t_rx = 14e-6
        self.t_obs = (self.t_rx * self.receiver.channels +
                      self.transmitter.t_chirp * self.transmitter.channels)
        self.t_tot = self.observations * self.t_obs
        self.oversample = 10
        self.samples_per_obs = int(self.receiver.f_sample * self.t_obs * self.oversample)
        self.light_speed = 300e6
        self.wavelength = self.light_speed / self.receiver.f_sample
        self.tx_pos = None
        self.rx_pos = None
        self.place_antennas()
        self.max_range = None
        self.min_range = self.transmitter.t_chirp * self.light_speed / 2
        self.range_res = self.light_speed / (2 * self.transmitter.bandwidth)

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
            Find time delay from receiver n to the target to the m'th transmitters.
            It is estimated at all antennas located in the same position in origo of
            the coordinate system.

            Args:
                theta (np.ndarray): x and y position of the target
                t_vec  (np.ndarray): Times for which to calculate the delays

            Returns:
                tau (float): Signal time delay
        """

        traj = self.trajectory(t_vec, theta)

        tau = 2 / self.light_speed * traj

        return tau

    def trajectory(self, t_vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
            Calculate trajectory used for time delay tau

            Args:
                t_vec (np.ndarray): time vector
                theta (np.ndarray): Target position

            Returns:
                r_k (np.ndarray): Short time trajectory model based on original position and velocity.
        """

        r_k = theta[0:2] + (t_vec - t_vec[0]) * theta[2:]

        return r_k

    def plot_region(self, states):
        """
            Plots the observation region with antenna locations and trajectory

            Args:
                no value

            Returns:
                no value
        """
        fig, ax = plt.subplots()
        ax.scatter(states[:,0], states[:,1], label="Trajectory")
        ax.scatter(self.tx_pos[0], self.tx_pos[1], label="TX Antennas")
        ax.scatter(self.rx_pos[0], self.rx_pos[1], label="RX Antennas")
        ax.set_aspect(1)
        ax.set_xlim(0, self.region)
        ax.set_ylim(0, self.region)
        plt.title('Observation region with antennas and trajectory')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.legend()
        plt.show()

    def observe(self, k_obs, state):
        """
            Create a time vector for a specific observation, generate the Tx
            signal and make the observation.

            Args:
                k_obs (int): Observation to start from

            Returns:
                t_vec (np.ndarray): time vector
        """
        t_vec = np.linspace(k_obs * self.t_tot, (k_obs + 1) * self.t_tot, self.samples_per_obs * self.oversample)
        
        
        
        return t_vec

if __name__ == '__main__':
    k_obs = 1000
    tx = Transmitter()
    rx = Receiver()

    radar = Radar(tx, rx, k_obs, "tdm", 2000)
        
    target = Target(radar.t_obs)
    states = target.generate_states(k_obs)
    radar.plot_region(states)

    t = radar.observe(1, states[0])


    # radar = Radar(tx, rx, 5, "tdm", 2000)
    # sig, freq = radar.transmitter.tx_tdm(radar.t_vec[0:radar.samples_per_obs], radar.t_rx, radar.receiver.f_sample*radar.oversample, plot=True)

