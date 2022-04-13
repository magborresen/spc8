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
            Find time delay from receiver n, to the target, to the m'th transmitters.
            It is assumed that all antennas are located in the origin of the
            coordinate system.

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
            
            Angle A is betweeen the line-of-sight from origin to target, and
            the velocity vector of the drone. los is a unit vector
            representation of the line-of-sight.

            Args:
                t_vec (np.ndarray): time vector
                theta (np.ndarray): Target position

            Returns:
                r_k (np.ndarray): Short time trajectory model based on original position and velocity.
        """

        los = theta[:2] / np.linalg.norm(theta[:2])         
        cos_A = (los[0]*theta[2]) + (los[1]*theta[3]) / np.linalg.norm(theta[2:])
        
        r_k = np.linalg.norm(theta[:2]) + (t_vec - t_vec[0]) * np.linalg.norm(theta[2:]) * cos_A
 
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
        if zoom==True:
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

    def observe(self, k_obs, theta):
        """
            Create a time vector for a specific observation, generate the Tx
            signal and make the observation.

            Args:
                k_obs (int): Observation to start from

            Returns:
                t_vec (np.ndarray): time vector
        """
        print('\nt_obs:', self.t_obs)
        
        t_vec = np.linspace(k_obs * self.t_obs, (k_obs + 1) * self.t_obs, self.samples_per_obs)
        t_tau = np.linspace(k_obs * self.t_obs, (k_obs + 1) * self.t_obs, self.n_channels)
        
        tau_kn = self.time_delay(theta, t_tau)
        
        print('\nObservation', k_obs, 'starting time:', t_vec[0])
        print('Actual length:', np.linalg.norm(theta[:2]))
        print('Distance (tau):', tau_kn * self.light_speed / 2)

        return None

if __name__ == '__main__':
    k_obs = 10
    tx = Transmitter()
    rx = Receiver()

    radar = Radar(tx, rx, k_obs, "tdm", 2000)
        
    target = Target(radar.t_obs)
    states = target.generate_states(k_obs, 'linear_away')
    radar.plot_region(states, True)

    radar.observe(0, states[0])
    radar.observe(1, states[1])
    radar.observe(k_obs-2, states[k_obs-2])
    radar.observe(k_obs-1, states[k_obs-1])

    # radar = Radar(tx, rx, 5, "tdm", 2000)
    # sig, freq = radar.transmitter.tx_tdm(radar.t_vec[0:radar.samples_per_obs], radar.t_rx, radar.receiver.f_sample*radar.oversample, plot=True)