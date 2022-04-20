"""
    Radar system object consisting of transmitter(s) and receiver(s)
"""
import numpy as np
import matplotlib.pyplot as plt
from src.receiver import Receiver
from src.transmitter import Transmitter
from src.target import Target

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
        self.t_rx = 14e-6
        self.t_obs = self.t_rx + self.transmitter.t_chirp
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
        self.k_space = 1

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

    def time_delay(self, theta: np.ndarray, t0: float, t_vec: np.ndarray) -> list:
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

        traj = self.trajectory(t0, t_vec, theta)

        tau = 2 / self.light_speed * traj

        return tau

    def trajectory(self, t0: float, t_vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
            Calculate target trajectory within an acquisition period
            
            Angle A is betweeen the line-of-sight from origin to target, and
            the velocity vector of the drone. los is a unit vector
            representation of the line-of-sight.

            Args:
                t_vec (np.ndarray): time vector
                theta (np.ndarray): Target position

            Returns:
                r_k (np.ndarray): Short time trajectory model based on original position and velocity.
        """
        # Normalized unitvector for position (line-of-sight)
        los = theta[:2] / np.linalg.norm(theta[:2])

        # Target trajectory within acquisition period
        r_k = np.linalg.norm(theta[:2]) + (t_vec - t0) * ((los[0]*theta[2]) + (los[1]*theta[3]))

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

    def create_time_tdm(self, k_obs):
        """
            Create the time vector for each receive time in the observation

            Args:
                k_obs (int): The observation number to create the time vector for

            Returns:
                t_vec (list): List of rx times after each tx is done
        """

        #t_vec = [np.random.uniform(
        #                           low=self.transmitter.t_chirp + (m_ch*self.t_obs) + k_obs*self.k_space,
        #                           high=self.t_obs*(m_ch+1) + k_obs*self.k_space)
        #         for m_ch in range(self.m_channels)]

        t_vec = [np.linspace(self.transmitter.t_chirp + (m_ch*self.t_obs) + k_obs*self.k_space, self.t_obs*(m_ch+1) + k_obs*self.k_space, self.samples_per_obs) for m_ch in range(self.m_channels)]

        # Find the start time for this observation as reference
        t0 = k_obs * self.k_space

        return  (t0, np.array(t_vec))


    def observation(self, k_obs, theta):
        """
            Create a time vector for a specific observation, generate the Tx
            signal and make the observation.

            Args:
                k_obs (int): Observation to start from
                theta (np.ndarray): Target position and velocity

            Returns:
                rx_sig (list): List of tdm rx signal
        """

        # Create time vector of scalar rx times
        t0, t_vec = self.create_time_tdm(k_obs)
        print(f"t_vec: {t_vec}")
        # Find the time delay between the tx -> target -> rx
        tau = self.time_delay(theta, t0, t_vec)
        print(f"tau: {tau}")
        # Shift the time vector for the tx signal
        delay = t_vec - tau
        print(f"delay: {delay}")

        # Find the originally transmitted signal
        tx_sig = self.transmitter.tx_tdm(delay, self.t_rx, t0)
        print(f"TX Signals: {tx_sig}")

        # Create the received signal
        rx_sig = self.receiver.rx_tdm(tau, tx_sig, self.transmitter.f_carrier)

        return rx_sig

if __name__ == '__main__':
    k = 10
    tx = Transmitter()
    rx = Receiver()

    radar = Radar(tx, rx, "tdm", 2000)

    target = Target(radar.t_obs + radar.k_space)
    states = target.generate_states(k, 'linear_away')
    #radar.plot_region(states, False)
    rx = radar.observation(0, states[0])
    print(f"RX signals: {rx}")

    # radar = Radar(tx, rx, 5, "tdm", 2000)
    # sig, freq = radar.transmitter.tx_tdm(radar.t_vec[0:radar.samples_per_obs], radar.t_rx, radar.receiver.f_sample*radar.oversample, plot=True)