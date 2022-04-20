"""
    This module creates a signal model for the estimator to use.
"""
import matplotlib.pyplot as plt
import numpy as np
from observation_model import Observation
from system_model import System


class Signal(Observation, System):
    """
        Class to create a signal model from Observation and System classes
    """

    def __init__(self, k_tot=10, region=2000, m_transmitters=10, n_receivers=10, t_tx_tot=1.33e-7, t_rx=1.8734e-5):
        self.k_tot = k_tot
        self.region = region
        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self.t_tx_tot = t_tx_tot
        self.t_tx = self.t_tx_tot / self.m_transmitters
        self.t_rx = t_rx
        self._fc = 1e9
        self._fs = 20 * self._fc
        self._c = 300e6
        self._t_obs = self.t_tx_tot + self.t_rx
        self._samples_per_obs = int(self._fs * self.t_rx)

        Observation.__init__(self, self.m_transmitters,
                             self.n_receivers, self.region,
                             self._samples_per_obs, self.t_tx, self._t_obs, self._fc)

        System.__init__(self, self._t_obs)
        self.states = System.generate_states(self, self.k_tot)

    def plot_region(self):
        """
            Plots the observation region with antenna locations and trajectory

            Args:
                no value

            Returns:
                no value
        """
        fig, ax = plt.subplots()
        ax.scatter(self.states[:,0], self.states[:,1], label="Trajectory")
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

    def add_noise(self, obs, SNR_dB : float):
        SNR = 10.0**((SNR_dB)/10.0)
        rec = len(obs)
        obs_n = []
        for r in range(rec):
            s_power = np.var(obs[r])
            n_power = s_power/SNR
            noise = np.random.multivariate_normal((0,0), (n_power/2)*np.eye(2), len(obs[r]))
            noise = noise.view(np.complex128)
            obs_n.append([sum(x)[0] for x in zip(obs[r], noise)])
        return np.array(obs_n)

    def observe_y(self, k_obs: int, d_dB : float) -> np.ndarray:
        """
            Creates observations for all receivers for all states at all times

            Args:
                k_obs (int): Current observation
                theta (np.ndarray): Target position and velocity: x, y, vx, vy
                d_dB (float): Desired SNR

            Returns:
                y_k (np.ndarray): Nested lists of receiver amplitudes for each state
        """
        y_k = self.observe_x(k_obs)
        #y_k = self.add_noise(np.array(y_k), d_dB)

        return y_k

    def observe_x(self, k_obs: int) -> np.ndarray:
        """
            Creates observations for all receivers for the k'th observation at all times in k

            Args:
                k_obs (int): Current observation
                theta (np.ndarray): Target position and velocity: x, y, vx, y

            Returns:
                x_k (np.ndarray): Nested lists of receiver amplitudes for each state
        """

        time = np.linspace((k_obs-1)*self._t_obs + self.t_tx_tot, k_obs * self._t_obs, self._samples_per_obs)
        x_k = self.observation(self.states[k_obs], time)

        return np.array(x_k)


if __name__ == '__main__':
    sig = Signal(10, [2000, 2000], 2, 2)
    sig.plot_region()