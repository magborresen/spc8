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

    def __init__(self, k_tot=10, region=[2000, 2000],
                 time_step=0.4, m_transmitters=10,
                 n_receivers=10):
        self.k_tot = k_tot
        self.region = region
        self.time_step = time_step
        self.num_samples = int(self.k_tot / self.time_step)
        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self._fc = 4e6
        self._fs = 2 * self._fc
        self._samples_per_obs = int(self._fs * self.time_step)
        self._t_obs = 2000 * self.time_step
        Observation.__init__(self, self.m_transmitters,
                             self.n_receivers, self.region,
                             self._samples_per_obs)

        System.__init__(self, self.time_step, self.region)
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
        plt.title('Observation region with antennas and trajectory')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.legend()
        plt.show()

    def observe_no_gain(self):
        """
            Creates observations for all receivers for all states at all times

            Args:
                no value

            Returns:
                s_k (np.ndarray): Nested lists of receiver amplitudes for each state
        """
        time = np.linspace(0, self.k_tot, self.num_samples)
        s_k = [self.observation_no_gain(self.states[i], time[i]) for i in range(len(self.states))]

        return np.array(s_k)

    def observe(self, k_obs):
        """
            Creates observations for all receivers for all states at all times

            Args:
                no value

            Returns:
                r_k (np.ndarray): Nested lists of receiver amplitudes for each state
        """
        time = np.linspace(k_obs*self._t_obs, self.time_step, self._samples_per_obs)
        r_k = self.observation(self.states[0], time, k_obs*self._t_obs)

        return np.array(r_k)


if __name__ == '__main__':
    sig = Signal(1, [2000, 2000], 4e-4, 10, 10)
    obs = sig.observe(1)
    print(np.transpose(obs).shape)
