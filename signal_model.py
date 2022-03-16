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

    def __init__(self, k_tot: int, region: list,
                 time_step: float, m_transmitters: int,
                 n_receivers: int):
        self.k_tot = k_tot
        self.region = region
        self.time_step = time_step
        self.num_samples = int(self.k_tot / self.time_step)
        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        Observation.__init__(self, self.m_transmitters, self.n_receivers, self.region)
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

    def observe(self):
        """
            Creates observations for all receivers for all states at all times

            Args:
                no value

            Returns:
                r_k (list): Nested lists of receiver amplitudes for each state
        """
        time = np.linspace(0, self.k_tot, self.num_samples)
        r_k = [self.observation(self.states[i], time[i]) for i in range(len(self.states))]

        return r_k[0]


if __name__ == '__main__':
    sig = Signal(1, [2000, 2000], 4e-4, 10, 10)
    obs = sig.observe()
    print(obs)
