"""
    This module creates an observation model for the general signal model to use.
"""
import numpy as np
import matplotlib.pyplot as plt

class Observation():
    """
        Class to represent an oberservation    
    """

    def __init__(self, m_transmitters: int, n_receivers: int, region: list):
        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self.region = region
        self.tx_pos = None
        self.rx_pos = None
        self._array_radius = np.sqrt(2) * self.region[0]
        self.place_antennas()
        self._c = 300000
        self.alpha = 1
        self._fc = 40e6

    def place_antennas(self) -> None:
        """ Place collacted RX and TX antennas in 2D space

            Args:
                no value

            Returns:
                no value
        """
        self.tx_pos = [self._array_radius*np.cos(np.linspace(0, 2*np.pi, self.m_transmitters)),
                       self._array_radius*np.sin(np.linspace(0, 2*np.pi, self.m_transmitters))]

        self.rx_pos = [self._array_radius*np.cos(np.linspace(0, 2*np.pi, self.n_receivers)),
                       self._array_radius*np.sin(np.linspace(0, 2*np.pi, self.n_receivers))]

    def plot_antennas(self) -> None:
        """ Plot antenna postions in 2D space

            Args:
                no value

            Returns:
                no value
        """
        plt.scatter(self.tx_pos[0], self.tx_pos[1], label="TX Antennas")
        plt.scatter(self.rx_pos[0], self.rx_pos[1], label="RX Antennas")
        plt.legend()
        plt.title("RX/TX positions in the plane")
        plt.xlabel("Position [m]")
        plt.ylabel("Position [m]")
        plt.show()

    def time_delay(self, rx_n: int, x_k: float, y_k: float) -> list:
        """ Find time delay from receiver n to the target to all m transmitters

            Args:
                rx_n (int): Receiver to calculate delays for
                x_k (float): x position of the target
                y_k (float): y position of the target

            Returns:
                tau (float): Signal time delay
        """
        tau = 1 / self._c * (np.sqrt((self.tx_pos[0] - x_k)**2 +
                            (self.tx_pos[1] - y_k)**2) +
                             np.sqrt((self.rx_pos[0][rx_n] - x_k)**2 +
                            (self.rx_pos[1][rx_n]- y_k)**2))

        return tau

    def tx_signal(self, t, tau) -> float:
        """ Create the tx radar signal

            Args:
                t (float): time
                tau (float): time delay

            Returns:
                sx_m (float): Transmitted signal amplitude at time t-tau
        """

        sx_m = np.cos(2*np.pi*self._fc*(t-tau))

        return sx_m

    def observation(self, target_pos: list, t: float) -> list:
        """ Calculate observed signal from target position

            Args:
                target_pos (list): Target x and y position in that order
                t (float): time

            Returns:
                rk (list): Observed signals from receiver 0 to n

        """
        r_k = []
        for rx_n in range(self.n_receivers):
            tau = self.time_delay(rx_n, target_pos[0], target_pos[1])
            sx_m = self.tx_signal(t, tau)
            rk_n = sum([self.alpha * sx_m * np.exp(1j*2*np.pi*self._fc*tau)])
            r_k.append(rk_n)

        return r_k
