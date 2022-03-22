"""
    This module creates an observation model for the general signal model to use.
"""
import numpy as np
import matplotlib.pyplot as plt

class Observation():
    """
        Class to represent an oberservation    
    """

    def __init__(self, m_transmitters: int, n_receivers: int,
                 region: list, samples_per_obs: float):
        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self.region = region
        self.tx_pos = None
        self.rx_pos = None
        self._array_radius = np.sqrt(2) * self.region[0]
        self.place_antennas()
        self._c = 300000
        self._fc = 4e6
        self._samples_per_obs = samples_per_obs

    def place_antennas(self) -> None:
        """ Place collacted RX and TX antennas in 2D space

            Args:
                no value

            Returns:
                no value
        """
        self.tx_pos = [self._array_radius*np.cos(np.linspace(0, 2*np.pi, self.m_transmitters)),
                       self._array_radius*np.sin(np.linspace(0, 2*np.pi, self.m_transmitters))]

        self.tx_pos = np.array(self.tx_pos)

        self.rx_pos = [self._array_radius*np.cos(np.linspace(0, 2*np.pi, self.n_receivers)),
                       self._array_radius*np.sin(np.linspace(0, 2*np.pi, self.n_receivers))]

        self.rx_pos = np.array(self.rx_pos)

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

    def time_delay(self, rx_n: int, tx_m: int,
                   theta: np.ndarray, t_vec: np.ndarray,
                   t_start: float) -> list:
        """ Find time delay from receiver n to the target to all m transmitters

            Args:
                rx_n (int): Receiver to calculate delays for
                x_k (float): x position of the target
                y_k (float): y position of the target

            Returns:
                tau (float): Signal time delay
        """
        traj = self.trajectory(t_vec, t_start, theta)

        tau = 1/ self._c * np.linalg.norm(self.tx_pos[:,tx_m] - traj.T) + np.linalg.norm(self.rx_pos[:,rx_n] - traj.T)

        return tau

    def trajectory(self, t_vec: np.ndarray, t_start: float, theta: np.ndarray) -> np.ndarray:
        """
            Calculate trajectory used for time delay tau
        """

        r_k = theta[0:2] + (t_vec - t_start) * theta[2:]

        return r_k


    def tx_signal(self, t_vec: np.ndarray, tau: np.ndarray) -> float:
        """ Create the tx radar signal

            Args:
                t (float): time
                tau (float): time delay

            Returns:
                sx_m (float): Transmitted signal amplitude at time t-tau
        """

        sx_m = np.cos(2*np.pi*self._fc*(t_vec-tau))

        return sx_m

    def observation_no_gain(self, theta: np.ndarray,
                            t_vec: np.ndarray, t_start: float) -> np.ndarray:
        """ Calculate observed signal without complex gain from target position

            Args:
                target_pos (list): Target x and y position in that order
                t (float): time

            Returns:
                s_k (list): Observed signals from receiver 0 to n

        """
        s_k = []
        for rx_n in range(self.n_receivers):
            sk_n = 0
            for tx_m in range(self.m_transmitters):
                tau = self.time_delay(rx_n, tx_m, theta, t_vec, t_start)
                sx_m = self.tx_signal(t_vec, tau)
                sk_n += sx_m * np.exp(1j*2*np.pi*self._fc*tau)
            s_k.append(sk_n)

        return np.array(s_k)

    def observation(self, theta: np.ndarray,
                    t_vec: np.ndarray, t_start: float, alpha=1) -> np.ndarray:
        """ Calculate observed signal from target position

            Args:
                target_pos (list): Target x and y position in that order
                t (float): time

            Returns:
                r_k (list): Observed signals from receiver 0 to n

        """
        r_k = []
        for rx_n in range(self.n_receivers):
            rk_n = 0
            for tx_m in range(self.m_transmitters):
                tau = self.time_delay(rx_n, tx_m, theta, t_vec, t_start)
                sx_m = self.tx_signal(t_vec, tau)
                rk_n += alpha * sx_m * np.exp(1j*2*np.pi*self._fc*tau)
            r_k.append(rk_n)

        return np.array(r_k)
