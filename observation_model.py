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
                 region: list, samples_per_obs: float, t_tx: float):

        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self.region = region
        self.tx_pos = None
        self.rx_pos = None
        self._array_radius = np.sqrt(2) * self.region[0]
        self.place_antennas()
        self._c = 300e6
        self._fc = 30e9
        self.gain = 40
        self._samples_per_obs = samples_per_obs
        self.t_tx = t_tx

    def place_antennas(self) -> None:
        """ Place collacted RX and TX antennas in 2D space

            Args:
                no value

            Returns:
                no value
        """

        lambda_f = self._c / self._fc

        self.tx_pos = np.array([np.linspace(lambda_f / 2,
                                            self.m_transmitters * lambda_f/2,
                                            self.m_transmitters),
                                np.linspace((self.n_receivers + self.m_transmitters-1) * lambda_f/2,
                                            self.n_receivers * lambda_f/2,
                                            self.m_transmitters)])

        self.rx_pos = np.array([np.linspace(self.tx_pos[0][-1] + lambda_f/2,
                                           (self.n_receivers + self.m_transmitters) * lambda_f/2,
                                            self.n_receivers),
                                np.linspace(self.tx_pos[1][-1] - lambda_f/2,
                                            0,
                                            self.n_receivers)])

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
                   theta: np.ndarray, t_vec: np.ndarray) -> list:
        """ Find time delay from receiver n to the target to the m'th transmitters

            Args:
                rx_n (int): Receiver to calculate delays for
                x_k (float): x position of the target
                y_k (float): y position of the target

            Returns:
                tau (float): Signal time delay
        """
        traj = self.trajectory(t_vec, theta)
        
        d_tx = np.sqrt((self.tx_pos[0,tx_m] - traj[0].T)**2 + (self.tx_pos[1,tx_m] - traj[1].T)**2)
        d_rx = np.sqrt((self.rx_pos[0,rx_n] - traj[0].T)**2 + (self.rx_pos[1,rx_n] - traj[1].T)**2)
        
        tau = 1 / self._c * (d_tx + d_rx)
        
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


    def tx_signal(self, tx_m: int, t_vec: np.ndarray, tau: np.ndarray) -> float:
        """ Create the tx radar signal

            Args:
                t (float): time
                tau (float): time delay

            Returns:
                sx_m (float): Transmitted signal amplitude at time t-tau
        """
        #w = t_vec[(np.where((tx_m * self.t_tx <= (t_vec - tau)) & ((t_vec - tau) < (tx_m+1) * self.t_tx), 1, 0))]
        
        w = np.zeros(self._samples_per_obs)
        
        i = np.full_like(w, True, dtype=bool)
        
        t_delay = t_vec - tau
        cond1 = self.t_tx * tx_m
        cond2 = self.t_tx * (tx_m + 1)
        
        i[t_delay <= cond1] = False
        i[t_delay > cond2] = False
        
        print(i)
        
        w[i] = 1
        
        #print(w)
        # start = int(tx_m * self._samples_per_obs / self.m_transmitters)
        # stop = int((1 + tx_m) * self._samples_per_obs / self.m_transmitters)

        # w[start:stop] = 1

        a_km = self.gain * w
        sx_m = a_km * np.exp(2j * np.pi * self._fc * (t_vec-tau))

        return sx_m

    def observation_no_alpha(self, theta: np.ndarray,
                            t_vec: np.ndarray) -> np.ndarray:
        """ Calculate observed signal without complex attenuation from target position

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
                tau = self.time_delay(rx_n, tx_m, theta, t_vec)
                sx_m = self.tx_signal(tx_m, t_vec, tau)
                sk_n += sx_m * np.exp(1j*2*np.pi*self._fc*tau)
            s_k.append(sk_n)

        return np.array(s_k)

    def observation(self, theta: np.ndarray,
                    t_vec: np.ndarray, alpha=1) -> np.ndarray:
        """ Calculate observed signal from target position

            Args:
                theta (np.ndarray): Target position and velocity: x, y, vx, vy
                t_vec (np.ndarray): Observation times
                t_start

            Returns:
                r_k (list): Observed signals from receiver 0 to n

        """
        r_k = []
        for rx_n in range(self.n_receivers):
            rk_n = 0
            for tx_m in range(self.m_transmitters):
                tau = self.time_delay(rx_n, tx_m, theta, t_vec)
                sx_m = self.tx_signal(tx_m, t_vec, tau)
                rk_n += alpha * sx_m * np.exp(2j*np.pi*self._fc*tau)
            r_k.append(rk_n)
        plt.plot(sx_m)
        plt.plot(rk_n[0])
        plt.show()
        return np.array(r_k)
