"""
    This module creates an observation model for the general signal model to use.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

class Observation():
    """
        Class to represent an oberservation   
    """

    def __init__(self, m_transmitters: int, n_receivers: int,
                 region: list, samples_per_obs: float, t_tx: float,
                 t_obs: float, fc: float):

        self.m_transmitters = m_transmitters
        self.n_receivers = n_receivers
        self.region = region
        self.tx_pos = None
        self.rx_pos = None
        self._array_radius = np.sqrt(2) * self.region[0]
        self.place_antennas()
        self._c = 300e6
        self._fc = fc
        self.gain = 40
        self._samples_per_obs = samples_per_obs
        self.t_tx = t_tx
        self.t_obs = t_obs

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
                tx_m (int): Current transmitter index
                t_vec (np.ndarray): Array containing times of the current observation
                tau (np.ndarray): Array containing calculated time delays between tx -> target -> rx

            Returns:
                sx_m (np.ndarray): Transmitted signal amplitudes at times t_vec-tau
        """

        omega = sig.square(2 * np.pi * 1/self.t_obs * ((t_vec - tx_m*self.t_tx) - tau),
                       duty=self.t_tx/self.t_obs)

        omega = np.where(omega < 1, 0, 1)

        chirp_t = np.where(omega==1)
        chirp_start = chirp_t[0][0]
        chirp_stop = chirp_t[0][-1]
        delay = t_vec - tau
        f_t = sig.chirp(delay[chirp_start:chirp_stop], self._fc - 150e6, delay[chirp_stop], self._fc + 150e6)
        omega = omega.astype(np.float64)
        omega[chirp_start:chirp_stop] = f_t
        a_km = self.gain * omega
        sx_m = a_km * np.exp(2j * np.pi * self._fc * (t_vec-tau))

        return a_km, sx_m

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
                alpha (complex): Attenuation of received signal

            Returns:
                r_k (list): Observed signals from receiver 0 to n

        """
        r_k = []
        s_x = []
        tau_mn = []
        for rx_n in range(self.n_receivers):
            rk_n = 0
            for tx_m in range(self.m_transmitters):
                tau = self.time_delay(rx_n, tx_m, theta, t_vec)
                tau_mn.append(tau)
                f_t, sx_m = self.tx_signal(tx_m, t_vec, tau)
                s_x.append(sx_m)
                rk_n += alpha * sx_m * np.exp(2j*np.pi*self._fc*tau)
            r_k.append(rk_n)

        self.smart_plotter((1,1), t_vec, tau_mn, s_x, r_k)

        return np.array(r_k)

    def smart_plotter(self, rx, t_vec, tau, s_x, r_k):
        """ Plot the observed signal (figure 1) and the transmitted signals (figure 2)

            Args:
                rx (list) = (a, b): Plot from receiver a to b
                t_vec (np.ndarray): Observation times
                tau (np.ndarray): Array containing calculated time delays between tx -> target -> rx
                s_x (list): Transmitted signal amplitudes 0 to m*n
                r_k (list): Observed signals from receiver 0 to n
            Returns:
                no value
        """
        for rx_n in range(rx[0]-1, rx[1]):
            fig_rx, axs_rx = plt.subplots(2, sharex=True)
            fig_rx.suptitle(f"Observed signal at receiver {rx_n+1}")
            time = t_vec
            idx = np.where(abs(r_k[rx_n]) > 0)
            axs_rx[0].plot(time[idx], r_k[rx_n][idx].real, label=f"Rx {rx_n+1}")
            axs_rx[1].plot(time[idx], r_k[rx_n][idx].imag, label=f"Rx {rx_n+1}")

            fig_tx, axs_tx = plt.subplots(2, sharex=True)
            fig_tx.suptitle(f"Signals transmitted to receiver {rx_n+1}")
            for tx_m in range(self.m_transmitters):
                time = t_vec - tau[self.m_transmitters * rx_n + tx_m]
                idx = np.where(abs(s_x[tx_m]) > 0)
                axs_tx[0].plot(time[idx], s_x[tx_m][idx].real, label=f"Tx {tx_m+1}")
                axs_tx[1].plot(time[idx], s_x[tx_m][idx].imag, label=f"Tx {tx_m+1}")

            axs_rx[0].set_title('Real part')
            axs_rx[1].set_title('Imaginary part')
            axs_rx[0].set_ylabel('Amplitude')
            axs_rx[1].set_ylabel('Amplitude')
            axs_rx[1].set_xlabel('Time (s)')
            axs_rx[1].legend(loc='center left', bbox_to_anchor=(1, 1), ncol=1)

            axs_tx[0].set_title('Real part')
            axs_tx[1].set_title('Imaginary part')
            axs_tx[0].set_ylabel('Amplitude')
            axs_tx[1].set_ylabel('Amplitude')
            axs_tx[1].set_xlabel('Time (s)')
            axs_tx[1].legend(loc='center left', bbox_to_anchor=(1, 1), ncol=1)

            plt.show()