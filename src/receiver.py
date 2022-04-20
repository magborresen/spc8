"""
    Receiver object for radar system
"""

import numpy as np
import matplotlib.pyplot as plt

class Receiver:
    """
        Radar receiver class

        Args:
            no value

        Returns:
            no value
    """

    def __init__(self, channels=5, f_sample=600e6, snr=30):
        self.f_sample = f_sample
        self.snr = snr
        self.channels = channels

    def rx_tdm(self, tau: np.ndarray, tx_sig: np.ndarray, f_carrier: float, alpha=1+1j) -> np.ndarray:
        """
            Receiver a time-division multiplexed signal

            Args:
                tau (np.ndarray): time delays between transmitter -> target -> receiver
                tx_sig (np.ndarray): Delayed transmitted signal
                f_carrier (float): Carrier frequency
                alpha (np.ndarray or float): Complex gain of the received signal

            Returns:
                y_k (np.ndarray): Received signals for the oberservation
        """

        # Create received signal without noise
        x_k = [np.sum(alpha * np.exp(2j*np.pi*f_carrier*tau) * tx_sig)
               for n_ch in range(self.channels)]

        # Add complex noise to signal
        y_k = np.array(x_k) + np.array(self.get_noise(x_k))

        return y_k

    def get_noise(self, signals):
        """
            Add noise to received signals

            Args:
                signals (np.ndarray): Received signals

            Returns:
                signals (np.ndarray): Received signals with added noise
        """
        noise = []
        for signal in signals:
            samples = len(signal)
            SNR = 10.0**(self.snr/10.0)

            s_var = np.var(signal)
            W_var = s_var/SNR
            v_var = np.sqrt(W_var/2)

            v = np.random.normal(0, 1, size=(2, samples))
            W = v_var * v[0,:] + 1j * v_var * v[1,:]

            noise.append(W)
            # print(10.0*np.log10(s_var/W_var), 10.0*np.log10(s_var/np.var(W)))
        return noise