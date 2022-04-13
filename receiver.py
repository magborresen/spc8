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

    def rx_tdm(self, tau: np.ndarray, tx_sig: np.ndarray, w_k: np.ndarray, f_carrier: float, alpha=1+1j) -> np.ndarray:
        """
            Receiver a time-division multiplexed signal

            Args:
                tau (np.ndarray): time delays between transmitter -> target -> receiver
                tx_sig (np.ndarray): Delayed transmitted signal
                w_k (np.ndarray): Circular symmetric gaussian noise sequence
                f_carrier (float): Carrier frequency
                alpha (np.ndarray or float): Complex gain of the received signal

            Returns:
                y_k (np.ndarray): Received signals for the oberservation
        """

        # Create received signal without noise
        x_k = [np.sum(alpha * np.exp(2j*np.pi*f_carrier*tau) * tx_sig) for n_ch in range(self.channels)]

        # Add complex noise to signal
        y_k = x_k + w_k

        return np.array(y_k)
