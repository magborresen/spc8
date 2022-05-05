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

    def __init__(self, channels=5, f_sample=80e6, snr=None):
        self.f_sample = f_sample
        self.snr = snr
        self.channels = channels
        self.sigma_noise = snr

    def rx_tdm(self, tau: np.ndarray, tx_sig: np.ndarray, f_carrier: float, alpha=1, add_noise=True) -> np.ndarray:
        """
            Receiver a time-division multiplexed signal

            Args:
                tau (np.ndarray): Collection of time delays between transmitter -> target -> receiver
                tx_sig (np.ndarray): Collection of delayed transmitted signals
                f_carrier (float): Carrier frequency
                alpha (np.ndarray or float): Complex gain of the received signal

            Returns:
                y_k (np.ndarray): Collection of received signals for the oberservation
        """

        # Get received signals
        x_k = []
        s_k = []
        tau_idx = 0
        
        for n_ch in range(self.channels):
            # Set signals to 0
            sig_x = 0
            sig_s = 0
            
            for m_ch in range(tx_sig.shape[0]):
                # Calculate clean signal for antenna pair
                sig = np.exp(2j*np.pi*f_carrier*tau[tau_idx]) * tx_sig[m_ch]
                # Iterate signal with gain
                sig_x += alpha * sig
                # Iterate signal without gain
                sig_s += sig
                # Iterate tau counter
                tau_idx += 1  
            
            x_k.append(sig_x)
            s_k.append(sig_s)  
        
        x_k = np.array(x_k)
        s_k = np.array(s_k)

        # Get observation, add complex noise to signal
        if self.snr and add_noise:
            y_k = x_k + np.array(self.get_noise(x_k))
        else:pyth
            y_k = x_k

        return (s_k, y_k)

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

            sigma_signal = np.var(signal)
            sigma_complex_noise = sigma_signal/SNR
            self.sigma_noise = sigma_complex_noise
            sigma_real_noise = np.sqrt(sigma_complex_noise/2)

            v = np.random.normal(0, 1, size=(2, samples))
            W = sigma_real_noise * v[0,:] + 1j * sigma_real_noise * v[1,:]

            noise.append(W)

        return noise