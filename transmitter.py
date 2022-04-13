"""
    Transmitter object for radar system.
"""
import numpy as np

class Transmitter:
    """
        Radar system transmitter

        Args:
            no value

        Returns:
            no value
    """

    def __init__(self, channels=5, f_carrier=10e9, t_chirp=1e-6, bandwidth=300e6, tx_power=30, mod="fmcw", prp=0.5):
        self.f_carrier = f_carrier
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        self.mod = mod
        self.channels = channels
        self.prp = prp
        self.t_chirp = t_chirp

    def tx_tdm(self, t_vec, t_rx, f_sample):
        """
            Transmit time division multiplexed signal
        """

        tx_array = []
        freq_array = []
        rx_samples = int(t_rx * f_sample)
        tx_samples = int(self.t_chirp * f_sample)

        for m in range(self.channels):
            tx_sig = np.zeros(t_vec.shape, dtype=np.complex128)
            freq = np.zeros(t_vec.shape)
            current_tx = m * (tx_samples + rx_samples)
            tx_stop = current_tx + tx_samples
            tx_sig[current_tx:tx_stop] = np.cos(2*np.pi + np.pi * self.bandwidth/self.t_chirp * t_vec[current_tx:tx_stop]**2)
            freq[current_tx:tx_stop] = self.bandwidth / self.t_chirp * t_vec[current_tx:tx_stop]
            tx_array.append(tx_sig)
            freq_array.append(freq)


        return np.array(tx_array), np.array(freq_array)
        