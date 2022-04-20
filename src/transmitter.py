"""
    Transmitter object for radar system.
"""
import numpy as np
import matplotlib.pyplot as plt

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

    def tx_tdm(self, t_vec, t_rx, f_sample, plot=False):
        """
            Transmit time division multiplexed signal
        """

        tx_array = []
        freq_array = []
        rx_samples = int(t_rx * f_sample)
        tx_samples = int(self.t_chirp * f_sample)
        t_vec_sig = t_vec[0:tx_samples]

        for m_ch in range(self.channels):
            tx_sig = np.zeros(t_vec.shape, dtype=np.complex128)
            freq = np.zeros(t_vec.shape)
            current_tx = m_ch * (tx_samples + rx_samples)
            tx_stop = current_tx + tx_samples
            tx_sig[current_tx:tx_stop] = np.exp(1j*np.pi * self.bandwidth/self.t_chirp * t_vec_sig**2)
            freq[current_tx:tx_stop] = self.bandwidth / self.t_chirp * t_vec_sig
            tx_array.append(tx_sig)
            freq_array.append(freq)

        tx_array = np.array(tx_array)
        freq = np.array(freq)

        if plot:
            self.plot_tx(t_vec, tx_array, f_sample)

        return np.array(tx_array), np.array(freq_array)

    def plot_tx(self, t_vec, tx_sig, f_sample):
        """
            Plot the transmitted signals
        """


        fig, axs = plt.subplots(self.channels, 2)
        for idx, sig in enumerate(tx_sig):
            tx_period = np.where(abs(sig) > 0)
            axs[idx][0].plot(t_vec[tx_period]*1e6, sig[tx_period].real)
            axs[idx][0].plot(t_vec[tx_period]*1e6, sig[tx_period].imag)
            fft_sig = np.fft.fft(sig[tx_period])
            N = len(fft_sig)
            T = N/f_sample
            n = np.arange(N)
            freq = n/T
            axs[idx][1].plot(freq/1e7, np.abs(fft_sig))

        fig.suptitle("Tx channel signals over time")
        plt.show()
