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

    def tx_tdm(self, t_vec, t_rx, t0):
        """
            Transmit time division multiplexed signal
        """

        tx_start_times = np.linspace(t0, (self.channels-1)*(t_rx + self.t_chirp), self.channels)
        tx_stop_times = tx_start_times + self.t_chirp
        tx_sigs = []

        for idx, t_ch in enumerate(t_vec):
            if tx_start_times[idx] <= t_ch <= tx_stop_times[idx]:
                tx_sig = np.exp(1j*np.pi * self.bandwidth/self.t_chirp * t_ch**2)
            else:
                tx_sig = 0

            tx_sigs.append(tx_sig)

        return np.array(tx_sigs)

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
