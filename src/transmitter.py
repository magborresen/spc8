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

    def __init__(self, channels=5, f_carrier=10e9, t_chirp=1e-6, bandwidth=300e6, tx_power=30, mod="lfm", prp=0.5, mult="tdm"):
        self.f_carrier = f_carrier
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        self.mod = mod
        self.channels = channels
        self.prp = prp
        self.t_chirp = t_chirp
        self.mult = mult
        self.chirps = 2

    def tx_tdm(self, t_vec, tau):
        """
            Transmit time division multiplexed signal
        """
        
        start_times = np.array([[(self.t_chirp*tx) + self.t_chirp*self.channels*chirp for chirp in range(self.chirps)] for tx in range(self.channels)])
        tx_sigs = []
        
        for tx in range(self.channels):
            # Create an empty array for the tx signal
            tx_sig = np.zeros(t_vec.shape, dtype=np.complex128)
            
            for chirp in range(self.chirps):
                # Find which times in t_vec that corresponds to transmitting with a given transmitter    
                tx_times = ((start_times[tx][chirp] <= t_vec) & (t_vec <= self.t_chirp + start_times[tx][chirp]))

                # Create the chirp at the transmitter times
                tx_sig[tx_times] = np.exp(1j*np.pi * self.bandwidth/self.t_chirp * (t_vec[tx_times]-t_vec[tx_times][0])**2)

            tx_sigs.append(tx_sig)
                
        for i in range(self.channels):
            plt.plot(t_vec*1e6, tx_sigs[i].real, label=f'Tx {i}')
        plt.legend()
       
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
