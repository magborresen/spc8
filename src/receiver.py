"""
    Receiver object for radar system
"""

import numpy as np

class Receiver:
    """
        Radar receiver class

        Args:
            channels (int): Value for how many receivers should be generated
            f_sample (float): Sample frequency of the system

        Returns:
            no value
    """

    def __init__(self, channels=5, f_sample=80e6):
        self.f_sample = f_sample
        self.channels = channels
        self.sigma_noise = 0
        self.noise_figure = 12 # in dB
        self.temp = 293.15 # in Kelvin (=> 20 celcius)
        self.input_noise = -89.2 # At room temperature

    # @profile
    def rx_tdm(self, tau: np.ndarray, tx_sig: np.ndarray,
               f_carrier: float, alpha: np.ndarray, t_vec: np.ndarray,
               t_chirp) -> np.ndarray:
        """
            Receiver a time-division multiplexed signal

            Args:
                tau (np.ndarray): Collection of time delays
                                  between transmitter -> target -> receiver
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
            tau_id_sample = 0

            for m_ch in range(tx_sig.shape[0]):
                # Find which tau to use for the signal offset
                tau_sample = int(self.f_sample * t_chirp * tau_id_sample) - 1

                # Find the signal offset in samples
                offset = tau[tau_idx][tau_sample]

                # Get delay in number of samples
                delay = round(offset / t_vec[1])
                
                # Delay signal by desired number of samples (pad with 0)
                tx_sig_offset = np.r_[np.full(delay, 0), tx_sig[m_ch][:-delay]]

                # Calculate clean signal for antenna pair
                sig = np.exp(2j*np.pi*f_carrier*tau[tau_idx]) * tx_sig_offset
                # Iterate signal with gain
                sig_x += sig * alpha[n_ch]
                # Iterate signal without gain
                sig_s += sig
                # Iterate tau counter
                tau_id_sample += 1
                tau_idx += 1

            x_k.append(sig_x)
            s_k.append(sig_s)

        x_k = np.array(x_k)
        s_k = np.array(s_k)

        y_k = x_k

        return (s_k, y_k)

    # @profile
    def rx_tdm_optimized(self, tau: np.ndarray, tx_sig: np.ndarray,
               f_carrier: float, alpha: np.ndarray, t_vec: np.ndarray,
               t_chirp) -> np.ndarray:
        """
            Receiver a time-division multiplexed signal

            Args:
                tau (np.ndarray): Collection of time delays
                                  between transmitter -> target -> receiver
                tx_sig (np.ndarray): Collection of delayed transmitted signals
                f_carrier (float): Carrier frequency
                alpha (np.ndarray or float): Complex gain of the received signal

            Returns:
                y_k (np.ndarray): Collection of received signals for the oberservation
        """

        x_k = np.zeros((self.channels, tx_sig.shape[1]), dtype=np.complex128)
        tau_idx = 0

        for n_ch in range(self.channels):
            for m_ch in range(tx_sig.shape[0]):
                
                # Find which tau to use for the signal offset
                tau_sample = int(self.f_sample * t_chirp * m_ch) 

                # Delay signal by desired number of samples (pad with 0)
                offset = tau[tau_idx][tau_sample]
                delay = round(offset / t_vec[1])
                tx_sig_offset = np.r_[np.full(delay, 0), tx_sig[m_ch][:-delay]]

                # Calculate clean signal for antenna pair
                iota = tx_sig_offset != 0
                x_k[n_ch,iota] += np.exp(2j*np.pi*f_carrier*tau[tau_idx,iota]) * tx_sig_offset[iota] * alpha[n_ch]

                tau_idx += 1

        return x_k