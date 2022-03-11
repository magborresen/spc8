import numpy as np



class Observation():

    def __init__(self, mtransmitters, nreceivers):
        self.m_transmitters = mtransmitters
        self.n_receivers = nreceivers
        self._c = 300000
        self.tx_pos = None
        self.rx_pos = None
        self.alpha = 1
        self._fc = 40e6

    def place_antennas(self):
        """ Place collacted RX and TX antennas in 2D space

            Args:
                no value

            Returns:
                no value
        """
        self.tx_pos = [np.cos(np.linspace(0, 2*np.pi, self.m_transmitters)),
                       np.sin(np.linspace(0, 2*np.pi, self.m_transmitters))]

        self.rx_pos = [np.cos(np.linspace(0, 2*np.pi, self.n_receivers)),
                       np.sin(np.linspace(0, 2*np.pi, self.n_receivers))]

    def time_delay(self, x_tm: float, y_tm: float, x_rn: float, y_rn: float, x_k: float, y_k: float):
        """ Find time delay from transmitter to target to receiver

            Args:
                x_tm (float): x position of m'th transmitter
                y_tm (float): y position of m'th transmitter
                x_rn (float): x position of n'th receiver
                y_rn (float): y position of n'th receiver
                x_k (float): x position of the target
                y_k (float): y position of the target

            Returns:
                tau (float): Signal time delay
        """
        tau = 1 / self._c * (np.sqrt(x_tm - x_k**2 + (y_tm - y_k)**2) +
                                  np.sqrt(x_rn - x_k)**2 + (y_rn - y_k)**2)

        return tau

    def tx_signal(self, t, tau):
        """ Create the tx radar signal

            Args:
                t (float): time
                tau (float): time delay

            Returns:
                sx_m (float): Transmitted signal amplitude at t
        
        """
        sx_m = np.cos(2*np.pi*self._fc*(t-tau))

        return sx_m

    def observation(self, n, m, t):
        return

