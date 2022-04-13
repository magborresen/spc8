"""
    Receiver object for radar system
"""

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