import numpy as np
import matplotlib.pyplot as plt


class ParticleFilter():

    def __init__(self, n_particles: int):
        self._n_particles = n_particles
        self.phi_hat = None
        self.alpha_hat = None
        self.alpha = None
        self.theta = None
        self.n = None

    def init_particles_uniform(self, region: list):
        """
            Initialize the particles based on a uniform distribution

            Args:
                region (list): x and y limits of the observation region

            Returns:
                no value
        """
        self.theta = np.random.uniform(low=region[0],
                                       high=region[1],
                                       size=(self._n_particles, 4, 1))
        self.n = np.zeros((self._n_particles, 2, 1))
        self.alpha = np.ones((self._n_particles, 2, 1))

    def init_phi_ML(self, y_1, s_1):
        """
            Initialize the phi vector of the filter

            Args:
                y_1 (np.ndarray): Received signal in the 1st observation
                s_1 (np.ndarray): No gain received signal in the
                                  1st observation as a function of true phi

            Returns:
                no value
        """
        tmp = np.transpose(y_1) * s_1
        phi_idx = np.argmax(abs(tmp))
        self.phi_hat = tmp[phi_idx]

    def init_alpha_ML(self, y_1, s_1):
        """
            Initialize the alpha vector of the filter

            Args:
                y_1 (np.ndarray): Received signal in the 1st observation
                s_1 (np.ndarray): No gain received signal in the 
                                  1st observation as a function of phi_hat

            Returns:
                no value
        """
        self.alpha_hat = (np.transpose(s_1) * y_1) / np.square(np.linalg.norm(s_1))
