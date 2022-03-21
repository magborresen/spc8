import numpy as np
import matplotlib.pyplot as plt


class ParticleFilter():
    """
        Creates a particle filter object
    """

    def __init__(self, n_particles: int, region: list, fs: float):
        self._n_particles = n_particles
        self.region = region
        self._fs = fs
        self.phi_hat = None
        self.alpha_hat = None
        self.alpha = None
        self.theta = None
        self.acc = None
        self.weights = None
        self.posterior = None

    def init_particles_uniform(self):
        """
            Initialize the particles based on a uniform distribution

            Args:
                region (list): x and y limits of the observation region

            Returns:
                no value
        """
        # Initialize positions and velocities
        self.theta = np.random.uniform(low=-self.region[0],
                                       high=self.region[1],
                                       size=(self._n_particles, 4, 1))

        # Initialize accelerations
        self.acc = np.zeros((self._n_particles, 2, 1))

        # Initialize gains
        self.alpha = np.ones((self._n_particles, 2, 1))

    def init_weights(self):
        """
            Initialize weights for the associated particles
        """
        # Initialize weights
        self.weights = np.random.uniform(low=0,
                                         high=self.region[1],
                                         size=(self._n_particles))

        # Normalize weights
        self.weights = np.abs(self.weights) / abs(np.sum(self.weights))

    def init_phi_ml(self, y_1, s_1):
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

    def init_alpha_ml(self, y_1, s_1):
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

    def update_particles(self, sk_m: np.ndarray, yk_m: np.ndarray, timestep: float):
        """
            Update particle parameters for the k'th observation
        """
        # Update positions
        self.theta[:,0:2] = (self.theta[:,0:2] + self.theta[:,2:] *
                             timestep + timestep**2 * self.acc / 2)

        # Update velocities
        self.theta[:,2:] = self.theta[:,2:] + timestep*self.acc

        # Update alphas
        self.alpha = (np.transpose(sk_m) * yk_m) / np.square(np.norm(sk_m))

        return None

    def update_weights(self):
        """
            Update weights of the associated particles
        """
        return None

    def update_posterior(self):
        return None 

    def plot_particles(self):
        """
            Plot the particle positions in the k'th observation

            Args:
                no value

            Returns:
                no value
        """
        plt.scatter(self.theta[:,0], self.theta[:,1])
        plt.xlabel("x position [m]")
        plt.ylabel("y position [m]")
        plt.title("Particle Locations")
        plt.show()

## Testing
if __name__ == '__main__':
    pf = ParticleFilter(1000, [2000, 2000])
    pf.init_particles_uniform()
    pf.init_weights()
