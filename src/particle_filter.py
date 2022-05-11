"""
    Implementation of particle filter for target tracking.
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class ParticleFilter():
    """
        Creates a particle filter object
    """

    def __init__(self, t_obs: float, n_rx_channels: int, n_particles=100, region=2000):
        self.n_particles = n_particles
        self.n_rx_channels = n_rx_channels
        self.region = region
        self._t_obs = t_obs
        self.phi_hat = None
        self.alpha_hat = None
        self.alpha_est = None
        self.theta_est = None
        self.acc = None
        self.weights = None
        self.likelihoods = None
        self.init_particles_uniform()
        self.init_weights()

    def init_particles_uniform(self) -> None:
        """
            Initialize the particles based on a uniform distribution

            Args:
                no value

            Returns:
                no value
        """
        # Initialize particle positions uniformly
        target_pos = np.random.uniform(low=0,
                              high=self.region,
                              size=(self.n_particles, 2, 1))

        # Initialize particle velocities uniformly
        target_velocity = np.random.uniform(low=1, high=22, size=(self.n_particles, 2, 1))

        # Concatenate to create the position and velocity vector theta
        self.theta_est = np.concatenate((target_pos, target_velocity), axis=1)

        # Initialize accelerations
        self.acc = np.zeros((self.n_particles, 2, 1))

        # Initialize gains for each particle at each receiver
        self.alpha_est = np.ones((self.n_particles, self.n_rx_channels), dtype=np.complex128)

        self.likelihoods = np.zeros((self.n_particles, 1))

    def init_weights(self):
        """
            Initialize weights for the associated particles
        """
        # Initialize weights
        w_value = 1 / self.n_particles

        # Normalize weights
        self.weights = np.full((self.n_particles, 1), w_value)


    def update_particle(self, particle, sk_n: np.ndarray, yk_n: np.ndarray):
        """
            Update particle parameters for the k'th observation
        """
        # Update positions
        self.theta_est[particle][:2] = (self.theta_est[particle][:2] +
                                         self.theta_est[particle][2:] *
                                         self._t_obs + self._t_obs**2 * self.acc[particle] / 2)

        # Update velocities
        self.theta_est[particle][2:] = (self.theta_est[particle][2:] +
                                        self._t_obs * self.acc[particle])

        # Update alphas for each receiver
        for i in range(sk_n.shape[0]):
            self.alpha_est[particle][i] = (np.dot(np.conjugate(sk_n[i]).T, yk_n[i]) /
                                                 np.square(np.linalg.norm(sk_n[i])))

    def update_likelihood(self, particle, y_k, x_k_i, sigma_w):
        """
            Update the likelihood for each particle
        """
        self.likelihoods[particle] = (np.exp(- 1 / sigma_w *
                                      np.square(np.linalg.norm(y_k - x_k_i))))

        #print(np.square(np.linalg.norm(y_k[:,100] - x_k_i[:,100])))

    def update_weights(self):
        """
            Update the posterior probability for each particle target signal

            Args:
                y_k (np.ndarray): Observed signal for observation k
                x_k (np.ndarray): Target signal for observation k
                sigma_w (float): Signal noise variance
        """

        numerator = self.weights * self.likelihoods
        denominator = np.sum(self.weights * self.likelihoods)

        self.weights = numerator / denominator


    def plot_particles(self):
        """
            Plot the particle positions in the k'th observation

            Args:
                no value

            Returns:
                no value
        """
        plt.scatter(self.theta_est[:,0], self.theta_est[:,1])
        plt.xlabel("x position [m]")
        plt.ylabel("y position [m]")
        plt.title("Particle Locations")
        plt.show()

    def plot_weights(self):
        """
            Creates a scatter plot of the weights. Size of each point
            corresponds to the size of the associated weight.

            Args:
                no value

            Returns:
                no value
        """

        plt.scatter(self.theta_est[:,0], self.theta_est[:,1], s=10000*self.weights, alpha=0.5)
        plt.xlabel("x position [m]")
        plt.ylabel("y position [m]")
        plt.xlim(0, self.region)
        plt.ylim(0, self.region)
        plt.title("Particle locations with weights")
        plt.show()

## Testing
if __name__ == '__main__':
    pf = ParticleFilter(1.33e-7 + 1.8734e-5, 1000)
    pf.init_particles_uniform()
    pf.init_weights()
    pf.plot_particles()
    