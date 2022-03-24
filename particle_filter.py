import numpy as np
import matplotlib.pyplot as plt
from signal_model import Signal


class ParticleFilter(Signal):
    """
        Creates a particle filter object
    """

    def __init__(self, n_particles=100, k_tot=10, region=[2000, 2000],
                 time_step=0.4, m_transmitters=10,
                 n_receivers=10):
        self._n_particles = n_particles
        self._k_tot = k_tot
        self.region = region
        self._time_step = time_step
        self._m_transmitters = m_transmitters
        self._n_receivers = n_receivers
        self.phi_hat = None
        self.alpha_hat = None
        self.alpha = None
        self.theta = None
        self.acc = None
        self.weights = None
        self.posterior = None
        Signal.__init__(self, k_tot=self._k_tot, region=self.region, time_step=self._time_step,
                        m_transmitters=self._m_transmitters, n_receivers=self._n_receivers)

    def init_particles_uniform(self):
        """
            Initialize the particles based on a uniform distribution

            Args:
                no value

            Returns:
                no value
        """
        # Initialize positions
        p = np.random.uniform(low=-self.region[0],
                                       high=self.region[1],
                                       size=(self._n_particles, 2, 1))

        # Initialize velocities
        v = np.full((self._n_particles, 2, 1), 5)

        # Concatenate to create theta
        self.theta = np.concatenate((p, v), axis=1)

        # Initialize accelerations
        self.acc = np.zeros((self._n_particles, 2, 1))

        # Initialize gains
        self.alpha = np.ones((self._n_particles, 1))

    def init_weights(self):
        """
            Initialize weights for the associated particles
        """
        # Initialize weights
        w_value = 1 / self._n_particles

        # Normalize weights
        self.weights = np.full((self._n_particles, 1), w_value)

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

    def update_posterior(self, y_k, x_k):
        """
            Update the posterior probability for each particle target signal

            Args:
                y_k (np.ndarray): Observed signal for observation k
                x_k (np.ndarray): Target signal for observation k
        """
        sigma_w = np.var(self.weights)
        print(sigma_w)
        c = (2 * np.pi * sigma_w)**(-self._samples_per_obs * self._n_receivers)
        #self.posterior = [(c * np.exp(- 1 / np.var(sigma_w) * np.square(np.linalg.norm(y_k - x_k_i)))) for x_k_i in x_k]

        #self.posterior = np.array(self.posterior)

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

    def plot_weights(self):
        """
            Creates a scatter plot of the weights. Size of each point
            corresponds to the size of the associated weight.

            Args:
                no value

            Returns:
                no value
        """

        plt.scatter(self.theta[:,0], self.theta[:,1], s=10000*self.weights, alpha=0.5)
        plt.xlabel("x position [m]")
        plt.ylabel("y position [m]")
        plt.title("Particle locations with weights")
        plt.show()

## Testing
if __name__ == '__main__':
    pf = ParticleFilter(10, 2, [2000, 2000], 4e-4, 10, 10)
    pf.init_particles_uniform()
    pf.init_weights()
    y_k = pf.observe_y(0, 30.0)
    x_k = pf.observe_x(0, pf.theta[0])
    #print(x_k.shape)
    pf.update_posterior(y_k, x_k)
    #print(pf.posterior)
