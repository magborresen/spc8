"""
    Implementation of particle filter for target tracking.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from filterpy.monte_carlo import systematic_resample, residual_resample, stratified_resample, multinomial_resample
from scipy.signal import butter, sosfilt


class ParticleFilter():
    """
        Creates a particle filter object
    """

    def __init__(self, t_obs: float, n_rx_channels: int, n_particles=100, region=2000):
        self.n_particles = n_particles
        self.n_rx_channels = n_rx_channels
        self.region = region
        self._t_obs = t_obs
        self.alpha_est = None
        self.theta_est = None
        self.theta_est_all_k = None
        self.particle_acc = None
        self.weights = None
        self.likelihoods = None
        self.light_speed = 300e6
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
        particle_pos = np.random.uniform(low=0,
                              high=self.region,
                              size=(self.n_particles, 2, 1))

        # Initialize particle velocities uniformly
        particle_velocity = np.random.uniform(low=-22, high=22, size=(self.n_particles, 2, 1))

        # Concatenate to create the position and velocity vector theta
        self.theta_est = np.concatenate((particle_pos, particle_velocity), axis=1)

        self.theta_est_all_k = self.theta_est

        # Initialize accelerations
        self.particle_acc = np.random.normal(loc=0, scale=1, size=(self.n_particles, 2, 1))

        # Initialize gains for each particle at each receiver
        self.alpha_est = np.zeros((self.n_particles, self.n_rx_channels), dtype=np.complex128)

        self.likelihoods = np.zeros((self.n_particles, 1))

    def init_particles_near_target(self, target_pos):
        """
            Initialize the particles to be near the target
        """
        particle_pos_x = np.random.normal(loc=target_pos[0], scale=100, size=(self.n_particles,1))
        particle_pos_y = np.random.normal(loc=target_pos[1], scale=100, size=(self.n_particles,1))
        particle_pos = np.zeros((self.n_particles, 2, 1))
        particle_pos[:,0] = particle_pos_x
        particle_pos[:,1] = particle_pos_y

        # Initialize particle velocities uniformly
        particle_velocity = np.random.uniform(low=-22, high=22, size=(self.n_particles, 2, 1))

        # Concatenate to create the position and velocity vector theta
        self.theta_est = np.concatenate((particle_pos, particle_velocity), axis=1)

        self.theta_est_all_k = self.theta_est

        # Initialize accelerations
        self.particle_acc = np.zeros((self.n_particles, 2, 1))

        # Initialize likelihoods
        self.likelihoods = np.zeros((self.n_particles, 1))

    def init_weights(self):
        """
            Initialize weights for the associated particles
        """
        # Initialize weights
        w_value = 1 / self.n_particles

        # Normalize weights
        self.weights = np.full((self.n_particles, 1), w_value)


    def predict_particle(self, particle):
        """
            Update particle parameters for the k'th observation
        """

        # Update positions
        self.theta_est[particle][:2] = (self.theta_est[particle][:2] +
                                        self.theta_est[particle][2:] *
                                        self._t_obs + self._t_obs**2 *
                                        self.particle_acc[particle] / 2)

        # Update velocities
        self.theta_est[particle][2:] = (self.theta_est[particle][2:] +
                                        self._t_obs * self.particle_acc[particle])

    def save_theta_hist(self):
        """
            Save the history of predicted positions

            Args:
                None

            Returns:
                None
        """
        self.theta_est_all_k = np.c_[self.theta_est_all_k, self.theta_est]


    def get_likelihood(self, target_range, particle_range):
        """
            Update the likelihood for each particle

            Args:
                target_range (np.ndarray): Estimated target range to each
                particle_range (np.ndarray): Particle range to each receiver
        """
        measurement_likelihood_sample = 1.0
        for idx, _ in enumerate(target_range):
            prob = np.exp(-(particle_range[idx] - target_range[idx])**2 /
                           (2*0.79**2))

            measurement_likelihood_sample *= prob

        return measurement_likelihood_sample

    def update_weights(self):
        """
            Update particle weights
        """

        self.weights *= self.likelihoods
        self.normalize_weights()

    def normalize_weights(self):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples
        sum_weights = np.sum(self.weights)

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print(f"Weight normalization failed: sum of all weights is {sum_weights} (weights will be reinitialized)")
            # Set uniform weights
            w_value = 1 / self.n_particles
            self.weights = np.full((self.n_particles, 1), w_value)
            return None

        # Return normalized weights
        self.weights /= sum_weights

    def resample(self, strat="systemic"):
        """
            Resamples the weights using filterpy functions and returns the indexes to use
            after resampling.

            Args:
                strat (string): Strategy to use for resampling

            Returns:
                Indexes (list): List of particle weights that should survive the resampling
        """

        if strat == "systemic":
            indexes = systematic_resample(self.weights)
        if strat == "residual":
            indexes = residual_resample(self.weights)
        if strat == "stratified":
            indexes = stratified_resample(self.weights)
        if strat == "multinomial":
            indexes = multinomial_resample(self.weights)

        return indexes

    def neff(self):
        """
            Calculate the number of effective particles (weights)
        """

        return 1. / np.sum(np.square(self.weights))

    def resample_from_index(self, indexes):
        """
            Resample the particles from the indexes found by the resampling scheme.
        """
        self.theta_est[:] = self.theta_est[indexes]
        self.theta_est[:,:2] *= np.abs(np.random.normal(loc=1, scale=0.01, size=self.theta_est[:,:2].shape))
        self.weights.resize(self.n_particles, 1)
        self.weights.fill(1.0 / self.n_particles)

    def get_posterior(self, target_true_hist):
        """
            Get the estimated posterior distribution

            Args:
                target_true_hist (np.ndarray): History of true distances and
                                               velocities of the target

            Returns:
                posterior_est (float): Estimated posterior pdf
        """
        omega_diff = target_true_hist - self.theta_est_all_k

        omega_idx = np.where(omega_diff > 1e-15, 1, 0)

        posterior_est = np.sum(self.weights * omega_idx)

        return posterior_est

    def get_estimated_state(self):
        """
            Get the estimated state of the target

            Args:
                None

            Returns:
                state_est (float): Estimated state of the target
        """
        state_est = np.sum(self.theta_est * np.expand_dims(self.weights, axis=1), axis=0)

        return state_est

    def plot_particles(self, target_state):
        """
            Plot the particle positions in the k'th observation

            Args:
                no value

            Returns:
                no value
        """
        plt.scatter(self.theta_est[:,0], self.theta_est[:,1])
        plt.scatter(target_state[0],target_state[1])
        # plt.scatter(self.region*0.5, 0, alpha=0.25, s=9e4)

        for i, txt in enumerate(self.theta_est):
            plt.annotate(i, (self.theta_est[i,0] + self.region*5e-3, self.theta_est[i,1] + self.region*5e-3))

        plt.xlim((0,self.region))
        plt.ylim((0,self.region))
        plt.gca().set_aspect('equal')
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
