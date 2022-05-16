"""
    Implementation of particle filter for target tracking.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from filterpy.monte_carlo import systematic_resample
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
        self.phi_hat = None
        self.alpha_hat = None
        self.alpha_est = None
        self.theta_est = None
        self.acc = None
        self.weights = None
        self.likelihoods = None
        self.light_speed = 300e6
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
        target_velocity = np.random.uniform(low=-22, high=22, size=(self.n_particles, 2, 1))

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


    def predict_particle(self, particle, sk_n: np.ndarray, yk_n: np.ndarray):
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
            self.alpha_est[particle][i] = (np.dot(np.conjugate(sk_n[i]), yk_n[i].T) /
                                                 np.square(np.linalg.norm(sk_n[i])))

    def update_likelihood_fft(self, particle, y_k, x_k_i):
        """
            Update the likelihood for each particle
        """
        y_k_fft = np.abs(self.fft_filter(y_k))
        x_k_i_fft = np.abs(self.fft_filter(x_k_i))

        temp = 0
        for idx in range(y_k_fft.shape[0]):
            temp += abs(np.trapz(y_k_fft[idx] - x_k_i_fft[idx]))

        self.likelihoods[particle] = np.exp(-temp*1e12)

        range_vec = self.get_range(y_k)
        print('Range difference (y_k)', range_vec[0] - range_vec[1])


    def get_likelihood(self, particle, target_range, particle_range):
        """
            Update the likelihood for each particle
        """
        # Compute the range FFT of the particle
        #particle_range = self.get_range(x_k_i)

        self.likelihoods[particle] = stats.norm(particle_range[0], 100).pdf(target_range)

    def update_weights(self):
        """
            Update the posterior probability for each particle target signal

            Args:
                no value

            Returns:
                no value
        """

        numerator = self.weights * self.likelihoods
        denominator = np.sum(self.weights * self.likelihoods)

        self.weights = numerator / denominator

    def resample(self, strat="systemic"):
        """
            Resamples the weights using filterpy functions
        """

        if strat=="systemic":
            indexes = systematic_resample(self.weights)

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
        self.theta_est[:,2] *= np.random.normal(loc=0, scale=100, size=self.theta_est[:,2].shape)

        self.weights.resize(self.n_particles, 1)
        self.weights.fill(1.0 / self.n_particles)

    def get_range(self, sig_vec):
        """
            Compute the range FFT of given FMCW IF signal.

            Args:
                sig_vec (np.ndarray): Array of received signals

            Returns:
                range_vec (np.ndarray): Vector of scalar ranges for each received signal
        """
        range_vec = []

        for _, sig in enumerate(sig_vec):
            # Calculate FFT of signal
            fft_sig = np.fft.fft(sig)
            # Find sample with highest power
            sample = np.argmax(2.0/len(fft_sig) * np.abs(fft_sig))

            fft_samples = len(fft_sig)
            fft_times = fft_samples/(80e6)
            fft_idx = np.arange(fft_samples, dtype=np.float64)
            freq = fft_idx/fft_times
            fft_range = (freq * self.light_speed / (2.0 * 300e6/60e-6))
            range_vec.append(fft_range[sample])

        return np.array(range_vec)

    def fft_filter(self, sig_vec, offset=100):
        """
            Calculate fft for signals, find where the power is located and
            neglect samples that is "offset" samples away from the desired
            spike. 

            Args:
                sig_vec (np.ndarray): Collection of signals
                offset (int): How many samples to take from either side

            Returns:
                filtered_vec (np.ndarray): Filtered signals
        """        

        # Create the filter
        nyq = 80e6
        sos = butter(10, 8e6, fs=nyq, btype='low', analog=False, output='sos')

        fft_sig_vec = []
        
        for idx, sig in enumerate(sig_vec):
            # Calculate FFT of signal
            fft_sig = np.fft.fft(sig)
            # Find sample with highest power
            sample = np.argmax(2.0/len(fft_sig) * np.abs(fft_sig))
            # Check if sample offset becomes negative (Go with first samples)
            if sample <= offset:
                start = 0
                stop = offset*2
            # Check if sample offset becomes to large (Go with last samples)
            elif sample+offset >= fft_sig.shape[0]:
                start = sig.shape[0] - offset*2
                stop = sig.shape[0]
            # Else, take samples +-offset
            else:
                start = sample-offset
                stop = sample+offset
            
            fft_sig = sosfilt(sos, fft_sig[start:stop])
            fft_sig_vec.append(fft_sig)
            
        return np.array(fft_sig_vec)


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
    