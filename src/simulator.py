import numpy as np
import matplotlib.pyplot as plt
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target
from multiprocessing import Pool
import time

class Simulator:
    """
        Simulator class for used for sequential simulation of particle
        filter and radar system observations

        Args:
            k_obs_tot (int): Total number of observations to
                             generate target states for
            radar (object): Radar system object
            target (object): Target object
            particle_filter (object): Particle Filter object
            animate_pf (bool): Whether to animate particle and target updates
                               in a plot

        Returns:
            no value
    """

    def __init__(self, k_obs_tot, radar, target, particle_filter, animate_pf=False):
        self.k_obs_tot = k_obs_tot
        self.radar = radar
        self.target = target
        self.animate_pf = animate_pf
        self.states = None
        self.generate_target_states()
        self.particle_filter = particle_filter
        self.target_state = self.states[0]
        #self.particle_filter.init_particles_uniform()
        self.particle_filter.init_particles_near_target(self.target_state)
        if self.animate_pf:
            plt.ion()
            # Setup the figure and axes...
            self.particle_fig, self.particle_ax = plt.subplots()
            self.particle_ax.set_title(f"{self.particle_filter.n_particles} Particles; Observation {0}")
            # Then setup animation
            self.setup_particle_animation()

    def generate_target_states(self):
        """
            Generate target states for the k_obs_tot observations

            Args:
                no value

            Returns:
                no value
        """
        self.states = self.target.generate_states(self.k_obs_tot, method="random")

    # @profile
    def target_estimate(self, k_obs, resampling="systemic"):
        """
            Estimate the target parameters for the k'th observation.
            Implemented in the naive Python way.
            
            Args:
                k_obs (int): k'th observation to estimate
                resampling (string): Resampling method
            Returns:
                state_est (list): Estimated state
        """
        self.target_state = self.states[k_obs]
        target_range, _ = self.radar.observation(k_obs, self.target_state, add_noise=True)
        for particle in range(self.particle_filter.n_particles):

            # Predict the particle positions using the state space model
            self.particle_filter.predict_particle(particle)

            # Find the distance to each particle
            particle_dist= self.radar.get_true_dist(self.particle_filter.theta_est[particle])
            
            # Get the observation likelihood given each particle observation
            self.particle_filter.likelihoods[particle] = self.particle_filter.get_likelihood(target_range, particle_dist)

        # Update the weights for all particles
        self.particle_filter.update_weights()
        effective_weights = self.particle_filter.neff()
        if effective_weights < self.particle_filter.n_particles / 2:
            # Resample the weights
            #print(f"Effetive weights is {effective_weights}... Resampling...")
            indexes = self.particle_filter.resample(strat=resampling)
            self.particle_filter.resample_from_index(indexes)

        if self.animate_pf:
            self.update_particle_animation()
            self.particle_ax.set_title(f"{self.particle_filter.n_particles} Particles; Observation {k_obs}")

        self.particle_filter.save_theta_hist()
        
        return self.particle_filter.get_estimated_state()

    # @profile
    def target_estimate_vectorized(self, k_obs):
        """
            Estimate the target parameters for the k'th observation.
            Implemented in a vectorized manner using Numpy.
            
            Args:
                k_obs (int): k'th observation to estimate
            Returns:
                state_est (list): Estimated state
        """
        self.target_state = self.states[k_obs]

        target_range, _ = self.radar.observation(k_obs, self.target_state, add_noise=True)

        # Predict the particle positions using the state space model
        self.particle_filter.predict_particle_vectorized()

        # Find the distance to each particle
        particle_dist = self.radar.get_true_dist_vectorized(self.particle_filter.theta_est)

        # Get the observation likelihood given each particle observation
        self.particle_filter.likelihoods = self.particle_filter.get_likelihood(target_range, particle_dist)

        # Update the weights for all particles
        self.particle_filter.update_weights()
        effective_weights = self.particle_filter.neff()
        if effective_weights < self.particle_filter.n_particles / 2:
            # Resample the weights
            #print(f"Effetive weights is {effective_weights}... Resampling...")
            indexes = self.particle_filter.resample()
            self.particle_filter.resample_from_index(indexes)

        if self.animate_pf:
            self.update_particle_animation()
            self.particle_ax.set_title(f"{self.particle_filter.n_particles} particles; Observation {k_obs}")

        self.particle_filter.save_theta_hist()

        return self.particle_filter.get_estimated_state()

    def target_estimate_multiprocessing(self, k_obs_tot, P=8):
        """
            Estimate the target parameters for the k'th observation, looping
            through k_obs_tot. Implemented as a multi-processing implementation.
            
            Args:
                k_obs (int): k'th observation to estimate
                P (int): Amount of workers
            Returns:
                all_state_est (list): List of all estimated states
        """
        with Pool(processes=P) as pool:
            result = pool.map(self.target_estimate_vectorized, range(k_obs_tot))
            pool.close()
            pool.join()
            # all_state_est = result.get()
        return result

    def setup_particle_animation(self):
        """
            Setup the animation scatter plot of particle positions

            Args:
                no value

            Returns:
                no value
        """
        x = np.append(self.target_state[0], self.particle_filter.theta_est[:,0])
        y = np.append(self.target_state[1], self.particle_filter.theta_est[:,1])

        plt.xlim(0, self.particle_filter.region)
        plt.ylim(0, self.particle_filter.region)
        particle_size = np.full(x.shape, 20.0)
        particle_size[1:] /= 20.0
        particle_color = np.full(x.shape, 'b')
        particle_color[0] = 'r'
        particle_alpha = np.full(x.shape, 0.1)
        particle_alpha[0] = 1.0
        self.particle_scat = self.particle_ax.scatter(x,
                                                      y,
                                                      alpha=particle_alpha,
                                                      s=particle_size,
                                                      c=particle_color)
        plt.pause(0.1)

    def update_particle_animation(self):
        """
            Update the particle positions in animation

            Args:
                no value

            Returns
                no value
        """
        pos = np.vstack((np.expand_dims(self.target_state[:2],axis=0),
                         self.particle_filter.theta_est[:, :2]))
        self.particle_scat.set_offsets(pos)
        self.particle_fig.canvas.draw_idle()
        plt.pause(0.1)

def test_functions(k, tx, rx, radar, target):
    """
        This functions times the different implementations of the function
        target_states in Simulator class.
    
        Args:
            no value
    
        Returns
            no value

    """
    print('Observations:', k)
    states = target.generate_states(k, method="random")

    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=10000, region=region_size)
    sim = Simulator(k, radar, target, pf, animate_pf=False)
    states_naive = []
    t0 = time.time()
    for i in range(k):
        states_naive.append(sim.target_estimate(i))
    t1 = time.time()
    print('\nNaive:', 'avg', (t1-t0)/k, 's')
    print('RMSE range =', np.sqrt(1/k * np.sum(np.square(np.array(states_naive[:2])-np.array(sim.states[:2])))))
    print('RMSE veloc =', np.sqrt(1/k * np.sum(np.square(np.array(states_naive[2:])-np.array(sim.states[2:])))))

    # for i, x in enumerate(sim.states):
    #     print(np.array(x)-np.array(states_naive[i]))

    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=10000, region=region_size)
    sim = Simulator(k, radar, target, pf, animate_pf=False)
    states_vector = []
    t0 = time.time()
    for i in range(k):
        states_vector.append(sim.target_estimate_vectorized(i))
    t1 = time.time()
    print('\nVectorized:', 'avg', (t1-t0)/k, 's')
    print('RMSE range =', np.sqrt(1/k * np.sum(np.square(np.array(states_vector[:2])-np.array(sim.states[:2])))))
    print('RMSE veloc =', np.sqrt(1/k * np.sum(np.square(np.array(states_vector[2:])-np.array(sim.states[2:])))))

    # for i, x in enumerate(sim.states):
    #     print(np.array(x)-np.array(states_vector[i]))

    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=10000, region=region_size)
    sim = Simulator(k, radar, target, pf, animate_pf=False)
    t0 = time.time()
    states_multi = sim.target_estimate_multiprocessing(k)
    t1 = time.time()
    print('\nMultiprocessing:', 'avg', (t1-t0)/k, 's')
    print('RMSE range =', np.sqrt(1/k * np.sum(np.square(np.array(states_multi[:2])-np.array(sim.states[:2])))))
    print('RMSE veloc =', np.sqrt(1/k * np.sum(np.square(np.array(states_multi[2:])-np.array(sim.states[2:])))))

    # for i, x in enumerate(sim.states):
    #     print(np.array(x)-np.array(states_multi[i]))

if __name__ == '__main__':
    region_size = 2000
    k = 10
    tx = Transmitter(channels=2, chirps=10)
    rx = Receiver(channels=10)
    radar = Radar(tx, rx, "tdm", region_size)
    t_obs_tot = radar.t_obs + radar.k_space
    target = Target(t_obs_tot)
  
    # test_functions(k, tx, rx, radar, target)
    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=10000, region=region_size)
    sim = Simulator(k, radar, target, pf, animate_pf=False)
    for i in range(k):
        # sim.target_estimate(i)
        sim.target_estimate_vector(i)
    # sim.target_estimate_multiprocessing(k, 8)
    # plt.waitforbuttonpress()