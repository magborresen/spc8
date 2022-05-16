import numpy as np
import matplotlib.pyplot as plt
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target


class Simulator:
    """
        Simulator class for used for sequential simulation of particle
        filter and radar system observations
    """

    def __init__(self, k_obs_tot, radar, target, particle_filter, animate_pf=False):
        self.k_obs_tot = k_obs_tot
        self.radar = radar
        self.target = target
        self.animate_pf = animate_pf
        self.states = self.target.generate_states(k_obs_tot, method='linear_away')
        self.particle_filter = particle_filter
        self.particle_filter.init_particles_uniform()
        self.target_state = self.states[0]
        if self.animate_pf:
            plt.ion()
            # Setup the figure and axes...
            self.particle_fig, self.particle_ax = plt.subplots()
            # Then setup animation
            self.setup_particle_animation()

    def target_estimate(self, k_obs):
        """
            Estimate the target parameters for the k'th observation
        """
        self.target_state = self.states[k_obs]
        _, rx_sig, alpha = self.radar.observation(k_obs, self.target_state, add_noise=True)
        target_range = self.particle_filter.get_range(rx_sig)[0]
        for particle in range(self.particle_filter.n_particles):
            # Create an observation with each particle location
            s_sig_i, _, alpha = self.radar.observation(k_obs,
                                                       self.particle_filter.theta_est[particle])

            # Predict the particle positions using the state space model
            self.particle_filter.predict_particle(particle, s_sig_i, rx_sig)

            # Generate observations for each of the particles
            #_, x_k_i, alpha = self.radar.observation(k_obs,
            #                                         self.particle_filter.theta_est[particle])

            particle_dist = self.radar.get_true_dist(self.particle_filter.theta_est[particle])

            # Get the observation likelihood given each particle observation
            self.particle_filter.get_likelihood(particle,
                                                target_range,
                                                particle_dist)

        #for i in range(self.particle_filter.n_particles):
        #    print(f'Particle {i}: {self.particle_filter.likelihoods[i]}')

        # Update the weights for all particles
        self.particle_filter.update_weights()
        if self.particle_filter.neff() < self.particle_filter.n_particles / 2:
            # Resample the weights
            print("Not enough effective weights... Resampling...")
            indexes = self.particle_filter.resample()
            self.particle_filter.resample_from_index(indexes)

        if self.animate_pf:
            self.update_particle_animation()

    def setup_particle_animation(self):
        """
            Setup the animation scatter plot of particle positions
        """
        x = np.append(self.target_state[0], self.particle_filter.theta_est[:,0])
        y = np.append(self.target_state[1], self.particle_filter.theta_est[:,1])

        plt.xlim(0, self.particle_filter.region)
        plt.ylim(0, self.particle_filter.region)
        particle_size = np.full(x.shape, 6.0)
        particle_size[1:] /= 6.0
        particle_marker = np.full(x.shape, '.')
        particle_marker[0] = 'x'
        self.particle_scat = self.particle_ax.scatter(x, y, s=particle_size)

    def update_particle_animation(self):
        """
            Update the particle positions in animation
        """
        pos = np.vstack((np.expand_dims(self.target_state[:2],axis=0), self.particle_filter.theta_est[:, :2]))
        self.particle_scat.set_offsets(pos)
        self.particle_fig.canvas.draw_idle()
        plt.pause(0.1)

if __name__ == '__main__':
    region_size = 2000
    k = 10
    tx = Transmitter(channels=2, t_chirp=60e-6, chirps=10)
    rx = Receiver(channels=2)

    radar = Radar(tx, rx, "tdm", region_size, snr=50)
    t_obs_tot = radar.t_obs + radar.k_space
    target = Target(t_obs_tot)
    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=50, region=region_size)

    sim = Simulator(k, radar, target, pf, animate_pf=True)
    for i in range(k):
        print(f"Observation {i}")
        sim.target_estimate(i)
    plt.waitforbuttonpress()
