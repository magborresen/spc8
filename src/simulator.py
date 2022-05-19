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
        self.states = self.target.generate_states(k_obs_tot, method='random')
        self.particle_filter = particle_filter
        self.target_state = self.states[0]
        self.particle_filter.init_particles_near_target(self.target_state)
        if self.animate_pf:
            plt.ion()
            # Setup the figure and axes...
            self.particle_fig, self.particle_ax = plt.subplots()
            self.particle_ax.set_title(f"Observation {0}")
            # Then setup animation
            self.setup_particle_animation()

    def target_estimate(self, k_obs):
        """
            Estimate the target parameters for the k'th observation
        """
        self.target_state = self.states[k_obs]
        target_range, alpha = self.radar.observation(k_obs, self.target_state, add_noise=True)
        for particle in range(self.particle_filter.n_particles):

            # Predict the particle positions using the state space model
            self.particle_filter.predict_particle(particle)

            # Find the distance to each particle
            particle_dist, _ = self.radar.get_true_dist(self.particle_filter.theta_est[particle])

            # Get the observation likelihood given each particle observation
            self.particle_filter.get_likelihood(particle,
                                                target_range,
                                                particle_dist)
        #print(self.particle_filter.likelihoods)
        # Update the weights for all particles
        self.particle_filter.update_weights()

        if self.particle_filter.neff() < self.particle_filter.n_particles / 2:
            # Resample the weights
            print("Not enough effective weights... Resampling...")
            indexes = self.particle_filter.resample()
            self.particle_filter.resample_from_index(indexes)

        if self.animate_pf:
            self.update_particle_animation()
            self.particle_ax.set_title(f"Observation {k_obs}")

    def setup_particle_animation(self):
        """
            Setup the animation scatter plot of particle positions
        """
        x = np.append(self.target_state[0], self.particle_filter.theta_est[:,0])
        y = np.append(self.target_state[1], self.particle_filter.theta_est[:,1])

        plt.xlim(0, self.particle_filter.region)
        plt.ylim(0, self.particle_filter.region)
        particle_size = np.full(x.shape, 20.0)
        particle_size[1:] /= 20.0
        particle_color = np.full(x.shape, 'b')
        particle_color[0] = 'r'
        particle_alpha = np.full(x.shape, 1.0)
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
        """
        pos = np.vstack((np.expand_dims(self.target_state[:2],axis=0),
                         self.particle_filter.theta_est[:, :2]))
        self.particle_scat.set_offsets(pos)
        self.particle_fig.canvas.draw_idle()
        plt.pause(0.1)

if __name__ == '__main__':
    region_size = 2000
    k = 100
    tx = Transmitter(channels=2, chirps=10)
    rx = Receiver(channels=2)

    radar = Radar(tx, rx, "tdm", region_size)
    t_obs_tot = radar.t_obs + radar.k_space
    target = Target(t_obs_tot)
    pf = ParticleFilter(t_obs_tot, rx.channels, n_particles=10, region=region_size)

    sim = Simulator(k, radar, target, pf, animate_pf=True)
    for i in range(k):
        sim.target_estimate(i)
    plt.waitforbuttonpress()
