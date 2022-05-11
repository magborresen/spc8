import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target


class Simulator:

    def __init__(self, k_obs_tot, radar, target, particle_filter, animate_pf=False):
        self.k_obs_tot = k_obs_tot
        self.radar = radar
        self.target = target
        self.states = self.target.generate_states(k_obs_tot, method='linear_away')
        self.particle_filter = particle_filter
        self.particle_filter.init_particles_uniform()
        if animate_pf:
            plt.ion()
            # Setup the figure and axes...
            self.particle_fig, self.particle_ax = plt.subplots()
            # Then setup FuncAnimation.
            self.setup_particle_animation()

    def target_estimate(self, k_obs):
        """
            Estimate the target parameters for the k'th observation
        """
        target_state = self.states[k_obs]
        _, rx_sig, alpha = self.radar.observation(k_obs, target_state, add_noise=True)

        for particle in range(self.particle_filter.n_particles):
            # Create an observation with each particle location
            s_sig_i, _, alpha = self.radar.observation(k_obs,
                                                self.particle_filter.theta_est[particle])

            # Update particle positions
            self.particle_filter.update_particle(particle, s_sig_i, rx_sig)
            #print(abs(self.particle_filter.alpha_est[particle]))
            #print(self.particle_filter.alpha_est[particle])

            _, x_k_i, alpha = self.radar.observation(k_obs,
                                              self.particle_filter.theta_est[particle])

            self.update_particle_animation()

            # Update likelihood for each particle
            self.particle_filter.update_likelihood(particle,
                                                   rx_sig,
                                                   x_k_i,
                                                   self.radar.receiver.sigma_noise)

        # Update the weights for all particles
        #self.particle_filter.update_weights()

    def setup_particle_animation(self):
        """
            Setup the animation scatter plot of particle positions
        """
        x = self.particle_filter.theta_est[:,0]
        y = self.particle_filter.theta_est[:,1]
        colors = np.linalg.norm(self.particle_filter.theta_est[:,2:], axis=1)

        plt.xlim(0, self.particle_filter.region)
        plt.ylim(0, self.particle_filter.region)

        self.particle_scat = self.particle_ax.scatter(x, y, c=colors)

    def update_particle_animation(self):
        """
            Update the particle positions in animation
        """
        pos = self.particle_filter.theta_est[:, :2]
        self.particle_scat.set_offsets(pos)

        self.particle_fig.canvas.draw_idle()
        plt.pause(0.1)

if __name__ == '__main__':
    region_size = 150
    k = 10
    tx = Transmitter(channels=2, t_chirp=60e-6, chirps=2)
    rx = Receiver(channels=2)

    radar = Radar(tx, rx, "tdm", region_size, snr=30)
    target = Target(radar.t_obs + radar.k_space)
    pf = ParticleFilter(radar.t_obs + radar.k_space, rx.channels, n_particles=10, region=region_size)

    sim = Simulator(k, radar, target, pf)
    plt.waitforbuttonpress()