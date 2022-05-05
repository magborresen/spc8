import numpy as np
import matplotlib.pyplot as plt
from radar import Radar
from receiver import Receiver
from transmitter import Transmitter
from particle_filter import ParticleFilter
from target import Target


class Simulator:

    def __init__(self, k_obs_tot, radar, target, particle_filter):
        self.k_obs_tot = k_obs_tot
        self.radar = radar
        self.target = target
        self.states = self.target.generate_states(k_obs_tot, method='linear_away')
        self.particle_filter = particle_filter
        self.particle_filter.init_particles_uniform()

    def target_estimate(self, k_obs):
        """
            Estimate the target parameters for the k'th observation
        """
        target_state = self.states[k_obs]
        _, rx_sig = self.radar.observation(k_obs, target_state, add_noise=True)

        for particle in range(self.particle_filter.n_particles):
            # Create an observation with each particle location
            s_sig_i, _ = self.radar.observation(k_obs,
                                                self.particle_filter.theta_est[particle])

            # Update particle positions
            self.particle_filter.update_particle(particle, s_sig_i, rx_sig)

            _, x_k_i = self.radar.observation(k_obs,
                                              self.particle_filter.theta_est[particle],
                                              alpha=self.particle_filter.alpha_est[particle])
            #print(self.particle_filter.alpha_est[particle])

            # Update likelihood for each particle
            self.particle_filter.update_likelihood(particle,
                                                   rx_sig,
                                                   x_k_i,
                                                   self.radar.receiver.sigma_noise)

        # Update the weights for all particles
        self.particle_filter.update_weights()


if __name__ == '__main__':
    k = 10
    tx = Transmitter(channels=2, t_chirp=60e-6, chirps=2)
    rx = Receiver(channels=2, snr=30)

    radar = Radar(tx, rx, "tdm", 2000)
    target = Target(radar.t_obs + radar.k_space)
    pf = ParticleFilter(radar.t_obs + radar.k_space, rx.channels, n_particles=100)

    sim = Simulator(k, radar, target, pf)
    sim.target_estimate(0)

