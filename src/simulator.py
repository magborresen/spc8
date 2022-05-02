import numpy as np
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
        _, rx_sig = self.radar.observation(k_obs, target_state)

        for particle in range(self.particle_filter.n_particles):
            s_sig_i, x_k_i = self.radar.observation(k_obs,
                                                    self.particle_filter.theta_est[particle],
                                                    add_noise=False)
            self.particle_filter.update_particle(particle, s_sig_i, rx_sig)
            self.particle_filter.update_weight(particle,
                                               rx_sig,
                                               x_k_i,
                                               self.radar.receiver.sigma_noise,
                                               self.radar.n_channels,
                                               self.radar.samples_per_obs)


if __name__ == '__main__':
    k = 10
    tx = Transmitter()
    rx = Receiver()

    radar = Radar(tx, rx, "tdm", 2000)
    target = Target(radar.t_obs + radar.k_space)
    pf = ParticleFilter(radar.t_obs + radar.k_space, n_particles=10)

    sim = Simulator(k, radar, target, pf)
    sim.target_estimate(0)