from signal_model import Signal
from particle_filter import ParticleFilter



sig = Signal()
pf = ParticleFilter(10)
obs_ng = sig.observe_no_gain()
obs = sig.observe()

pf.init_particles_uniform([2000, 2000])

pf.plot_particles()
