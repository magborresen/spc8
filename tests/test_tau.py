"""
    This is a test script for function time_delay in Radar class.
    Running this script runs all tests in directory.
"""
import numpy as np
import pytest
from src.radar import Radar
from src.receiver import Receiver
from src.transmitter import Transmitter
from src.target import Target

@pytest.fixture
def radar():
    tx = Transmitter()
    rx = Receiver()
    radar = Radar(tx, rx, "tdm", 2000)
    return radar

@pytest.fixture
def states(radar):
    k_obs = 2
    target = Target(radar.t_obs + radar.k_space)
    states = target.generate_states(k_obs, 'linear_away')
    return states

def test_values(radar, states):
    """
    This test function will compare states from Target object with calculated
    distance, using time-delay. Two states are used, state 0 and 1, for a target
    that is moving away from origin, starting in (1000, 1000). Since the movement
    is completely linear, time-delay is used to calculate where the target should
    be for state 1 (only using state 0 in calculations), and comparing distances.
    """
    # True distance for state 0
    true_0 = np.linalg.norm((states[0][0], states[0][1]))
    # Distance from tau for state 0
    tau_0 = radar.time_delay(states[0], 0, 0)
    dist_0 = tau_0 * 300e6 / 2
    
    # True distance for state 1
    true_1 = np.linalg.norm((states[1][0], states[1][1]))
    # Very last moment for state 0 = state 1
    tau_1 = radar.time_delay(states[0], 0, radar.t_obs + radar.k_space)
    dist_1 = tau_1 * 300e6 / 2
    
    assert dist_0 == true_0
    assert dist_1 == true_1

def test_types(radar, states):
    """
    This test function will test output types, using a vector in calculations.
    """
    t_vec = np.linspace(0, radar.t_obs, radar.n_channels)
    tau = radar.time_delay(states[0], 0, t_vec)
    assert tau.size == radar.n_channels
    assert tau.dtype == np.float64
    
retcode = pytest.main()