"""
    This is a test script for function tx_tdm in Transmitter class.
    Running this script runs all tests in directory.
"""
import numpy as np
import pytest
from src.radar import Radar
from src.receiver import Receiver
from src.transmitter import Transmitter
from src.target import Target

# Data types and shape, amount of signals must correspond to m_channels
# Find a way to check time divisions

@pytest.fixture
def radar():
    tx = Transmitter()
    rx = Receiver()
    radar = Radar(tx, rx, "tdm", 2000)
    return radar

@pytest.fixture
def states(radar):
    target = Target(radar.t_obs + radar.k_space)
    states = target.generate_states(1, 'linear_away')
    return states

def test_transmitters(radar, states):
    t_vec = radar.create_time_vector()
    tau = radar.time_delay(states[0], t_vec)
    delay = t_vec - tau
    tx_sig = radar.transmitter.tx_tdm(delay, radar.t_rx)
    
    assert tx_sig.shape == (radar.m_channels, radar.samples_per_obs)
    assert tx_sig.dtype == np.complex128

def test_transmission_times(radar, states):
    tol = 1e-9
    t_vec = radar.create_time_vector()
    tau = radar.time_delay(states[0], t_vec)
    delay = t_vec - tau
    tx_sig = radar.transmitter.tx_tdm(delay, radar.t_rx)

    # Looping through all transmissions
    for tx_m in range(radar.m_channels):
        # Get indexations in signal, where it is non-zero
        idx = np.nonzero(tx_sig[tx_m])
        # Get times for transmissions
        times = t_vec[idx] - tau[idx]
        # Calculate desired time of transmission
        truth = tx_m * (radar.t_rx + radar.transmitter.t_chirp)
        
        assert np.abs(truth - times[0]) < tol
        assert np.abs(truth + radar.transmitter.t_chirp - times[-1]) < tol


pytest.main()