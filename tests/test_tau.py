import numpy as np
import pytest
from src.radar import Radar
from src.receiver import Receiver
from src.transmitter import Transmitter

@pytest.fixture
def radar():
    tx = Transmitter()
    rx = Receiver()
    radar = Radar(tx, rx, "tdm", 2000)
    return radar

def test_output_scalar_dtype(radar):
    theta = np.array(np.array([[1], [1], [1], [1]]))   
    tau = radar.time_delay(theta, 1, 0)
    
    assert type(tau) == np.ndarray
    assert type(tau[0]) == np.float64

def test_output_vector_dtype(radar):
    theta = np.array(np.array([[1], [1], [1], [1]]))   
    tau = radar.time_delay(theta, 1, 0)
    
    assert type(tau) == np.ndarray
    assert type(tau[0]) == np.float64

