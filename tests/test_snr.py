import numpy as np
import pytest
from src.receiver import Receiver

@pytest.fixture
def arb_sig():
    samples = 100
    t = np.linspace(0, 2*np.pi, samples)
    return [np.cos(t) + 1j * np.sin(t)]

@pytest.fixture
def noise(arb_sig):
    rx = Receiver(snr=0)
    return rx.get_noise(arb_sig)

def test_output_length(noise, arb_sig):
    assert np.array(arb_sig).size == np.array(noise).size
    
def test_output_dtype(noise, arb_sig):
    assert arb_sig[0][0].dtype == noise[0][0].dtype
    
def test_snr_value(noise, arb_sig):
    tol = 0.1
    res_dB = np.log10(np.var(arb_sig) / np.var(noise))
    assert abs(res_dB) < tol