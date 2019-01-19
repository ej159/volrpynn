import volrpynn.nest as v
from volrpynn.util import get_pynn as pynn
import numpy as np

def test_spike_argmax():
    p = pynn().Population(10, pynn().SpikeSourcePoisson(rate = 10.0))
    p.record('spikes')
    pynn().run(1000)
    trains = p.get_data('spikes').segments[0].spiketrains
    expected = np.zeros(10)
    expected[np.argmax([len(x) for x in trains])] = 1
    assert np.allclose(v.spike_argmax(trains), expected)

def test_spike_rate():
    rate = 1000
    p = pynn().Population(2, pynn().SpikeSourcePoisson(rate = rate))
    p.record('spikes')
    pynn().run(1000)
    trains = p.get_data('spikes').segments[0].spiketrains
    assert np.allclose(v.spike_rate(1000)(trains), np.ones(1), atol=0.1)
