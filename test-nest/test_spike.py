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

def test_spike_count():
    rate = 1000
    p = pynn().Population(2, pynn().SpikeSourcePoisson(rate = rate))
    p.record('spikes')
    pynn().run(1000)
    trains = p.get_data('spikes').segments[0].spiketrains
    assert np.allclose(v.spike_count(trains), [1000, 1000], atol=100)

def test_spike_count_linear():
    p = pynn().Population(2, pynn().IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p.set(i_offset=12)
    p.record('spikes')
    pynn().run(50)
    trains = p.get_data('spikes').segments[0].spiketrains
    assert np.allclose(v.spike_count_linear(trains), [1, 1], atol=0.1)

def test_spike_rate():
    rate = 1000
    p = pynn().Population(2, pynn().SpikeSourcePoisson(rate = rate))
    p.record('spikes')
    pynn().run(1000)
    trains = p.get_data('spikes').segments[0].spiketrains
    assert np.allclose(v.spike_rate(1000)(trains), np.ones(1), atol=0.1)
