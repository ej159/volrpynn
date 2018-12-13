import volrpynn.nest as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup()

def test_nest_dense_create():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.relu_derived)
    expected_weights = np.ones((12, 10))
    assert np.array_equal(d.projection.get('weight', format='array'),
            expected_weights)

def test_nest_dense_spikes_shape():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 10))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.relu_derived)
    pynn.run(1000)
    d.store_spikes()
    assert d.spikes.shape == (12,)

def test_nest_dense_projection():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 10))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.relu_derived)
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    assert len(spiketrains) == 10
    avg_len = np.array(list(map(len, spiketrains))).mean()
    # Should have equal activation
    for train in spiketrains:
        assert abs(len(train) - avg_len) <= 1

def test_nest_dense_reduced_weight_fire():
    p1 = pynn.Population(2, pynn.SpikeSourcePoisson(rate = 1))
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.relu_derived, weights = np.array([[1], [0]]))
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    assert len(spiketrains) == 1
    assert spiketrains[0].size > 0

def test_nest_dense_chain():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 100))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p3 = pynn.Population(2, pynn.IF_cond_exp())
    p3.record('spikes')
    d1 = v.Dense(p1, p2, v.relu_derived)
    d2 = v.Dense(p2, p3, v.relu_derived)
    pynn.run(1000)
    assert len(p3.get_data().segments[-1].spiketrains) > 0

def test_nest_dense_restore():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.relu_derived, weights = 2)
    d.set_weights(-1)
    assert np.array_equal(d.projection.get('weight', format='array'),
             np.ones((12, 10)) * -1)
    d.projection.set(weight = 1) # Simulate reset()
    assert np.array_equal(d.projection.get('weight', format='array'),
            np.ones((12, 10)))
    d.restore_weights()
    assert np.array_equal(d.projection.get('weight', format='array'),
            np.ones((12, 10)) * -1)

def test_nest_dense_backprop():
    p1 = pynn.Population(4, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived, weights = 1)
    old_weights = l.get_weights()
    l.spikes = np.ones((4, 1)) # Mock spikes
    _, errors = l.backward(np.array([1, 1]), np.array([0, 1]), lambda w, g: w - g)
    expected_errors = np.ones((4,))
    assert np.allclose(errors, expected_errors)
    expected_weights = np.tile([1, 0], (4, 1))
    assert np.allclose(l.get_weights(), expected_weights)
