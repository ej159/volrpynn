import volrpynn as v
import pyNN.nest as pynn
import numpy as np

pynn.setup()

def test_nest_dense_create():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(pynn, p1, p2, v.relu_derived)
    expected_weights = np.ones((12, 10))
    assert np.array_equal(d.projection.get('weight', format='array'),
            expected_weights)

def test_nest_dense_projection():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 100))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(pynn, p1, p2, v.relu_derived)
    pynn.run(1000)
    assert len(p2.get_data().segments[-1].spiketrains) > 0

def test_nest_dense_chain():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 100))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p3 = pynn.Population(2, pynn.IF_cond_exp())
    p3.record('spikes')
    d1 = v.Dense(pynn, p1, p2, v.relu_derived)
    d2 = v.Dense(pynn, p2, p3, v.relu_derived)
    pynn.run(1000)
    assert len(p3.get_data().segments[-1].spiketrains) > 0

def test_nest_dense_restore():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(pynn, p1, p2, v.relu_derived, weights = 2)
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
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived, weights = 2)
    old_weights = l.get_weights()
    l.spikes = np.zeros((10, 1)) # Mock spikes
    errors = l.backward(np.ones((10)), lambda w, g: w - g)
    expected_errors = np.zeros((12,)) + 0.2
    assert np.allclose(errors, expected_errors)
    expected_weights = old_weights - 0.2
    assert np.allclose(l.get_weights(), expected_weights)
