import volrpynn.nest as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup(rng_seeds_seed = 100)

### Basic PyNN tests

def test_nest_population():
    p = pynn.Population(12, pynn.IF_cond_exp())
    assert p.size == 12

def test_nest_projection():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    assert len(c.get('weight', format='list')) == 4

def test_nest_projection_gaussian():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    c.set(weight=pynn.random.RandomDistribution('normal', mu=0.5, sigma=0.1))
    weights = c.get('weight', format='array')
    assert len(weights[weights == 0]) < 1

### Model tests

def test_nest_input_projection():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived)
    m = v.Model(l)
    assert m.input_projection[0].weight == 1
    assert m.input_projection[1].weight == 1
    m.predict([1, 1], 2000)
    spiketrains = l.spikes
    assert abs(len(spiketrains[0]) - len(spiketrains[1])) <= 20
 
def test_nest_create_input_populations():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived)
    m = v.Model(l)
    assert len(m.input_populations) == 2
    m.set_input([1, 0.2])
    assert m.input_populations[0].get('rate') == 1.0
    assert m.input_populations[1].get('rate') == 0.2

def test_nest_model_predict_active():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    d = pynn.random.RandomDistribution('normal', mu=1, sigma=0.1)
    l = v.Dense(p1, p2, v.relu_derived, weights = d, decoder = v.spike_count)
    m = v.Model(l)
    out = m.predict(np.array([1, 1]), 1000)
    assert len(out) == 2
    assert abs(out[0] - out[1]) <= 10 # Must be approx same spikes

def test_nest_model_predict_inactive():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived)
    m = v.Model(l)
    out = m.predict(np.array([0, 0]), 1000)
    assert len(out) == 2
    assert out.sum() == 0

def test_nest_model_backwards():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(3, pynn.IF_cond_exp())
    l1 = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_count_normalised)
    m = v.Model(l1)
    xs = np.array([1, 1])
    spikes = m.predict(xs, 1000)
    m.backward(spikes, [0, 1, 1], lambda w, g: w - g) # no learning rate
    expected_weights = np.array([[1, 0, 0], [1, 0, 0]])
    assert np.allclose(l1.get_weights(), expected_weights, atol=0.1)

def test_nest_model_backwards_reset():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l1 = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_argmax)
    m = v.Model(l1)
    xs1 = np.array([1, 1])
    ys1 = np.array([0, 1])
    xs2 = np.array([1, 1])
    ys2 = np.array([0, 1])
    # First pass
    target1 = m.predict(xs1, 2000)
    m.backward(ys1, [0, 1], lambda w, g: w - g)
    expected_weights = np.array([[1, 1], [1, 0]])
    assert np.allclose(l1.get_weights(), expected_weights)
    # Second pass
    target2 = m.predict(xs2, 2000)
    m.backward(ys2, [0, 1], lambda w, g: w - g)
    expected_weights = np.array([[1, 0], [1, 0]])
    assert np.allclose(l1.get_weights(), expected_weights)
