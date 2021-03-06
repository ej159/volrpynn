import json
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
    p1 = pynn.Population(2, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p2 = pynn.Population(2, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    l = v.Dense(p1, p2, v.ReLU(), weights = 1)
    m = v.Model(l)
    t = v.LinearTranslation()
    assert np.allclose(l.get_weights(), np.ones((2, 2)))
    assert m.input_projection[0].weight == t.weights(1, 1)
    assert m.input_projection[1].weight == t.weights(1, 1)
    m.predict([1, 1], 50)
    spiketrains = l.output
    assert abs(len(spiketrains[0]) - len(spiketrains[1])) <= 20
 
def test_nest_create_input_populations():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU())
    m = v.Model(l)
    assert len(m.input_populations) == 2
    inp = np.array([1, 0.2])
    m.set_input(inp)
    assert m.input_populations[0].get('i_offset') == 1
    assert m.input_populations[1].get('i_offset') == 0.2

def test_nest_model_predict_active():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count, weights = 1)
    m = v.Model(l)
    out = m.predict(np.array([1, 1]), 1000)
    assert len(out) == 2
    assert abs(out[0] - out[1]) <= 10 # Must be approx same spikes

def test_nest_model_predict_inactive():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count)
    m = v.Model(l)
    out = m.predict(np.array([0, 0]), 10)
    assert len(out) == 2
    assert out.sum() < 10

def test_nest_dense_normalisation():
    p1 = pynn.Population(12, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p2 = pynn.Population(10, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    l = v.Dense(p1, p2, v.ReLU(), weights = 1)
    m = v.Model(l)
    out = m.predict(np.ones(12) * 12, 50)
    assert np.allclose(np.ones(10), out, atol=0.1)

def test_nest_model_linear_scaling():
    p1 = pynn.Population(2, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p2 = pynn.Population(2, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    l1 = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count, weights = 1)
    m = v.Model(l1)
    xs = np.array([12, 12])
    out = m.predict(xs, 50)
    assert np.allclose([38, 38], out)

#def test_nest_dense_forward():
#    p1 = pynn.Population(4, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
#    p2 = pynn.Population(3, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
#    weights = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
#    bias = np.array([1, 2, 3])
#    l = v.Dense(p1, p2, v.UnitActivation(), weights=weights.T, biases=bias)
#    m = v.Model(l)
#    t = v.LinearTranslation()
#    xs = t.to_current(np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
#    ys = np.array([m.predict(x, 50) for x in xs])
#    assert False

def test_nest_model_backwards():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(3, pynn.IF_cond_exp())
    l1 = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count_linear, weights = 1)
    m = v.Model(l1)
    xs = np.array([12, 12])
    spikes = m.predict(xs, 50)
    def l(w, g, b, bg):
        return w - g, b - bg

    m.backward([0, 1, 1], l) # no learning rate
    expected_weights = np.array([[1, 0, 0], [1, 0, 0]])
    assert np.allclose(l1.get_weights(), expected_weights, atol=0.2)

def test_nest_model_backwards_reset():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l1 = v.Dense(p1, p2, v.ReLU(), decoder=v.spike_count_normalised,weights = 1)
    m = v.Model(l1)
    xs1 = np.array([10, 10])
    ys1 = np.array([0, 10])
    xs2 = np.array([10, 10])
    ys2 = np.array([0, 10])
    # First pass
    target1 = m.predict(xs1, 50)
    m.backward([0, 1], lambda w, g, b, bg: (w - g, b - bg))
    expected_weights = np.array([[1, 0], [1, 0]])
    assert np.allclose(l1.get_weights(), expected_weights)
    # Second pass
    target2 = m.predict(xs2, 50)
    m.backward([1, 0], lambda w, g, b, bg: (w - g, b - bg))
    expected_weights = np.array([[-1, 0], [-1, 0]])
    assert np.allclose(l1.get_weights(), expected_weights)

def test_nest_model_spike_normalisation():
    p1 = pynn.Population(2, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p2 = pynn.Population(4, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    p3 = pynn.Population(4, pynn.IF_cond_exp(**v.DEFAULT_NEURON_PARAMETERS))
    l1 = v.Dense(p1, p2, decoder=v.spike_rate(50), weights=1)
    l2 = v.Dense(p2, p3, decoder=v.spike_rate(50), weights=1)
    m = v.Model(l1, l2)
    m.predict([12, 12], 50)
    for rate in np.concatenate((l1.get_output(), l2.get_output())):
        assert rate > 0
        assert rate < 1
