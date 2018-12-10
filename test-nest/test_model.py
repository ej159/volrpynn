import volrpynn as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup()

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
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    m = v.Model(pynn, l)
    actual_weights = m.input_projection.get('weight', format='array')
    expected_weights = np.array([[1, np.NaN], [np.NaN, 1]])
    assert np.allclose(actual_weights, expected_weights, equal_nan=True)

def test_nest_create_input_populations():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    m = v.Model(pynn, l)
    assert len(m.input_populations) == 2
    m.set_input([1, 0.2])
    assert m.input_populations[0].get('rate') == 1.0
    assert m.input_populations[1].get('rate') == 0.2

def test_nest_model_predict_active():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    d = pynn.random.RandomDistribution('normal', mu=1, sigma=0.1)
    l = v.Dense(pynn, p1, p2, v.relu_derived, weights = d)
    m = v.Model(pynn, l)
    out = m.predict(np.array([10, 0]), 1000)
    assert len(out) == 2
    assert len(out[0]) > 0
    assert len(out[1]) > 0 # Should be above 0 despite 0 input, due to all-to-all

def test_nest_model_predict_inactive():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    m = v.Model(pynn, l)
    out = m.predict(np.array([0, 0]), 1000)
    assert len(out) == 2
    assert len(out[0]) == 0
    assert len(out[1]) == 0 # Should be above 0 despite 0 input, due to all-to-all

def test_nest_model_backwards():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(3, pynn.IF_cond_exp())
    l1 = v.Dense(pynn, p1, p2, v.relu_derived)
    m = v.Model(pynn, l1)
    xs = np.array([1, 0])
    ys = np.array([1, 0, 0])
    spikes = m.predict(xs, 500)
    out = m.backward(ys, lambda w, g: w - g)
    expected_weights = np.array([[2/3.0, 1, 1], [2/3.0, 1, 1]])
    assert np.allclose(l1.get_weights(), expected_weights)
    assert np.allclose(out, np.array([0.2222222, 0.2222222]))

def test_nest_model_backwards_reset():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l1 = v.Dense(pynn, p1, p2, v.relu_derived)
    m = v.Model(pynn, l1)
    xs1 = np.array([1, 1])
    ys1 = np.array([0, 1])
    xs2 = np.array([1, 1])
    ys2 = np.array([0, 1])
    # First pass
    target1 = m.predict(xs1, 1000)
    out = m.backward(ys1, lambda w, g: w - g)
    assert np.allclose(out, np.array([0.25, 0.25]))
    expected_weights = np.array([[1, 0.5], [1, 0.5]])
    assert np.allclose(l1.get_weights(), expected_weights)
    # Second pass
    target2 = m.predict(xs2, 1000)
    out = m.backward(ys2, lambda w, g: w - g)
    assert np.allclose(out, np.array([0, 0]), atol=0.001)
    expected_weights = np.array([[1, 0.5], [1, 0.5]])
    assert np.allclose(l1.get_weights(), expected_weights)
    assert np.allclose(v.spike_argmax(target1), np.array([1, 0]))
    assert np.allclose(v.spike_argmax(target2), np.array([1, 0]))
