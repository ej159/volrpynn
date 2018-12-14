import pyNN.nest as pynn
import numpy as np
import volrpynn.nest as v
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup(rng_seeds_seed = 100)

def test_gradient_optimiser_train_simple():
    p1 = pynn.Population(1, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_softmax, weights = 1)
    model = v.Model(l)
    optimiser = v.GradientDescentOptimiser(0.1)
    xs = np.array([[1]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.5, 0.5]]))
    assert np.allclose(l.get_weights(), np.array([[1.025, 0.975]])) # No change
    xs = np.array([[1]])
    ys = np.array([[0, 1]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.881, 0.119]]), atol=0.01)
    assert np.allclose(l.get_weights(), np.array([[0.945, 0.98]]), atol=0.01) # Strengthen right
    xs = np.array([[1]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.269, 0.731]]), atol=0.01)
    assert np.allclose(l.get_weights(), np.array([[0.964, 0.936]]), atol=0.01) # Strengthen left

def test_gradient_optimiser_train():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_softmax, weights = 1)
    model = v.Model(l)
    optimiser = v.GradientDescentOptimiser(0.1)
    xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    ys = np.array([[1, 0], [1, 0], [1, 0]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.5, 0.5], [0.88, 0.12], [0.95, 0.04]]), atol=0.01)
    xs = np.array([[1, 0, 0]])
    ys = np.array([[0, 1]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.999, 0.0003]]), atol=0.01)

def test_gradient_optimiser_train_categorical():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_softmax, weights = 1)
    model = v.Model(l)
    optimiser = v.GradientDescentOptimiser(0.1)
    xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    ys = np.array([[1, 0], [1, 0], [1, 0]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.5, 0.5], [0.88, 0.12], [0.95, 0.05]]),
            atol=0.1)
    assert np.allclose(l.get_weights(), np.ones((3, 2)), atol = 0.1)
    xs = np.array([[1, 0, 0]])
    ys = np.array([[0, 1]])
    m, y = optimiser.train(model, xs, ys)
    assert np.allclose(y, np.array([[0.999, 0.0003]]), atol=0.01)
    assert np.allclose(l.get_weights(), np.array([[0.95, 1.05], [0.95, 1.05],
        [0.95, 1.05]]), atol= 0.1)

