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
    l = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count_normalised, weights = 1)
    model = v.Model(l)
    optimiser = v.GradientDescentOptimiser(0.1)
    xs = np.array([[1]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(l.get_output(), np.array([[1, 1]]))
    assert np.allclose(l.get_weights(), np.array([[1, 0.9]]))
    xs = np.array([[1]])
    ys = np.array([[0, 1]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(l.get_output(), np.array([[1, 0.95]]), atol=0.02)
    assert np.allclose(l.get_weights(), np.array([[0.9, 0.9]]), atol=0.02) # Weaken left
    xs = np.array([[1]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(l.get_output(), np.array([[0.99, 1]]), atol=0.02)
    assert np.allclose(l.get_weights(), np.array([[0.9, 0.82]]), atol=0.02) # Weaken right

def test_gradient_optimiser_train_left():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count_normalised, weights = 1)
    l2 = v.Decode(p2, decoder = v.spike_count_normalised)
    model = v.Model(l, l2)
    optimiser = v.GradientDescentOptimiser(1)
    xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    ys = np.array([[1, 0], [1, 0], [1, 0]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([0.5, 0, 0]), atol=0.01)
    xs = np.array([[1, 0, 0]])
    ys = np.array([[0, 1]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([[1]]), atol=0.01)
    y = model.predict(xs[0], 1000)
    assert np.allclose(y, np.array([[0]]), atol=0.01)

def test_gradient_optimiser_train_right():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), decoder = v.spike_count_normalised, weights = 1)
    l2 = v.Decode(p2, decoder = v.spike_count_normalised)
    model = v.Model(l, l2)
    optimiser = v.GradientDescentOptimiser(1)
    xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    ys = np.array([[0, 1], [0, 1], [0, 1]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([0.5, 0, 0]), atol=0.01)
    xs = np.array([[1, 0, 0]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([[1]]), atol=0.01)
    y = model.predict(xs[0], 1000)
    assert np.allclose(y, np.array([[0]]), atol=0.01)

