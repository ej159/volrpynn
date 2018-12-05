import pyNN.nest as pynn
import numpy as np
import volrpynn as v
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup()

def test_gradient_descent_error():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    l.set_weights(np.array([[20], [0]]))
    model = v.Model(pynn, l)
    optimiser = v.GradientDescentOptimiser(v.spike_argmax, 0.1)
    xs = np.array([[1, 0], [0, 1], [0, 0]])
    ys = np.array([[1], [0], [0]])
    errors = optimiser.test(model, xs, ys, v.sum_squared_error)
    assert np.allclose(errors, np.array([[0], [0], [0]]))
    
def test_gradient_descent_optimiser():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    model = v.Model(pynn, l)
    optimiser = v.GradientDescentOptimiser(v.spike_softmax, 0.1)
    xs = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    ys = np.array([[1, 0], [0.5, 0.5], [0.5, 0.5]])
    m, y = optimiser.train(model, xs, ys, v.sum_squared_error)
    assert np.allclose(y, np.array([[0.25, 0.25], [0, 0], [0, 0]]))

def test_gradient_descent_optimiser_categorical():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(5, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, v.relu_derived)
    model = v.Model(pynn, l)
    optimiser = v.GradientDescentOptimiser(v.spike_argmax, 0.1)
    error = v.sum_squared_error
    xs = np.array([[8, 2], [3, 4]])
    ys = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
    m, y = optimiser.train(model, xs, ys, error)

