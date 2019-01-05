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
    l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_count_normalised, weights = 1)
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
    assert np.allclose(l.get_output(), np.array([[1, 0.95]]), atol=0.01)
    assert np.allclose(l.get_weights(), np.array([[0.9, 0.9]]), atol=0.01) # Weaken left
    xs = np.array([[1]])
    ys = np.array([[1, 0]])
    m, y = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(l.get_output(), np.array([[0.99, 1]]), atol=0.01)
    assert np.allclose(l.get_weights(), np.array([[0.9, 0.81]]), atol=0.01) # Weaken right

# def test_gradient_optimiser_train():
#     p1 = pynn.Population(3, pynn.IF_cond_exp())
#     p2 = pynn.Population(2, pynn.IF_cond_exp())
#     l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_count_normalised, weights = 1)
#     l2 = v.Decode(p2, decoder = v.spike_count_normalised)
#     model = v.Model(l, l2)
#     optimiser = v.GradientDescentOptimiser(0.1)
#     xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
#     ys = np.array([[1, 0], [1, 0], [1, 0]])
#     m, y = optimiser.train(model, xs, ys, v.SumSquared())
#     assert np.allclose(y, np.array([[1, 1], [1, 0.95], [1, 0.90]]), atol=0.01)
#     xs = np.array([[1, 0, 0]])
#     ys = np.array([[0, 1]])
#     m, y = optimiser.train(model, xs, ys, v.SumSquared())
#     assert np.allclose(y, np.array([[1, 0.87]]), atol=0.01)
#     y = model.predict(xs[0], 1000)
#     assert np.allclose(y, np.array([[1, 0.91]]), atol=0.01)
# 
# def test_gradient_optimiser_train_categorical():
#     p1 = pynn.Population(3, pynn.IF_cond_exp())
#     p2 = pynn.Population(2, pynn.IF_cond_exp())
#     l = v.Dense(p1, p2, v.relu_derived, decoder = v.spike_softmax, weights = 1)
#     l2 = v.Decode(p2, decoder = v.spike_softmax)
#     model = v.Model(l, l2)
#     optimiser = v.GradientDescentOptimiser(0.1)
#     xs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
#     ys = np.array([[1, 0], [1, 0], [1, 0]])
#     m, y = optimiser.train(model, xs, ys, v.CrossEntropy())
#     assert np.allclose(y, np.array([[0.5, 0.5], [0.99, 0.00], [0.99, 0.00]]),
#             atol=0.1)
#     assert np.allclose(l.get_weights(), np.ones((3, 2)), atol = 0.1)
#     xs = np.array([[1, 0, 0]])
#     ys = np.array([[0, 1]])
#     m, y = optimiser.train(model, xs, ys, v.SumSquared())
#     assert np.allclose(y, np.array([[0.999, 0.0003]]), atol=0.01)
#     assert np.allclose(l.get_weights(), np.array([[0.95, 1.05], [0.95, 1.05],
#         [0.95, 1.05]]), atol= 0.1)
# 
