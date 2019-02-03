import pyNN.nest as pynn
import numpy as np
import volrpynn.nest as v
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup(rng_seeds_seed = 100)

#def test_gradient_optimiser_train_simple():
#    p1 = pynn.Population(1, pynn.IF_cond_exp())
#    p2 = pynn.Population(2, pynn.IF_cond_exp())
#    l = v.Dense(p1, p2, v.ReLU(), weights = 1)
#    model = v.Model(l)
#    optimiser = v.GradientDescentOptimiser(0.1)
#    xs = np.array([[12]])
#    ys = np.array([[1, 0]])
#    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
#    assert np.allclose(l.get_output(), np.array([[1, 1]]), atol=0.2)
#    assert np.allclose(l.get_weights(), np.array([[1, 0.9]]), atol=0.04)
#    xs = np.array([[12]])
#    ys = np.array([[0, 1]])
#    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
#    assert np.allclose(l.get_output(), np.array([[0.9, 0.95]]), atol=0.1)
#    assert np.allclose(l.get_weights(), np.array([[0.9, 0.9]]), atol=0.1) # Weaken left
#    xs = np.array([[12]])
#    ys = np.array([[1, 0]])
#    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
#    assert np.allclose(l.get_output(), np.array([[0.99, 1]]), atol=0.1)
#    assert np.allclose(l.get_weights(), np.array([[0.9, 0.82]]), atol=0.1) # Weaken right

def test_gradient_optimiser_train_left():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), weights = 1)
    l2 = v.Decode(p2)
    model = v.Model(l, l2)
    optimiser = v.GradientDescentOptimiser(1)
    xs = np.array([[12, 0, 0], [12, 0, 0], [12, 0, 0]])
    ys = np.array([[1, 0], [1, 0], [1, 0]])
    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([0.75]), atol=0.01)
    xs = np.array([[12, 0, 0]])
    ys = np.array([[0, 1]])
    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([[1]]), atol=0.01)
    y = model.predict(xs[0], 50)
    assert np.allclose(y, np.array([[1, 0]]), atol=0.03)

def test_gradient_optimiser_train_right():
    p1 = pynn.Population(3, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.ReLU(), weights = 1)
    l2 = v.Decode(p2)
    model = v.Model(l, l2)
    optimiser = v.GradientDescentOptimiser(1)
    xs = np.array([[12, 0, 0], [12, 0, 0], [12, 0, 0]])
    ys = np.array([[0, 1], [0, 1], [0, 1]])
    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([0.75]), atol=0.01)
    xs = np.array([[12, 0, 0]])
    ys = np.array([[1, 0]])
    m, y, _ = optimiser.train(model, xs, ys, v.SumSquared())
    assert np.allclose(y, np.array([[1]]), atol=0.01)
    y = model.predict(xs[0], 50)
    assert np.allclose(y, np.array([[0, 1]]), atol=0.1)

#def test_gradient_optimiser_error():
#    p1 = pynn.Population(4, pynn.IF_cond_exp())
#    p2 = pynn.Population(3, pynn.IF_cond_exp())
#    weights = np.array([[1.0,  2.0,  3.0,  4.0],
#           [5.0,  6.0,  7.0,  8.0],
#           [9.0, 10.0, 11.0, 12.0]])
#    biases = np.array([1.0, 2.0, 3.0])
#    l = v.Dense(p1, p2, v.UnitActivation(), weights = 1)
#    model = v.Model(l)
#    optimiser = v.GradientDescentOptimiser(0.1)
#
#    class UnitError(v.ErrorFunction):
#        def __call__(self, output, labels):
#            return output
#
#        def prime(self, output, labels):
#            return output
#
#    xs = np.array([[1.0, 2.0, 3.0, 4.0],
#        [2.0, 3.0, 4.0, 5.0],
#        [3.0, 4.0, 5.0, 6.0]]) 
#    ys = np.matmul(xs, weights.T)
#    _, _, errors = optimiser.train(model, xs, ys, UnitError())
#    expected_error = np.array([[1408.0, 1624.0, 1840.0, 2056.0],
#          [1926.0, 2220.0, 2514.0, 2808.0],
#          [2444.0, 2816.0, 3188.0, 3560.0]])
#    print(errors[0][0])
#    print(expected_error[0])
#    assert np.allclose(errors[0], expected_error)
