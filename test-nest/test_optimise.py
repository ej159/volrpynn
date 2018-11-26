import pyNN.nest as pynn
import numpy as np
import volrpynn as v

pynn.setup()

def test_gradient_descent_optimiser():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2)
    model = v.Model(pynn, l)
    optimiser = v.GradientDescentOptimiser(v.spike_softmax, 0.1)
    error = v.SumSquaredError().error
    xs = np.array([[1, 0], [0.5, 1], [0, 0]])
    ys = np.array([1, 0, 0])
    m, e = optimiser.train(model, xs, ys, error, v.relu_derived)
    assert np.allclose(e, np.array([0, 1, 1]))
