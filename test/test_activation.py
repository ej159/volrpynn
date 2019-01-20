import numpy as np
import volrpynn as v

def test_relu_derived():
    out = np.array([[-1], [0], [1]])
    actual = v.ReLU().prime(out)
    expected = np.array([[0], [0], [1]])
    assert np.allclose(actual, expected)

def test_sigmoid_derived():
    out = np.array([[0], [1], [2]])
    actual = v.Sigmoid().prime(out)
    expected = np.array([[0.25], [0.19661193], [0.10499359]])
    assert np.allclose(actual, expected)
