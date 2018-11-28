import numpy as np
import volrpynn as v

def test_relu_derived():
    xs = np.array([[-1], [0], [1]])
    error = np.array([0.2, 0.3, 0.5])
    weights = np.array([[1], [1], [1]])
    w, e = v.relu_derived(xs, error, weights)
    expected_weights = np.array([[0], [0], [0.5]])
    expected_errors = np.array([[0.2], [0.3], [0.5]])
    assert np.allclose(w, expected_weights)
    assert np.allclose(e, expected_errors)

def test_relu_leaky_derived():
    xs = np.array([[-1], [0], [1]])
    error = np.array([0.2, 0.3, 0.5])
    weights = np.array([[1], [1], [1]])
    w, e = v.relu_leaky_derived(xs, error, weights)
    expected_weights = np.array([[-0.002], [0], [0.5]])
    expected_errors = np.array([[0.2], [0.3], [0.5]])
    assert np.allclose(w, expected_weights)
    assert np.allclose(e, expected_errors)
