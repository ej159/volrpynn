import numpy as np
import volrpynn as v

def test_relu_derived():
    out = np.array([[-1], [0], [1]])
    error = np.array([0.2])
    weights = np.array([[1], [1], [1]])
    w, e = v.relu_derived(out, weights, error)
    expected_weights = np.array([[0], [0], [0.2]])
    expected_errors = np.array([[0.2], [0.2], [0.2]])
    assert np.allclose(w, expected_weights)
    assert np.allclose(e, expected_errors)

def test_relu_derived_dimension_augment():
    out = np.ones((2, 1))
    error = np.ones(1)
    weights = np.ones((2, 1))
    w, e = v.relu_derived(out, weights, error)
    assert w.shape == (2, 1)
    assert e.shape == (2,)

def test_relu_derived_dimension_reduction():
    out = np.ones((3,1))
    weights = np.ones((2, 3))
    error = np.ones(3)
    w, e = v.relu_derived(out, weights, error)
    assert w.shape == (3, 1)
    assert e.shape == (2,)

def test_relu_leaky_derived():
    xs = np.array([[-1], [0], [1]])
    error = np.array([0.2])
    weights = np.array([[1], [1], [1]])
    w, e = v.relu_leaky_derived(xs, weights, error)
    expected_weights = np.array([[-0.002], [0], [0.2]])
    expected_errors = np.array([0.2, 0.2, 0.2])
    assert np.allclose(w, expected_weights)
    assert np.allclose(e, expected_errors)
