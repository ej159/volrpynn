import numpy as np
import volrpynn as v

def test_relu_derived():
    out = np.array([[-1], [0], [1]])
    error = np.array([0.2])
    actual_errors = v.relu_derived(out, error)
    expected_errors = np.array([[0], [0], [0.2]])
    assert np.allclose(actual_errors, expected_errors)

def test_relu_derived_dimension_augment():
    out = np.ones((2, 1))
    error = np.ones(1)
    g = v.relu_derived(out, error)
    assert g.shape == (2,1)

def test_relu_derived_dimension_reduction():
    out = np.ones((3,1))
    error = np.ones(3)
    g = v.relu_derived(out, error)
    assert g.shape == (3, 1)

