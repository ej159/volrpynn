import numpy as np
import volrpynn as v

def test_relu_derived():
    out = np.array([[-1], [0], [1]])
    actual = v.relu_derived(out)
    expected = np.array([[0], [0], [1]])
    assert np.allclose(actual, expected)

