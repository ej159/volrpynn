import numpy as np
import volrpynn as v

def test_error_SSE():
    xs = np.zeros((10,1))
    ys = np.repeat(2, 10)
    expected = np.repeat(4, 10).reshape(-1, 1)
    out = v.sum_squared_error(xs, ys)
    assert out.shape == (10, 1)
    assert np.array_equal(out, expected)

def test_error_SSE_layer():
    xs = np.array([1, 0])
    ys = np.array([1, 0])
    assert np.allclose(v.sum_squared_error(xs, ys), np.array([0, 0]))
