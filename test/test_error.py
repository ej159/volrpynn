import numpy as np
import volrpynn as v

def test_error_SSE():
    xs = np.zeros(10)
    ys = np.repeat(2, 10)
    expected = np.repeat(4, 10)
    assert np.allclose(v.sum_squared_error(xs, ys), expected)

