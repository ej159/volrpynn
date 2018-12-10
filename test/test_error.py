import numpy as np
import volrpynn as v

def test_error_SSE_single():
    xs = 10
    ys = 12
    assert v.sum_squared_error(xs, ys) == 4

def test_error_SSE():
    xs = np.zeros((4, 1))
    ys = np.repeat(2, 4)
    expected = np.repeat(4, 10).reshape(-1, 1)
    out = v.sum_squared_error(xs, ys)
    assert np.allclose(out, np.array([4, 4, 4, 4]))


