import numpy as np
import volrpynn as v

def test_error_SSE_single():
    xs = np.array(10)
    ys = np.array(12)
    assert v.SumSquared()(xs, ys) == 2

def test_error_SSE():
    xs = np.zeros((4))
    ys = np.repeat(2, 4)
    expected = np.repeat(2, 4).sum()
    out = v.SumSquared()(xs, ys)
    assert np.allclose(out, expected)


