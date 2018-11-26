import numpy as np
import volrpynn as v

def test_error_SSE():
    l = v.SumSquaredError()
    xs = np.ones(10)
    ys = np.zeros(10)
    assert l.error(xs, ys) == 10

def test_error_SSE_derived():
    l = v.SumSquaredError()
    xs = np.ones(10)
    ys = np.zeros(10)
    assert np.allclose(l.error_derived(xs, ys), xs - ys)
    
