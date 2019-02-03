import numpy as np
import volrpynn as v

alpha = 3.22500557
beta = 1.61295370014

def test_translate_current():
    t = v.LinearTranslation()
    xs = np.array([-1, 0, 1])
    actual = t.to_current(xs)
    expected = np.array([1, 6.5, 12])
    assert np.allclose(actual, expected)

def test_from_spikes():
    t = v.LinearTranslation()
    xs = np.array([1, 6.5, 12])
    actual = t.from_spikes(xs * alpha - beta)
    expected = np.array([0, 0.5, 1])
    assert np.allclose(actual, expected, atol=0.05)

def test_weight_translate():
    t = v.LinearTranslation();
    xs = np.array([-1, -0.1, 1, 20])
    actual = t.weights(xs, 3)
    expected = xs * (0.065 / 3)
    assert np.allclose(actual, expected)
