import volrpynn as v
from neo.core import SpikeTrain
from quantities import s
import numpy as np

def test_softmax():
    train1 = SpikeTrain([3, 4] * s, t_stop = 10)
    train2 = SpikeTrain([5, 6] * s, t_stop = 10)
    trains = np.array([train1, train2])
    expected = np.array([0.5, 0.5])
    assert np.allclose(v.spike_softmax(trains), expected)

def test_softmax_deriv():
    train1 = SpikeTrain([3, 4] * s, t_stop = 10)
    train2 = SpikeTrain([5, 6] * s, t_stop = 10)
    trains = np.array([train1, train2])
    ret = v.spike_softmax_deriv(trains)
    assert callable(ret)
    expected = np.array([-0.25, 0.25])
    print(ret(1))
    assert np.allclose(ret(1), expected)
