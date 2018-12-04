import volrpynn as v
from neo.core import SpikeTrain
from quantities import s
import numpy as np

def test_softmax_spike():
    train1 = SpikeTrain([3, 4] * s, t_stop = 10)
    train2 = SpikeTrain([5, 6] * s, t_stop = 10)
    trains = np.array([train1, train2])
    expected = np.array([0.5, 0.5])
    assert np.allclose(v.spike_softmax(trains), expected)

def test_spike_argmax_randomised():
    train1 = SpikeTrain([3, 4] * s, t_stop = 10)
    train2 = SpikeTrain([5, 6] * s, t_stop = 10)
    trains = np.array([train1, train2])
    out = v.spike_argmax(trains)
    assert out.size == len(trains)
    assert out.sum() == 1

def test_spike_argmax_indexed():
    train1 = SpikeTrain([3, 4] * s, t_stop = 10)
    train2 = SpikeTrain([5, 6] * s, t_stop = 10)
    trains = np.array([train1, train2])
    out = v.spike_argmax(trains)
    expected = np.array([1, 0])
    assert np.array_equal(out, expected)
