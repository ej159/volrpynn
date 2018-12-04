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

def test_softmax_spike_inactive():
    train1 = SpikeTrain([] * s, t_stop = 10)
    train2 = SpikeTrain([] * s, t_stop = 10)
    trains = np.array([train1, train2])
    expected = np.array([0.5, 0.5])
    assert np.allclose(v.spike_softmax(trains), expected)

def test_spike_argmax_regular():
    train1 = SpikeTrain([2, 20] * s, t_stop = 100)
    train2 = SpikeTrain([31, 40, 61] * s, t_stop = 100)
    train3 = SpikeTrain([1, 4] * s, t_stop = 100)
    train4 = SpikeTrain([1, 47] * s, t_stop = 100)
    trains = np.array([train1, train2, train3, train4])
    out = v.spike_argmax(trains)
    assert np.array_equal(out, np.array([0, 1, 0, 0]))

def test_spike_argmax_zeros():
    train1 = SpikeTrain([] * s, t_stop = 10)
    train2 = SpikeTrain([] * s, t_stop = 10)
    trains = np.array([train1, train2])
    out = v.spike_argmax(trains)
    assert out.size == len(trains)
    assert out.sum() == 0

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
