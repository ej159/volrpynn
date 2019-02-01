import volrpynn.nest as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup()

def test_replicate_assert_size():
    p1 = pynn.Population(6, pynn.IF_cond_exp())
    p2 = pynn.Population(6, pynn.IF_cond_exp())
    p3 = pynn.Population(5, pynn.IF_cond_exp())
    with pytest.raises(ValueError):
        v.Replicate(p1, (p2, p3))
    with pytest.raises(ValueError):
        v.Replicate(p1, (p3, p2))

def test_replicate_create():
    p1 = pynn.Population(6, pynn.IF_cond_exp())
    p2 = pynn.Population(6, pynn.IF_cond_exp())
    p3 = pynn.Population(6, pynn.IF_cond_exp())
    l = v.Replicate(p1, (p2, p3), v.ReLU(), weights=(1, 1))
    pynn.run(1000)
    l.store_spikes()
    assert l.layer1.input.shape == (6,0)
    assert l.layer2.input.shape == (6,0)
    assert l.get_output().shape == (2, 6)

def test_replicate_can_replicate():
    p1 = pynn.Population(6, pynn.IF_cond_exp(i_offset = 10))
    p2 = pynn.Population(6, pynn.IF_cond_exp())
    p3 = pynn.Population(6, pynn.IF_cond_exp())
    l = v.Replicate(p1, (p2, p3), v.ReLU(), weights=(1, 1))
    pynn.run(1000)
    l.store_spikes()
    expected = np.ones((2, 6))
    assert np.allclose(expected, l.get_output())

# For backward test, see test_merge.py
