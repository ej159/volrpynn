import volrpynn.nest as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.reset()
    pynn.setup()

# def test_merge_assert_size():
#     p1 = pynn.Population(6, pynn.IF_cond_exp())
#     p2 = pynn.Population(6, pynn.IF_cond_exp())
#     p3 = pynn.Population(11, pynn.IF_cond_exp())
#     with pytest.raises(ValueError):
#         v.Merge((p1, p2), p3)
# 
# def test_merge_create():
#     p1 = pynn.Population(6, pynn.IF_cond_exp(i_offset = 10))
#     p2 = pynn.Population(6, pynn.IF_cond_exp(i_offset = 10))
#     p3 = pynn.Population(12, pynn.IF_cond_exp())
#     l = v.Merge((p1, p2), p3, v.ReLU(), weights=(1, 1))
#     pynn.run(1000)
#     l.store_spikes()
#     assert l.layer1.input.shape[0] == 6
#     assert l.layer2.input.shape[0] == 6
#     assert l.get_output().shape[0] == 12
#     assert np.allclose(np.ones((6, 12)), l.layer1.weights)
#     assert np.allclose(np.ones((6, 12)), l.layer1.weights)
# 
# def test_merge_stimulus():
#     p1 = pynn.Population(6, pynn.IF_cond_exp(i_offset = 10))
#     p2 = pynn.Population(6, pynn.IF_cond_exp(i_offset = 10))
#     p3 = pynn.Population(12, pynn.IF_cond_exp())
#     l = v.Merge((p1, p2), p3, v.ReLU(), weights=(1, 1))
#     pynn.run(50)
#     l.store_spikes()
#     expected = np.ones(12)
#     assert np.allclose(expected, l.get_output())
# 
# def test_merge_backwards():
#     p0 = pynn.Population(6, pynn.IF_cond_exp())
#     p1 = pynn.Population(6, pynn.IF_cond_exp())
#     p2 = pynn.Population(6, pynn.IF_cond_exp())
#     p3 = pynn.Population(12, pynn.IF_cond_exp())
#     l1 = v.Replicate(p0, (p1, p2), weights=(1, 1))
#     l2 = v.Merge((p1, p2), p3, v.ReLU())
#     m = v.Model(l1, l2)
#     m.predict(np.zeros(6) + 10, 50)
#     error = m.backward(np.ones((1, 12)), lambda w, g, b, bg: (w, b))
#     assert error.shape == (6, )
#     assert np.allclose(np.zeros(6) + 12, error, atol=5)
