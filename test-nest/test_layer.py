import volrpynn as v
import pyNN.nest as pynn
import numpy

def test_nest_dense_create():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(pynn, p1, p2, lambda x: x)
    expected_weights = numpy.ones((12, 10))
    assert numpy.array_equal(d.projection.get('weight', format='array'),
            expected_weights)
