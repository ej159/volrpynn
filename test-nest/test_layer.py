import volrpynn as v
import pyNN.nest as pynn
import numpy

pynn.setup()

def test_nest_dense_create():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(pynn, p1, p2, lambda x: x)
    expected_weights = numpy.ones((12, 10))
    assert numpy.array_equal(d.projection.get('weight', format='array'),
            expected_weights)

def test_nest_dense_restore():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(pynn, p1, p2, lambda x: x, weights = 2)
    d.set_weights(-1)
    assert numpy.array_equal(d.projection.get('weight', format='array'),
            numpy.ones((12, 10)) * -1)
    d.projection.set(weight = 1) # Simulate reset()
    assert numpy.array_equal(d.projection.get('weight', format='array'),
            numpy.ones((12, 10)))
    d.restore_weights()
    assert numpy.array_equal(d.projection.get('weight', format='array'),
            numpy.ones((12, 10)) * -1)

