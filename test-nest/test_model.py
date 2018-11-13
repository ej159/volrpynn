import volrpynn as v
import pyNN.nest as pynn
import numpy

pynn.setup()

### Basic PyNN tests

def test_nest_population():
    p = pynn.Population(12, pynn.IF_cond_exp())
    assert p.size == 12

def test_nest_projection():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    assert len(c.get('weight', format='list')) == 4

def test_nest_projection_gaussian():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    c.set(weight=pynn.random.RandomDistribution('normal', mu=0.5, sigma=0.1))
    weights = c.get('weight', format='array')
    assert len(weights[weights == 0]) < 1

### Model tests

def test_nest_input_projection():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2)
    m = v.Model(pynn, l)
    actual_weights = m.input_projection.get('weight', format='array')
    expected_weights = numpy.array([[1, numpy.NaN], [numpy.NaN, 1]])
    assert numpy.allclose(actual_weights, expected_weights, equal_nan=True)

def test_nest_create_input_populations():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2)
    m = v.Model(pynn, l)
    assert len(m.input_populations) == 2
    m.set_input([1, 0.2])
    assert m.input_populations[0].get('rate') == 1.0
    assert m.input_populations[1].get('rate') == 0.2

def test_nest_predict():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    d = pynn.random.RandomDistribution('normal', mu=1, sigma=0.1)
    l = v.Dense(pynn, p1, p2, weights = d)
    m = v.Model(pynn, l)
    out = m.predict(numpy.array([10, 0]), 1000)
    assert len(out) == 2
    assert len(out[0]) > 0
    assert len(out[1]) > 0 # Should be above 0 despite 0 input, due to all-to-all

