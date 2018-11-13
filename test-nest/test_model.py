import volrpynn as v
import pyNN.nest as pynn
import numpy

pynn.setup()

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

def test_nest_create_input_populations():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(pynn, p1, p2, lambda x: x)
    m = v.Model(pynn, p1, p2, l)
    assert len(m.input_populations) == 2
    m.set_input([1, 0.2])
    assert m.input_populations[0].get('rate') == 1.0
    assert m.input_populations[1].get('rate') == 0.2

def test_nest_predict():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    d = pynn.random.RandomDistribution('normal', mu=0.5, sigma=0.1)
    l = v.Dense(pynn, p1, p2, lambda x: x, weights = d)
    m = v.Model(pynn, p1, p2, l)
    out = m.predict(numpy.array([1, 0]), 1000)
    print(out)

    
