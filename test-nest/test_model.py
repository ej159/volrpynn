import volrpynn as v
import pyNN.nest as pynn

def test_nest_population():
    p = pynn.Population(12, pynn.IF_cond_exp())
    assert p.size == 12

def test_nest_projection():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_ext())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    assert len(c.get('weight', format='list')) == 4
    
def test_nest_projection_gaussian():
    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_ext())
    c = pynn.Projection(p1, p2,
            pynn.AllToAllConnector(allow_self_connections=False))
    c.set('weight', pynn.RandomDistribution('normal', mu=0.5, sigma=0.1))
    weights = c.get('weight', format='array')
    assert len(weights[weights == 0]) < 1
