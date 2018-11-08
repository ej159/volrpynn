import volrpynn as v
import pyNN.nest as pynn

def test_nest_population():
    p = pynn.Population(12, pynn.IF_cond_exp())
    assert p.size == 12
