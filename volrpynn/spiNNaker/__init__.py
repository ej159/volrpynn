import sys

module = sys.modules['volrpynn']

if hasattr(module, '__pynn__'):
    raise RuntimeError("PyNN has already been initialised")
else:
    import pyNN.spiNNaker as pynn
    module.__pynn__ = pynn

from volrpynn._pynn_setup import setup
setup(pynn)
del setup

from volrpynn import *
