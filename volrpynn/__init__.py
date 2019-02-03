"""
VolrPyNN
========

A Python library for training and testing spiking neural network
models in PyNN.

Usage:
  import volrpynn.[backend] as v

Example:
  import volrpynn.nest as v

Available backends:
  nest
"""

# Import VolrPyNN 
from . import activation
from .activation import *
from . import error
from .error import *
from . import spike
from .report import *
from . import translation
from .translation import *
from .spike import *
from . import optimise
from .optimise import *
from . import layer
from .layer import *
from . import model
from .model import *
from . import main
from .main import *

