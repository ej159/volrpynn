"""
VolrPyNN
========

A Python library for training and testing spiking neural network
models in PyNN.
"""

# Hack to avoid Unicode crash
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

# Import VolrPyNN 
from . import layer
from .layer import *
from . import model
from .model import *
