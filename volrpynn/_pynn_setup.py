"""
Module for initialising PyNN.

This module sets up the PyNN simulator, irrespective of the backend.

"""

def setup(pynn):
    pynn.setup(rng_seeds_seed = 100)
