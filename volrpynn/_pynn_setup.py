"""
Module for initialising PyNN.

This module sets up the PyNN simulator, irrespective of the backend.

"""

def setup(pynn):
    # Use true randomness 
    from multiprocessing import cpu_count
    pynn.setup(rng_seeds_seed = 1,  
               local_num_threads = cpu_count())
