"""
A module for error functions and their derivatives
"""
import abc
import numpy as np
import volrpynn.activation

def sum_squared_error(output, labels):
    return ((output - labels) ** 2).sum()

def argmax_index(xs, randomise_ties=True):
    max_value = xs.max()

    if max_value == 0:
        return np.zeros(xs.shape)
    
    non_zero_indices = np.flatnonzero(xs == max_value)

    if randomise_ties:
        return np.random.choice(non_zero_indices)
    else:
        return non_zero_indices[0]
