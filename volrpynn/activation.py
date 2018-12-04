"""
A module for derived activation functions
"""
import numpy as np

def relu_derived(output, error):
    gradient = np.multiply(output.T, error).T
    gradient[output < 0] = 0 # ReLU gradient
    return gradient

