"""
A module for derived activation functions
"""
import numpy as np

def relu_derived(output, weights, error):
    weight_gradient = np.multiply(output.T, error).T
    weight_gradient[output < 0] = 0 # ReLU gradient
    layer_error = np.inner(weights, error)
    return weight_gradient, layer_error

def relu_leaky_derived(output, weights, error):
    weight_gradient = np.multiply(output.T, error).T
    weight_gradient[output < 0] *= 0.01
    layer_error = np.inner(weights, error)
    return weight_gradient, layer_error
