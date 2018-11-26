"""
A module for activation functions and their derivatives
"""
import numpy as np

def relu_derived(output, error, weights):
    weight_gradient = output.dot(error)
    layer_error = error.dot(weights.T).T
    return weight_gradient, layer_error
