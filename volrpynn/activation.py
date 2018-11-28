"""
A module for activation functions and their derivatives
"""
import numpy as np

def relu_derived(output, error, weights):
    weight_gradient = np.multiply(output.T, error).T
    weight_gradient[output < 0] = 0 # ReLU gradient
    layer_error = np.outer(weights.T, error).T
    return weight_gradient, layer_error

def relu_leaky_derived(output, error, weights):
    weight_gradient = np.multiply(output.T, error).T
    weight_gradient[output < 0] *= 0.01
    layer_error = np.outer(weights.T, error).T
    return weight_gradient, layer_error
