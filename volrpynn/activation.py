"""
A module for derived activation functions
"""
import numpy as np

def relu_derived(output):
    output[output < 0] = 0 # ReLU gradient
    return output

def sigmoid(output):
    return 1.0 / (1.0 + np.exp(-output)) 

def sigmoid_derived(output):
    s = sigmoid(output)
    return s * (1 - s)
