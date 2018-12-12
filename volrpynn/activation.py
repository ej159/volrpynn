"""
A module for derived activation functions
"""
import numpy as np

def relu_derived(output):
    output[output < 0] = 0 # ReLU gradient
    return output

def sigmoid_derived(output):
    return output * (1 - output)
