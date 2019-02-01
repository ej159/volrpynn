"""
A module for derived activation functions
"""
import abc
import numpy as np

class ActivationFunction():
    """An activation function with a forward and backward (derived) pass"""

    @abc.abstractmethod
    def __call__(self, activation):
        """Computes the activation given some input"""

    @abc.abstractmethod
    def prime(self, activation):
        """The derived version of the activation function"""

class UnitActivation(ActivationFunction):

    def __call__(self, activation):
        return activation

    def prime(self, activation):
        return activation

class ReLU(ActivationFunction):
    """The Rectified Linear Unit activation function"""

    def __call__(self, activation):
        return np.maximum(0, activation)

    def prime(self, activation):
        activation_copy = np.ones(activation.shape)
        activation_copy[activation <= 0] = 0 # ReLU gradient
        return activation_copy

class Sigmoid(ActivationFunction):

    def __call__(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def prime(self, activation):
        sigmoid = self(activation)
        return sigmoid * (1 - sigmoid)
