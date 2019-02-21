"""Translation module for approximating spiking model data to and from differentiable functions"""

import abc
import numpy as np

class Translation():

    @abc.abstractmethod
    def from_spikes(self, spikes):
        pass

    @abc.abstractmethod
    def to_current(self, data):
        pass

    @abc.abstractmethod
    def normalise_weights(self, weights, input_neurons):
        pass

class LinearTranslation(Translation):
    """Translates according to a linear model for spiking neuron
    activation, where f(x) = 3.225x - 1.613"""

    alpha = 3.22500557
    beta = 1.61295370014

    def from_spikes(self, data):
        copy = data.copy()
        copy = (copy + self.beta) / self.alpha
        return copy
        
    def to_current(self, data):
        """Normalises and converts data into [1;12]"""
        copy = data.copy().astype(np.float64)
        divisor = 1 if data.max() == 0 else data.max() * 2
        copy /= divisor
        copy += 0.5 # Shift to positive
        copy *= 11
        copy += 1
        return copy

    def weights(self, weights, input_neurons):
        """Normalise the weights, such that the neuron activations are roughly
        following a linear progression"""
        if isinstance(weights, np.ndarray):
            least = weights.min()
            largest = weights.max()
            # Crop the weights to the interval [0;1] to get variable activation levels
            if least < 0:
                weights -= least
            if largest > 0:
                weights /= largest

        return (weights) * (0.065 / input_neurons)
