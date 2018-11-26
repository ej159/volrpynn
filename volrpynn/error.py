"""
A module for error functions and their derivatives
"""
import abc
import numpy
import volrpynn.activation

class Error():
    """An error function consists of a function that can calculate
       a numeric 'loss' from a list of numerical values and their expected
       numerical labels (y values), as well as the derivative of that function,
       for use in backpropagation.
       Please note that the error functions do not accept SpikeTrains."""
    
    @abc.abstractmethod
    def error(self, output, labels):
        pass

    @abc.abstractmethod
    def error_derived(self, output, labels):
        pass

class SumSquaredError(Error):
    """A loss function based on the sum of squared errors"""

    def error(self, output, labels):
        return ((output - labels) ** 2).sum()

    def error_derived(self, output, labels):
        return output - labels
