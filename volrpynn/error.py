"""
A module for error functions and their derivatives
"""
import abc
import numpy as np
import volrpynn.activation

class ErrorFunction():
    """An error function consists of a function that calculates the error
    between some actual and expected output, as well as a derived version of
    that function"""

    @abc.abstractmethod
    def __call__(self, output, labels):
        pass

    def prime(self, output, labels):
        pass

class CrossEntropy(ErrorFunction):

    def __call__(self, output, labels):
        return np.multiply(labels, np.log(output))

    def prime(self, output, labels):
        return - labels / output

class SumSquared(ErrorFunction):

    def __call__(self, output, labels):
        # We half the output to simplify the differentiation
        return 0.5 * ((output - labels) ** 2).sum(axis=0)

    def prime(self, output, labels):
        return -(output - labels)

def argmax_index(xs, randomise_ties=True):
    max_value = xs.max()
    non_zero_indices = np.flatnonzero(xs == max_value)

    if randomise_ties:
        return np.random.choice(non_zero_indices)
    else:
        return non_zero_indices[0]
