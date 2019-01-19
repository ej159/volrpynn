"""
A module for error functions and their derivatives
"""
import abc
import numpy as np
import volrpynn.activation

epsilon = 1e-8

class ErrorFunction():
    """An error function consists of a function that calculates the error
    between some actual and expected output, as well as a derived version of
    that function"""

    @abc.abstractmethod
    def __call__(self, output, labels):
        pass

    def prime(self, output, labels):
        pass

class SumSquared(ErrorFunction):

    def __call__(self, output, labels):
        # We half the output to simplify the differentiation
        return 0.5 * ((output - labels) ** 2).sum(axis=0)

    def prime(self, output, labels):
        return (output - labels)

class CrossEntropy(ErrorFunction):

    def __call__(self, output, labels):
        return - (np.log(output + epsilon) * labels).sum()

    def prime(self, output, labels):
        return - labels / output

class SoftmaxCrossEntropy(ErrorFunction):

    def softmax(self, x):
        shifted = x - np.max(x)
        e_x = np.exp(shifted)
        return (e_x / e_x.sum(axis = 0))

    def __call__(self, output, labels):
        return CrossEntropy()(self.softmax(output), labels)

    def prime(self, output, labels):
        return self.softmax(output) - labels

