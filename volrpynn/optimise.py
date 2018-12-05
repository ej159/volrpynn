"""
A module for optimisers that alters the network weights of a Model
based on stimulus and expected output
"""

import abc
import numpy
from volrpynn import *

class Optimiser():
    """An optimizer that, given a model, can train it to perform better on a
    given data set using the `train` method.
    """

    @abc.abstractmethod
    def train(self, model, x, y, loss_function):
        """Trains the given model using the input data as input and the output 
        data as target (expected) output, using a specific loss function
        
        Args:
        model -- A model with the 'predict' and 'train' methods
        xs -- The input data set as a numpy array where the first dimension
              describes the number of training instances and the second dimension
              the number of input neurons.
              The input data is expected to describe Poisson spike rates per
              neuron.
        ys -- The expected label data as a numpy array where the first dimension
              describes the number of training instances and the second dimension
              the expected neuron output.
        loss_function -- A function that takes the predicted output of the model
                         and calculates numerical values for how 'wrong' that 
                         output is, compared to the y labels

        Returns:
        A tuple of a trained Model and a list of predicted outputs
        """
        pass

class GradientDescentOptimiser(Optimiser):
    """A gradient descent optimiser that takes Poisson rate input data to
       produce spike outputs, calculates weight changes and propagates them
       with a given learning rate
    """
    
    def __init__(self, decoder, learning_rate, simulation_time = 500):
        """Constructs a gradient descent optimiser given a learning rate
    
        Args:
        decoder -- A decoder that can decode a list of SpikeTrains to a list of
                   numerical values for use in error and backpropagation
                   functions
        learning_rate -- The alpha parameter for the rate of weight changes
                         (learning)
        simulation_time -- Time in milliseconds how long each data point
                           be simulated. Defaults to 500 ms
        """
        assert callable(decoder)
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.simulation_time = simulation_time

    def test(self, model, xs, ys, error_function):
        """Test the model with the given input and expected output. 
        The error function calculates the error rate, given the predicted
        and expected (target) output.

        Args:
        model -- The model to test
        xs -- The input data as a 2-dimensional numpy array
              where the first dimension is the separate data entries and
              the second dimension is the Poisson rates for the neurons
        ys -- The expected (target) output data as a 2-dimensional array
              where the first dimension is the separate data entries and
              the second dimension is the expected predicted output

        Returns:
        A list of errors
        """
        assert callable(error_function), "error_function must be callable"
        errors = []
        for x, target_y in zip(xs, ys):
            actual_y = model.predict(xs, self.simulation_time)
            error = error_function(actual_y, target_y)
            errors.append(error)
        return errors

    def train(self, model, xs, ys, error_function):
        assert len(xs) == len(ys),  """Length of input data ({}) must be the same as output data ({})""".format(len(xs), len(ys))
        assert callable(error_function), "Error function must be callable"

        # Define update function
        def calculate_weights(weights, deltas):
            return weights - (np.multiply(self.learning_rate, deltas))

        actual = []
        for x, target_y in zip(xs, ys):
            y = model.predict(x, self.simulation_time)
            actual.append(self.decoder(y))
            error = error_function(self.decoder(y), target_y)
            model.backward(error, calculate_weights)
            
        return model, actual
