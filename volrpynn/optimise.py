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
    
    def __init__(self, learning_rate, simulation_time = 1000):
        """Constructs a gradient descent optimiser given a learning rate
    
        Args:
        learning_rate -- The alpha parameter for the rate of weight changes
                         (learning)
        simulation_time -- Time in milliseconds how long each data point
                           be simulated. Defaults to 1000 ms
        """
        self.learning_rate = learning_rate
        self.simulation_time = simulation_time

    def test(self, model, xs, ys):
        """Test the model with the given input and expected output. 
        The error function calculates the error rate, given the predicted
        and expected (target) output.
        The output is a list of booleans that indicates whether the output
        was the same as the expected (True) or not (False).

        Args:
        model -- The model to test
        xs -- The input data as a 2-dimensional numpy array
              where the first dimension is the separate data entries and
              the second dimension is the Poisson rates for the neurons
        ys -- The expected (target) output data as a 2-dimensional array
              where the first dimension is the separate data entries and
              the second dimension is the expected predicted output

        Returns:
        A list of booleans
        """
        hits = []
        for x, target_y in zip(xs, ys):
            output = self.test_single(model, x, target_y)
            hit = np.allclose(output, target_y)
            hits.append(hit)
        return hits

    def test_single(self, model, x, target_y):
        """Tests a single data entry by simulating the input 'x' and
        returning the error between the actual and expected output.

        Args:
        model -- The model to test
        x -- An array of input rates for the input neurons
        target_y -- The expected output as an array of numbers

        Returns:
        The predicted output of the model
        """
        return model.predict(x, self.simulation_time)

    def train(self, model, xs, ys):
        assert len(xs) == len(ys),  """Length of input data ({}) must be the same as output data ({})""".format(len(xs), len(ys))

        # Define update function
        def calculate_weights(weights, deltas):
            return weights - (np.multiply(self.learning_rate, deltas))

        actual_ys = []
        for x, target_y in zip(xs, ys):
            # Forward pass
            output = self.test_single(model, x, target_y)
            actual_ys.append(output)
            # Backward pass
            error = output - target_y
            model.backward(output, error, calculate_weights)
            
        return model, actual_ys
