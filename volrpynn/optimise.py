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
        error_function -- A function that takes the predicted output of the model
                          and calculates numerical values for how 'wrong' that 
                          output is, compared to the y labels
        
        Returns:
        A tuple of the trained model and a list of prediction errors
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
        self.iterations = 1

    def test(self, model, xs, ys, report = None):
        """Test the model with the given input and expected output.
        The output is a report of type Report.

        Args:
        model -- The model to test
        xs -- The input data as a 2-dimensional numpy array
              where the first dimension is the separate data entries and
              the second dimension is the Poisson rates for the neurons
        ys -- The expected (target) output data as a 2-dimensional array
              where the first dimension is the separate data entries and
              the second dimension is the expected predicted output
        report -- A Report with an associated cost function to calculate 
                  the model score. Defaults to SumSquared

        Returns:
        A report of the output values and accuracy for hitting the target
        """
        report = ErrorCost(SumSquared()) if report == None else report
        for x, target_y in zip(xs, ys):
            output = self.test_single(model, x, target_y)
            report.add(output, target_y)
        return report

    def test_single(self, model, x, target_y):
        """Tests a single data entry by simulating the input 'x' and
        returning the error between the actual and expected output.

        Args:
        model -- The model to test
        xs -- An array of input rates for the input neurons
        ys -- The expected output as an array of numbers
        error_function -- The function to calculate the loss of the predictions

        Returns:
        The predicted output of the model
        """
        return model.predict(x, self.simulation_time)

    def train(self, model, xs, ys, error_function):
        assert len(xs) == len(ys),  """Length of input data ({}) must be the same as output data ({})""".format(len(xs), len(ys))
        assert isinstance(error_function, ErrorFunction), "Error function must \
be an instance of the ErrorFunction class"

        errors = []
        for x, target_y in zip(xs, ys):
            # Forward pass
            output = self.test_single(model, x, target_y)
            error_forward = error_function(output, target_y)

            errors.append(error_forward)

            # Backward pass
            error_prime = error_function.prime(output, target_y)

            # Define update optimisation function
            def optimise_weights(weights, weight_gradients,
                                 biases, bias_gradients):
                wg = np.multiply(self.learning_rate, weight_gradients)
                bg = np.multiply(self.learning_rate, bias_gradients)
                return (weights - wg, biases - bg)

            model.backward(error_prime, optimise_weights)
            self.iterations += 1

        return model, errors
