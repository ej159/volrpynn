"""
A module for optimisers that alters the network weights of a Model
based on stimulus and expected output
"""

import abc
import volrpynn.activation
import numpy

import numpy

class Optimiser():
    """An optimizer that, given a model, can train it to perform better on a
    given data set using the `train` method.
    """

    @abc.abstractmethod
    def train(model, x, y, loss_function):
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
        A trained model
        """
        pass

class GradientDescentOptimiser(Optimiser):
    """A gradient descent optimiser that takes Poisson rate input data to
       produce spike outputs, calculates weight changes and propagates them
       with a given learning rate
    """
    
    def __init__(self, learning_rate):
        """Constructs a gradient descent optimizer given a learning rate
    
        Args:
        learning_rate -- The alpha parameter for the rate of weight changes
                         (learning)
        """

    def train(model, xs, ys, loss_function):
        assert len(xs) == len(ys),  """Length of input data ({}) must be the same as output {}
               must be the same as output data ({})""".format(len(xs), len(ys))

        for index in range(len(xs)):
            x = xs[index]
            target_y = ys[index]
            y = model.predict(x)
            loss = loss_function(x, y)
        
            for layer in reversed(self.layer):
                loss = layer.backward(error, optimizer)
        return model

