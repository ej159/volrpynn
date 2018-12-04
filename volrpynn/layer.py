"""
The layers of VolrPyNN which must all define a method for a backward-pass
through the layers (to update the layer weights), as well as getting, setting and
storing weights.
"""
import abc
import numpy as np

class Layer():
    """A neural network layer with a PyNN-backed neural network population and a backwards
       weight-update function, based on existing spikes"""

    @abc.abstractmethod
    def backward(self, error, activation):
        """Performs backwards optimisation based on the given error and
        activation derivative"""
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def restore_weights(self):
        """Restores the current weights of the layer"""
        self.set_weights(self.weights)
        return self.weights

    @abc.abstractmethod
    def set_weights(self, weights):
        """Sets the weights of the network layer"""
        pass

    @abc.abstractmethod
    def store_spikes(self):
        """Stores the spikes of the current run"""
        pass

class Dense(Layer):
    """A densely connected neural layer between two populations.
       Assumes the PyNN projection is as an all-to-all connection."""

    def __init__(self, pynn, pop_in, pop_out, gradient_model, weights=None):
        """
        Initialises a densely connected layer between two populations

        Args:
        pynn -- The PyNN backend
        pop_in -- The input population
        pop_out -- The output population
        gradient_model -- The function that calculates the neuron gradients
                          given the current spikes and errors from this layer
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to 1
        """
        self.projection = pynn.Projection(pop_in, pop_out,
                pynn.AllToAllConnector(allow_self_connections=False))

        # Store gradient model
        assert callable(gradient_model), "gradient_model must be a function"
        self.gradient_model = gradient_model

        # Assign given weights or default to 1
        self.set_weights(weights if weights else 1)

        # Prepare spike recordings
        self.projection.post.record('spikes')

        
    def backward(self, errors, optimiser):
        """Backward pass in the dense layer

        Args:
        errors -- The errors in the output from this layer
        optimiser -- The optimiser that calculates the new layer weights, given
                     the current weights and the gradient deltas
        """
        assert callable(optimiser), "Optimiser must be callable"

        # Activation gradient
        gradient = self.gradient_model(self.spikes, errors)
  
        # Calculate weight changes and update
        layer_delta = np.multiply(weights.T, gradient).T
        new_weights = optimiser(self.weights, layer_delta)
        self.set_weights(new_weights)
        
        # Return errors changes in backwards layer
        return layer_delta

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight = weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments = self.projection.post.get_data('spikes').segments
        self.spikes = segments[-1].spiketrains
        return self.spikes

#class MergeLayer(Layer):

#class ReplicateLayer(Layer):
