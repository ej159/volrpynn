"""
The layers of VolrPyNN which must all define a method for a backward-pass
through the layers (to update the layer weights), as well as getting, setting and
storing weights.
"""
import abc
import numpy as np
import volrpynn as v
from volrpynn.util import get_pynn as pynn

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

    def __init__(self, pop_in, pop_out, gradient_model, weights=None,
            decoder=v.spike_count):
        """
        Initialises a densely connected layer between two populations

        Args:
        pop_in -- The input population
        pop_out -- The output population
        gradient_model -- The function that calculates the neuron gradients
                          given the current spikes and errors from this layer
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to a normal distribution with mean 1.0
                   and standard deviation of 0.2
        decoder -- A function that can code a list of SpikeTrains into a numeric
                   numpy array
        """
        self.projection = pynn().Projection(pop_in, pop_out,
                pynn().AllToAllConnector(allow_self_connections=True))

        # Store gradient model
        assert callable(gradient_model), "gradient_model must be a function"
        self.gradient_model = gradient_model

        # Store decoder
        assert callable(decoder), "spike decoder must be a function"
        self.decoder = decoder

        # Assign given weights or default to 1
        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(1.0, 0.2, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

        # Prepare spike recordings
        self.projection.pre.record('spikes')

        
    def backward(self, output, error, optimiser):
        """Backward pass in the dense layer

        Args:
        output -- The output from the previous layer as a numpy array
        error -- The error in the output from this layer as a numpy array
        optimiser -- The optimiser that calculates the new layer weights, given
                     the current weights and the gradient deltas

        Returns:
        A tuple of the cached spikes from the first (input) layer and the errors
        """
        assert callable(optimiser), "Optimiser must be callable"

        # Activation gradient for the output
        output_derived = self.gradient_model(output)
        print("out", output_derived)
        print("err", error)
        output_delta = np.multiply(output_derived, error)

        # Calculate weight delta and error
        input_layer = self.decoder(self.spikes)
        layer_delta = np.outer(input_layer, output_delta)
        error_weighted = np.multiply(self.weights, layer_delta)

        # Optimise weights and store
        new_weights = optimiser(self.weights, layer_delta)
        self.set_weights(new_weights)
        
        # Return errors changes in backwards layer
        return input_layer, error_weighted.sum(axis=1)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight = weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments = self.projection.pre.get_data('spikes').segments
        self.spikes = np.array(segments[-1].spiketrains)
        return self.spikes

#class MergeLayer(Layer):

#class ReplicateLayer(Layer):
