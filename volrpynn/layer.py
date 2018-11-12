"""
The layers of VolrPyNN which must all define a method for a backward-pass
through the layers (to update the layer weights), as well as getting, setting and
storing weights.
"""
import abc
import numpy

class Layer():
    """A neural network layer with a PyNN-backed neural network population and a backwards
       weight-update function, based on existing spikes"""

    @abc.abstractmethod
    def backward(self, optimizer):
        """Performs backwards optimisation based on the given optimizer"""
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass

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

    def __init__(self, pynn, pop_in, pop_out, decoder, weights=None):
        """
        Initialises a densely connected layer between two populations

        Args:
        pynn -- The PyNN backend
        pop_in -- The input population
        pop_out -- The output population
        decoder -- The spike decoder
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to 1
        """
        assert callable(decoder), "Decoder must be a function"
        self.decoder = decoder

        self.projection = pynn.Projection(pop_in, pop_out,
                pynn.AllToAllConnector(allow_self_connections=False))

        if weights:
            self.projection.set(weight = weights)
        else:
            self.projection.set(weight = 1)

        # Prepare spike recordings
        self.projection.pre.record('spikes')
        self.weights = self.projection.get('weight', format='array')
        self.cache = numpy.repeat(1, len(self.weights))

    def backward(self, delta_y, optimizer):
        assert callable(optimizer), "Optimizer must be a function"

        decoded = self.decode(self.spikes) # Decode spike values
        # Calculate weight changes
        delta = numpy.multiply(decoded, delta_y).T # Element-wise (hadamard)
        weight_gradient = numpy.dot(delta, decoded)
        sum_vectorize = numpy.vectorize(lambda xs: numpy.sum(xs))
        cache_gradient = sum_vectorize(delta)

        # Update weights and cache
        self.weights, self.cache = optimizer(self.weight, self.cache, weight_gradient, cache_gradient)
        
        # Return error changes in backwards layer
        return numpy.dot(self.weights.T, delta_y).T

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        assert self.weights.shape == weights.shape
        self.weights = weights        

    def store_spikes(self):
        segments = self.projection.pre.getSpikes().segments
        assert len(segments) == 1, "Only single compartment models supported"
        self.spikes = segments[0].spiketrains
        return self.spikes

#class MergeLayer(Layer):

#class ReplicateLayer(Layer):
