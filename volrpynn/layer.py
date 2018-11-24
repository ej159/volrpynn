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

    def __init__(self, pynn, pop_in, pop_out, weights=None):
        """
        Initialises a densely connected layer between two populations

        Args:
        pynn -- The PyNN backend
        pop_in -- The input population
        pop_out -- The output population
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to 1
        """
        self.projection = pynn.Projection(pop_in, pop_out,
                pynn.AllToAllConnector(allow_self_connections=False))

        # Assign given weights or default to 1
        self.set_weights(weights if weights else 1)

        # Prepare spike recordings
        self.projection.pre.record('spikes')

    def backward(self, delta_y, optimizer):
        """Backward pass in the dense layer

        Args:
        delta_y -- The error in the output from this layer
        optimizer -- The optimizer that calculates the weight changes and the
                     error to propagate to the next layer, given the spikes and
                     weights from this layer
        """
        assert callable(optimizer), "Optimizer must be a function"
  
        # Calculate weight changes and update
        self.weights, error = optimizer(self.spikes, self.weight) 
        
        # Return error changes in backwards layer
        return error

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight = weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments = self.projection.pre.getSpikes().segments
        self.spikes = segments[-1].spiketrains
        return self.spikes

#class MergeLayer(Layer):

#class ReplicateLayer(Layer):
