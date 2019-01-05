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

    def get_output(self):
        """Returns a numpy array of the decoded output"""
        return self.decoder(self.output)

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

# Decoder  = forward pass
# Gradient model = backward pass
# Include rate forward pass
# - Check that it performs as a regular AI
# Rearrange to have linearity -> nonlinearity
# Demonstrate scientifically novelty
#  - Convenience
#  - Interoperability
#    - ONNX?
#  - So far: repackaged in-the-loop training idea
#  - Next: More complicated ways to use spikes
#    - NN to calculate weight update: meta learning
#    - Instead of static learning, do it parameterised 
#
# Define a task to test against
# From a high level: Getting something with a spiking neural network and an ANN
# that does the learning
# Idea: Transform n-dimensional signal to fewer neurons
#   -> Consider internal weight update mechanisms as parameterised, to be learned

class Decode(Layer):
    def __init__(self, pop_in, decoder = v.spike_argmax):
        assert callable(decoder), "Decoder must be a function"
        self.pop_in = pop_in
        self.decoder = decoder
        # Prepare population for recording
        self.pop_in.record('spikes')
        self.weights = np.ones(pop_in.size)

    def backward(self, error, optimiser):
        return error

    def store_spikes(self):
        self.output = np.array(self.pop_in.getSpikes().segments[-1].spiketrains)
        return self

class Dense(Layer):
    """A densely connected neural layer between two populations,
       creating a PyNN all-to-all connection (projection) between the
       populations."""

    def __init__(self, pop_in, pop_out, gradient_model, weights=None,
            decoder=v.spike_softmax):
        """
        Initialises a densely connected layer between two populations
output
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
        self.pop_in = pop_in
        self.pop_out = pop_out
        self.projection = pynn().Projection(pop_in, pop_out,
                pynn().AllToAllConnector(allow_self_connections=True))

        # Store gradient model
        assert callable(gradient_model), "gradient_model must be a function"
        self.gradient_model = gradient_model

        # Store decoder
        assert callable(decoder), "spike decoder must be a function"
        self.decoder = decoder

        # Assign given weights or default to a normal distribution
        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(1.0, 0.2, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

        # Prepare spike recordings
        self.projection.pre.record('spikes')
        self.projection.post.record('spikes')
        
    def backward(self, error, optimiser):
        """Backward pass in the dense layer

        Args:
        error -- The error in the output from this layer as a numpy array
        optimiser -- The optimiser that calculates the new layer weights, given
                     the current weights and the gradient deltas

        Returns:
        A tuple of the cached spikes from the first (input) layer and the errors
        """
        assert callable(optimiser), "Optimiser must be callable"

        try:
            self.input
        except AttributeError:
            raise RuntimeError("No input data found. Please simulate the model" + 
                               " before doing a backward pass")

        # Calculate activations for output layer
        input_decoded = self.decoder(self.input)
        output_activations = np.matmul(input_decoded, self.weights)
        
        # Calculate layer delta and weight optimisations
        output_gradients = self.gradient_model(output_activations)
        delta = np.multiply(error, output_gradients)

        # Ensure correct multiplication of data
        if len(input_decoded.shape) == 1:
            weights_delta = np.outer(input_decoded, delta)
        else:
            weights_delta = np.matmul(input_decoded.T, delta)

        # Calculate weight optimisation and store
        new_weights = optimiser(self.weights, weights_delta)
        self.set_weights(new_weights)
        
        # Return errors changes in backwards layer
        backprop = np.matmul(delta, self.weights.T)
        return backprop
    
    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight = weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments_in = self.projection.pre.get_data('spikes').segments
        self.input = np.array(segments_in[-1].spiketrains)
        segments_out = self.projection.post.get_data('spikes').segments
        self.output = np.array(segments_out[-1].spiketrains)
        return self

#class MergeLayer(Layer):

#class ReplicateLayer(Layer):
