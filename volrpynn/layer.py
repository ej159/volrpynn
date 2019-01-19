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

    def __init__(self, pop_in, pop_out, gradient_model, decoder=v.spike_softmax):
        """
        Initialises a densely connected layer between two populations output
        Args:
        pop_in -- The input population
        pop_out -- The output population
        gradient_model -- An ActivationFunction that calculates the neuron gradients
                          given the current spikes and errors from this layer
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to a normal distribution with mean 1.0
                   and standard deviation of 0.2
        decoder -- A function that can code a list of SpikeTrains into a numeric
                   numpy array
        wta -- Winner-takes-all is a boolean flag to set if inhibitory
               connections should be installed to inhibit the entire population
               after the first spike was fired
        """
        self.pop_in = pop_in
        self.pop_out = pop_out
        self.output = None
        self.weights = None

        # Store gradient model
        if not isinstance(gradient_model, v.ActivationFunction):
            raise ValueError("gradient_model must be an activation function")

        self.gradient_model = gradient_model

        # Store decoder
        assert callable(decoder), "spike decoder must be a function"
        self.decoder = decoder

    @abc.abstractmethod
    def backward(self, error, optimiser):
        """Performs backwards optimisation based on the given error and
        activation derivative"""

    def get_output(self):
        """Returns a numpy array of the decoded output"""
        return self.decoder(self.output)

    def get_weights(self):
        """Returns the weights as a matrix of size (input, output)"""
        return self.weights

    def restore_weights(self):
        """Restores the current weights of the layer"""
        self.set_weights(self.weights)
        return self.weights

    @abc.abstractmethod
    def set_weights(self, weights):
        """Sets the weights of the network layer"""

    @abc.abstractmethod
    def store_spikes(self):
        """Stores the spikes of the current run"""

class Decode(Layer):
    """A layer that only decodes the spike trains without any activation passes"""

    def __init__(self, pop_in, decoder=v.spike_softmax):
        super(Decode, self).__init__(pop_in, None, v.UnitActivation(), decoder)
        self.weights = np.ones(pop_in)

    def backward(self, error, optimiser):
        return error

    def set_weights(self, weights):
        return self.weights

    def store_spikes(self):
        self.output = np.array(self.pop_in.getSpikes().segments[-1].spiketrains)
        return self

class Dense(Layer):
    """A densely connected neural layer between two populations,
       creating a PyNN all-to-all connection (projection) between the
       populations."""

    def __init__(self, pop_in, pop_out, gradient_model=v.ReLU(), weights=None,
                 decoder=v.spike_count_normalised):
        """
        Initialises a densely connected layer between two populations
output
        """
        super(Dense, self).__init__(pop_in, pop_out, gradient_model, decoder)

        self.input = None

        # Prepare spike recordings
        self.projection = pynn().Projection(pop_in, pop_out,
                                            pynn().AllToAllConnector(allow_self_connections=False))

        self.pop_in.record('spikes')
        self.pop_out.record('spikes')

        # Assign given weights or default to a normal distribution
        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(0, 0.5, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

        self.biases = np.zeros(pop_out.size)

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
        output_gradients = self.gradient_model.prime(output_activations + self.biases)
        delta = np.multiply(error, output_gradients)

        # Ensure correct multiplication of data
        if len(input_decoded.shape) == 1:
            weights_delta = np.outer(input_decoded, delta)
        else:
            weights_delta = np.matmul(input_decoded.T, delta)

        # Calculate weight and bias optimisation and store
        (new_weights, new_biases) = optimiser(self.weights, weights_delta,
                                              self.biases, delta)

        # NEST cannot handle too large weight values, so this guard
        # ensures that the simulation keeps running, despite large weights
        new_weights[new_weights > 1000] = 1000
        new_weights[new_weights < -1000] = -1000

        self.set_weights(new_weights)
        self.biases = new_biases

        # Return errors changes in backwards layer
        backprop = np.matmul(delta, self.weights.T)
        return backprop

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight=weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments_in = self.projection.pre.get_data('spikes').segments
        self.input = np.array(segments_in[-1].spiketrains)
        segments_out = self.projection.post.get_data('spikes').segments
        self.output = np.array(segments_out[-1].spiketrains)
        return self

class Merge(Layer):

    def __is_tuple(x, name):
        assert type(x) is list or type(x) is tuple and len(x) == 2, \
            name + " must be tuple of length two"

    def __init__(self, pop_in, pop_out, gradient_model = v.ReLU(), weights = None, decoder
            = v.spike_softmax, wta = True):
        super(Merge, self).__init__(pop_in, pop_out, gradient_model, decoder,
                wta)

        self.__is_tuple(pop_in, "Input populations")
        self.__is_tuple(weights, "Layer weights")

        # Connect and prepare spike recordings
        connection_list1 = list(permutations(range(pop_in[0].size,pop_in[0].size), 2))
   #     connection_list2 = list(permutations(range(pop_in[0].size[(x, x + pop_in[0].size) for x in range(pop_in[1].size)]
        #self.projection1 = pynn().Projection(pop_in[0], pop_out,
        #        pynn().FromListConnector(connection_list1))
        #self.projection2 = pynn().Projection(pop_in[1], pop_out,
        #        pynn().FromListConnector(connection_list2))
        self.pop_in[0].record('spikes')
        self.pop_in[1].record('spikes')
        self.pop_out.record('spikes')

        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(0, 1, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.__is_tuple(weights, "Layer weights")
        #self.projection1.set(weight = weights[0])
        #self.projection1.set(weight = weights[1])
        self.weights = weights

class Replicate(Layer):
    
    def __is_tuple(self, x, name):
        assert type(x) is list or type(x) is tuple and len(x) == 2, \
            name + " must be tuple of length two"

    def __init__(self, pop_in, pop_out, gradient_model = v.ReLU(), weights = None, decoder
            = v.spike_softmax):
        super(Replicate, self).__init__(pop_in, pop_out, gradient_model, decoder)

        self.__is_tuple(pop_out, "Output populations")

        # Connect and prepare spike recordings
        connection_list1 = [(x, x) for x in range(pop_out[0].size)]
        #self.projection1 = pynn().Projection(pop_in[0], pop_out,
        #        pynn().FromListConnector(connection_list1))
        #self.projection2 = pynn().Projection(pop_in[1], pop_out,
        #        pynn().FromListConnector(connection_list2))
        self.pop_in.record('spikes')
        self.pop_out[0].record('spikes')
        self.pop_out[1].record('spikes')

        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(0, 1, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.__is_tuple(weights, "Layer weights")
        #self.projection1.set(weight = weights[0])
        #self.projection1.set(weight = weights[1])
        self.weights = weights
