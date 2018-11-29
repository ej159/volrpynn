"""
The model of VolrPyNN
"""

import numpy as np
from volrpynn import util

class Model(object):
    """A model of a neural network experiment"""

    def __init__(self, pynn, *layers):
        """Instantiates the model with a PyNN implementation and a Layer description

        Args:
        pynn -- The PyNN simulator instance, needed to interface with the
                backend
        pop_in -- The PyNN input population
        pop_out -- The PyNN output population
        layers -- A list of Layers build over PyNN populations
        """
        # Ensure that pynn is set
        assert pynn != None, "Please assign PyNN backend"
        assert len(layers) > 0, "Layers must not be empty"
        
        # Assign populations and layers
        self.pynn = pynn
        self.node_input = layers[0].projection.pre
        self.node_output = layers[-1].projection.post
        self.layers = layers

        # Prepare recording
        self.node_output.record('spikes')

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            self.input_populations.append(pynn.Population(1,
                pynn.SpikeSourcePoisson(rate = 1.0)))
        self.input_source = pynn.Assembly(*self.input_populations)
        self.input_projection = pynn.Projection(self.input_source, self.node_input,
                pynn.OneToOneConnector(), pynn.StaticSynapse(weight = 1.0))

    def set_input(self, poisson_rates):
        """Assigns the vector of poisson rates to the input neurons, that inject
        spikes into the model"""
        assert len(poisson_rates) == len(self.input_populations),\
                "Input dimension ({}) must match input node size ({})"\
                  .format(len(poisson_rates), len(self.input_populations))
        for index in range(len(poisson_rates)):
            self.input_populations[index].set(rate = poisson_rates[index])

    def predict(self, xs, time):
        """Predicts an output by simulating the model with the given input
        
        Args:
            xs -- A list of Poisson rates, with the same dimension as the
                  input layer
            time -- The number of time to run the simulation in milliseconds

        Returns:
            An array of neo.core.SpikeTrains from the output layer
        """
        self.set_input(xs)
        return self.simulate(time)

    def backward(self, error, error_update):
        """Performs a backwards pass through the model *without* executing the
        simulation, which is assumed to happen *before* this method is called.
        This function has side-effects: while performing the backward pass,
        the model is updated with new weights.

        Args:
        error -- The numerical error that the model should adjust to
        error_update -- A function that calculates weight delta and backward
                        errors according to some optimisation algorithm.
                        The function is expected to take the layer spikes,
                        weights and the error to adjust to. It is expected to
                        return a tuple of (weight deltas, errors).

        Returns:
        The error (loss) from the input layer after backpropagation from the output
        layer to the input layer.
        """
        layer_error = np.copy(error)
        # Backprop through the layers
        for layer in reversed(self.layers):
            layer_error = layer.backward(layer_error, error_update)
        return layer_error

    def simulate(self, time):
        self.pynn.reset()

        self.pynn.run(time)

        # Collect spikes
        for layer in self.layers:
            layer.store_spikes()
        output_spikes = self.node_output.getSpikes().segments[-1].spiketrains

        self.pynn.end()
        return output_spikes
