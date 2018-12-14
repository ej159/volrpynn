"""
The model of VolrPyNN
"""

import numpy as np
import volrpynn as v
from volrpynn.util import get_pynn as pynn

class Model(object):
    """A model of a neural network experiment"""

    def __init__(self, *layers):
        """Instantiates the model with a PyNN implementation and a Layer description

        Args:
        pop_in -- The PyNN input population
        pop_out -- The PyNN output population
        layers -- A list of Layers build over PyNN populations
        """
        assert len(layers) > 0, "Layers must not be empty"
        
        # Assign populations and layers
        self.node_input = layers[0].projection.pre
        self.node_output = layers[-1].projection.post
        self.layers = layers

        # Prepare recording
        self.node_output.record('spikes')

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            population = pynn().Population(1, pynn().SpikeSourcePoisson(rate = 1.0))
            population.record('spikes')
            self.input_populations.append(population)
            
        self.input_assembly = pynn().Assembly(*self.input_populations)

        self.input_projection = pynn().Projection(self.input_assembly, 
                 self.node_input, pynn().OneToOneConnector(),
                 pynn().StaticSynapse(weight = 1.0))

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

    def backward(self, output, error, optimiser):
        """Performs a backwards pass through the model *without* executing the
        simulation, which is assumed to happen *before* this method is called.
        This function has side-effects: while performing the backward pass,
        the model is updated with new weights.

        Args:
        error -- The numerical error that the model should adjust to as a numpy
                 array
        output -- The output from the model, with which to backpropagate, as a
                  numpy array
        optimiser -- A function that calculates the new weights of a layer
                     given the current layer weights and the weights deltas
                     from the derived layer activation function

        Returns:
        The error (loss) from the input layer after backpropagation from the output
        layer to the input layer.
        """
        layer_error = np.copy(error)
        layer_output = np.copy(output)
        # Backprop through the layers
        for layer in reversed(self.layers):
            layer_output, layer_error = layer.backward(layer_output, layer_error, optimiser)
        return layer_error

    def simulate(self, time):
        # Reset simulation and restore weights
        pynn().reset()
        for layer in self.layers:
            layer.restore_weights()

        pynn().run(time)

        # Collect spikes
        for layer in self.layers:
            layer.store_spikes()
        output_spikes = self.node_output.getSpikes().segments[-1].spiketrains
        output_values = self.layers[-1].decoder(output_spikes)
        
        pynn().end()
        return output_values
