"""
The model of VolrPyNN
"""

import numpy as np
from volrpynn import util

class Model(object):
    """A model of a supervised neural network experiment"""

    def __init__(self, pynn, node_input, node_output, *layers):
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
        self.node_input = node_input
        self.node_output = node_output
        self.layers = layers

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            self.input_populations.append(pynn.Population(input_size,
                pynn.SpikeSourcePoisson(rate = 1.0), label = 'input'))
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

    def train(self, xs, ys, error_function, optimizer, time):
        """Trains the model on a sequence of inputs and expected outputs.
        The input data is expected to describe Poisson spike rates per input neuron
        and the output is expected to describe anticipated output spikes.
        This method updates the weights and errors in the model layers.
        
        Args:
        xs -- The input data
        ys -- The expected output data
        error_function -- The error function that calculates the error of the
                          spike train output (input: spike array)
        optimizer -- A function that optimizes the weights and error cache and
                     returns a new weight and error cache configuration
        time -- The time to simulate during the training
        """
        self.set_input(xs)
        output_spikes = simulate(time)
        error = error_function(output_spikes)
        
        for layer in reversed(self.layer):
            error = layer.backward(error, optimizer)
        return error
    
    def predict(self, xs, time):
        """Predicts an output by simulating the model with the given input
        
        Args:
            xs -- A list of Poisson rates, with the same dimension as the
                  input layer
            time -- The number of time to run the simulation in milliseconds
        """
        self.set_input(xs)
        return self.simulate(time)

    def simulate(self, time):
        self.pynn.reset()
        for layer in self.layers:
            layer.restore_weights()

        self.pynn.run(time)

        # Collect spikes
        for layer in self.layers:
            layer.store_spikes()
        output_spikes = self.node_output.getSpikes().segments[0].spiketrains

        self.pynn.end()
        return output_spikes
