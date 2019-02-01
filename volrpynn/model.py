"""
The model of VolrPyNN
"""

import numpy as np
from volrpynn.layer import Dense, Replicate
from volrpynn.util import get_pynn as pynn
from volrpynn.activation import ActivationFunction, ReLU

class Model():
    """A model of a neural network experiment"""

    # These neuron parameters lead to a iaf_psc_alpha neuron that fires with a
    # constant rate of approximately f_out = I_e / 10.0
    FIXED_RATE_NEURON = {
            "tau_syn_I":5,  # Decay time for inhibitory inputs
            "tau_syn_E":5,  # Decay time for excitatory inputs
            "tau_refrac":0, # Refractory period
            "tau_m":20,     # Membrane time constant
            "v_thresh":-50, # Voltage threshold
            "v_rest":-65,   # Resting potential
            "v_reset":-65,  # Reset potential
            "e_rev_I":-70,  # Reverse potential for inhibitions
            "e_rev_E":0,    # Reverse potential for excitations
            "i_offset":0,   # Offset (constant) input current
            "cm":1          # Membrane capacity
        }

    def __init__(self, *layers):
        """Instantiates the model with a PyNN implementation and a Layer description

        Args:
        pop_in -- The PyNN input population
        pop_out -- The PyNN output population
        layers -- A list of Layers build over PyNN populations
        """
        if not layers:
            raise ValueError("Layers must not be empty")

        if not isinstance(layers[0], (Dense, Replicate)):
            raise ValueError("First layer must be a Dense or Replicate layer")

        # Assign populations and layers
        self.node_input = layers[0].pop_in
        self.layers = layers

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            population = pynn().Population(1, pynn().IF_cond_exp(**self.FIXED_RATE_NEURON))
            self.input_populations.append(population)

        self.input_assembly = pynn().Assembly(*self.input_populations)

        self.input_projection = pynn().Projection(self.input_assembly,
                                                  self.node_input, pynn().OneToOneConnector(),
                                                  pynn().StaticSynapse(weight=1.0))

    def backward(self, error, optimiser):
        """Performs a backwards pass through the model *without* executing the
        simulation, which is assumed to happen *before* this method is called.
        This function has side-effects: while performing the backward pass,
        the model is updated with new weights.

        Args:
        error -- The numerical error that the model should adjust to as a numpy
                 array
        optimiser -- A function that calculates the new weights of a layer
                     given the current layer weights and the weights deltas
                     from the derived layer activation function

        Returns:
        The error (loss) from the input layer after backpropagation from the output
        layer to the input layer.
        """
        # Backprop through the layers
        layer_error = error
        for layer in reversed(self.layers):
            layer_error = layer.backward(layer_error, optimiser)

        return layer_error

    def _normalise_data(self, data):
        """Normalises the data according to a linear model for spiking neuron
        activation, where f(x) = 3.225x - 1.613"""
        # Normalise the data to [1;40]
        copy = data.copy().astype(np.float64)
        copy /= max(1, data.max())
        copy *= 39
        copy += 1
        # Scale the data linearly
        return (copy + 1.61295370014) / 3.22500557

    def predict(self, rates, time):
        """Predicts an output by simulating the model with the given input

        Args:
            rates -- A list of Poisson rates, with the same dimension as the
                     input layer
            time -- The number of time to run the simulation in milliseconds

        Returns:
            An array of neo.core.SpikeTrains from the output layer
        """
        self.set_input(rates)

        return self.simulate(time)

    def reset(self):
        """Resets the PyNN simulation backend and all the recorded weights and
           spiketrains"""
        pynn().reset()

        for layer in self.layers:
            layer.restore_weights()

        # Reset recorders
        for recorder in pynn().simulator.state.recorders:
            recorder.clear()
    
    def set_input(self, data):
        """Assigns the vector of input current values to the input neurons"""
        assert len(data) == len(self.input_populations),\
                "Input dimension ({}) must match input node size ({})"\
                  .format(len(data), len(self.input_populations))
        normalised = self._normalise_data(np.array(data))
        for index, x in enumerate(normalised):
            self.input_populations[index].set(i_offset=x)
        self.inputs = normalised

    def simulate(self, time):
        """Reset simulation and restore weights"""
        self.reset()

        pynn().run(time)

        # Collect spikes
        for layer in self.layers:
            layer.store_spikes()
        output_values = self.layers[-1].get_output()

        pynn().end()
        return output_values
