"""
The model of VolrPyNN
"""

import numpy as np
from volrpynn.layer import Dense, Replicate
from volrpynn.util import get_pynn as pynn
from volrpynn.activation import ActivationFunction, ReLU
from volrpynn.translation import Translation, LinearTranslation

# Default neuron parameters for stable networks
DEFAULT_NEURON_PARAMETERS = {
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

class Model():
    """A model of a neural network experiment"""

    def __init__(self, *layers, translation=LinearTranslation()):
        """Instantiates the model with a PyNN implementation and a Layer description

        Args:
        layers -- A list of Layers build over PyNN populations
        translation -- A method for translating the input data and weights into
                       a spiking model
        """
        if not layers:
            raise ValueError("Layers must not be empty")

        if not isinstance(layers[0], (Dense, Replicate)):
            raise ValueError("First layer must be a Dense or Replicate layer")

        if not isinstance(translation, Translation):
            raise ValueError("Translation scheme must be a Translation")

        # Assign populations and layers
        self.node_input = layers[0].pop_in
        self.layers = layers
        self.translation = translation

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            population = pynn().Population(1,
                    pynn().IF_cond_exp(**DEFAULT_NEURON_PARAMETERS))
            self.input_populations.append(population)

        self.input_assembly = pynn().Assembly(*self.input_populations)

        # Calculate the input weights, but with only 1 input neuron because of
        # the 1:1 connector
        input_weights = self.translation.weights(1, 1)
        self.input_projection = pynn().Projection(self.input_assembly,
                                                  self.node_input, pynn().OneToOneConnector(),
                                                  pynn().StaticSynapse(weight=input_weights))

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
        error = np.array(error)
        if len(error.shape) == 1:
            layer_error = error.reshape(1, -1)
        else:
            layer_error = error

        for layer in reversed(self.layers):
            layer_error = layer.backward(layer_error, optimiser)

        return layer_error

    def predict(self, data, time):
        """Predicts an output by simulating the model with the given input

        Args:
            data -- A list of inputs
            time -- The number of time to run the simulation in milliseconds

        Returns:
            An array of decoded values from the layers
        """
        self.set_input(data)
        return self.simulate(time)

    def reset_cache(self):
        """Resets all the caches in the layers"""
        for layer in self.layers:
            layer.reset_cache()

    def reset_weights(self):
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
        for index, x in enumerate(data):
            self.input_populations[index].set(i_offset=x)
        self.inputs = data

    def simulate(self, time):
        """Restore weights and reset simulation"""
        self.reset_weights()

        pynn().run(time)

        # Collect spikes
        for layer in self.layers:
            layer.store_spikes()
        output_values = self.layers[-1].get_output()

        pynn().end()
        return output_values
