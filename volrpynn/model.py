"""
The model of VolrPyNN
"""

from volrpynn.util import get_pynn as pynn

class Model():
    """A model of a neural network experiment"""

    # These neuron parameters lead to a iaf_psc_alpha neuron that fires with a
    # constant rate of approximately f_out = I_e / 10.0
    FIXED_RATE_NEURON = {'i_offset': 0.,        # constant input
                         'tau_m': 82.,     # membrane time constant
                         'v_thresh': -55.,     # threshold potential
                         'v_rest': -70.,      # membrane resting potential
                         'tau_refrac': 2.,      # refractory period
                         'v_reset': -80.,  # reset potential
                         'cm': 320.,      # membrane capacitance
                         }      # initial membrane potential

    def __init__(self, *layers):
        """Instantiates the model with a PyNN implementation and a Layer description

        Args:
        pop_in -- The PyNN input population
        pop_out -- The PyNN output population
        layers -- A list of Layers build over PyNN populations
        """
        if not layers:
            raise ValueError("Layers must not be empty")

        # Assign populations and layers
        self.node_input = layers[0].projection.pre
        self.layers = layers

        # Create input Poisson sources
        self.input_populations = []
        input_size = self.node_input.size
        for _ in range(input_size):
            population = pynn().Population(1, pynn().IF_curr_alpha(i_offset=0.0))
            self.input_populations.append(population)

        self.input_assembly = pynn().Assembly(*self.input_populations)

        self.input_projection = pynn().Projection(self.input_assembly,
                                                  self.node_input, pynn().OneToOneConnector(),
                                                  pynn().StaticSynapse(weight=1.0))

    def set_input(self, poisson_rates):
        """Assigns the vector of poisson rates to the input neurons, that inject
        spikes into the model"""
        assert len(poisson_rates) == len(self.input_populations),\
                "Input dimension ({}) must match input node size ({})"\
                  .format(len(poisson_rates), len(self.input_populations))
        for index, rate in enumerate(poisson_rates):
            self.input_populations[index].set(i_offset=rate)

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
        layer_error = error
        # Backprop through the layers
        for layer in reversed(self.layers):
            layer_error = layer.backward(layer_error, optimiser)
        return layer_error

    def reset(self):
        """Resets the PyNN simulation backend and all the recorded weights and
           spiketrains"""
        pynn().reset()

        for layer in self.layers:
            layer.restore_weights()

        # Reset recorders
        for recorder in pynn().simulator.state.recorders:
            recorder.clear()

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
