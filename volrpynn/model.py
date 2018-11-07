import numpy as np
from volrpynn import util

class Model(object):
    """A model of a supervised neural network experiment"""

    def __init__(self, input_node, output_node, nodes, pynn):
        """Instantiates the model with a single input node, output node
        and a list of LearningNodes, from which spikes can be read.

        Args:
        input_node -- A single node used to inject Poisson inputs
        output_node -- A single node used for learning
        nodes -- A list of LearningNodes
        pynn -- The PyNN simulator instance, needed to interface with the
                backend
        """
        # Ensure that pynn is set
        assert pynn != None, "Please assign PyNN backend"
        
        # Assign nodes
        self.input_node = input_node
        self.output_node = output_node
        self.pynn_nodes = pynn_nodes

        # Create input Poisson source
        self.input_populations = []
        for _ in range(self.input_node.size):
            self.input_populations.append(pynn.Population(self.input_node.size,
                pynn.SpikeSourcePoisson(rate = 1.0), label = 'input'))
        self.input_source = pynn.Assembly(*self.input_populations)
        self.input_projection = pynn.Projection(self.input_source, self.input_node.node,
                pynn.OneToOneConnector(), pynn.StaticSynapse(weight = 1.0))

    def set_input(self, poisson_rates):
        """Assigns the vector of poisson rates to the input neurons, that inject
        spikes into the model"""
        assert len(poisson_rates) == len(self.input_populations),\
                "Input dimension ({}) must match input node size ({})"\
                  .format(len(poisson_rates), len(self.input_populations))
        for index in range(len(poisson_rates)):
            self.input_populations[index].set(rate = poisson_rates[index])

    def train(self, xs, ys):
        """Trains the model on a sequence of inputs and expected outputs.
        The input data is expected to describe Poisson spike rates per input neuron
        and the output is expected to describe anticipated output spikes."""
        self.set_input(xs)
        
    
    def predict(self, data, time):
        """Predicts an output by simulating the model with the given input
        Poisson rates"""
        self.size = node.size
        spikes = []
        for rates in data:
            assert self.input_source.size == len(rates)
            self.input_source.set(rate=rates)
            self.simulate(time)
            spikes.append(self.output_node.spikes)
        return spikes

    def simulate(self, time):
        # Start recording
        for _, node in self.pynn_nodes.items():
            node.record('spikes')

        self.pynn.run(time)

        # Collect spikes
        for _, node in self.learning_nodes.items():
            node.store_spikes()

        self.pynn.end()
        return self

class LearningNode(object):
    """A node that can learn in the neural network, backed by a PyNN
    population."""

    def __init__(self, node):
        self.edges_in = []
        self.edges_out = []
        self.node = node 
        self.spikes = None
        self.size = node.size

    def connect_to(self, to, edge):
        self.edges_out.append(edge)
        to.edges_in.append(edge)

    def backprop(self, errors, is_output = True):
        if is_output:
            assert len(edges_out) == 0

        output = self.get_spikes(self.node)

    def store_spikes(self):
        segments = self.node.getSpikes().segments
        assert len(segments) == 1
        self.spikes = segments[0].spiketrains        

    def set_weights(self, index, weights):
        pass
