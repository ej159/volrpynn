"""A module for spike related functions"""

import numpy

def spike_softmax(spiketrains):
    """Finds the softmax of a list of spiketrains by counting the spike rate

    Args:
    spiketrains -- A numpy array of neo.core.SpikeTrain

    Returns:
    An array of softmax values over the spike rates from each train
    """
    lengths = numpy.array(list(map(len, spiketrains)))
    shifted = lengths - numpy.max(lengths)
    e_x = numpy.exp(shifted)
    return numpy.diag(e_x / e_x.sum(axis = 0))

def spike_softmax_deriv(spiketrains):
    """Returns a function that can calculate the derived spike train rate with
    respect to a given neuron (index).

    Args:
    spiketrains -- A numpy array of neo.core.SpikeTrain

    Returns:
    A function that gives the desired spike train derivation wrt. a specific
    index.
    """
    softmax_res = spike_softmax(spiketrains)
    delta_res = numpy.vectorize(lambda x: -x * x)(softmax_res)
    def spike_index(index):
        delta_copy = numpy.copy(delta_res)
        delta_copy[index] = softmax_res[index] + delta_copy[index]
        return delta_copy
    return spike_index
