"""A module for spike related functions"""

import numpy as np

def spike_count(spiketrains):
    """Counts the number of spikes in an array of SpikeTrains"""
    return np.array(list(map(len, spiketrains)))

def spike_softmax(spiketrains):
    """Finds the softmax of a list of spiketrains by counting the spike rate

    Args:
    spiketrains -- A numpy array of neo.core.SpikeTrain

    Returns:
    An array of softmax values over the spike rates from each train
    """
    lengths = np.array(list(map(len, spiketrains)))
    shifted = lengths - np.max(lengths)
    e_x = np.exp(shifted)
    return (e_x / e_x.sum(axis = 0))

def spike_argmax(spiketrains):
    """Argmax over the neuron with the largest number of spikes.
    If no spikes exist, or if there are ties between spike counts,
    the first neuron is chosen. This behaviour can be changed by
    using the spike_argmax_random.

    Args:
    spiketrains -- A numpy array of neo.core.SpikeTrain

    Returns:
    An array with zeros, except for the neuron with the highest spike count
    """
    lengths = np.array(list(map(len, spiketrains)))
    max_value = lengths.max()
    max_array = np.zeros(lengths.shape)

    non_zero_indices = np.flatnonzero(lengths == max_value)
    max_index = non_zero_indices[0]
    max_array[max_index] = 1

    return max_array
    
def spike_argmax_randomise(spiketrains):
    """Argmax over the neuron with the largest number of spikes.
    If no spikes exist, or if there are ties between spike counts,
    a random neuron exist. This behaviour can be changed with the
    randomise_ties argument.

    Args:
    spiketrains -- A numpy array of neo.core.SpikeTrain

    Returns:
    An array with zeros, except for the neuron with the highest spike count
    """
    lengths = np.array(list(map(len, spiketrains)))
    max_value = lengths.max()
    max_array = np.zeros(lengths.shape)

    if max_value == 0:
        max_index = 0
    else: 
        non_zero_indices = np.flatnonzero(lengths == max_value)
        max_index = np.random.choice(non_zero_indices)
        max_array[max_index] = 1

    return max_array
