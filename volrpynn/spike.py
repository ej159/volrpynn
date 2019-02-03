"""A module for spike related functions"""

import numpy as np
from volrpynn import translation

LINEAR_TRANSLATION = translation.LinearTranslation()

def spike_count(spiketrains):
    """Counts the number of spikes in an array of SpikeTrains"""
    return np.array(list(map(len, spiketrains)))

def spike_count_normalised(spiketrains, interval_max = 1.0):
    """Counts the number of spikes in an array, normalised to an
    interval between 0 and a maximum value (that must be positive"""
    assert interval_max > 0, "Maximum interval value must be above 0"
    lengths = np.array(list(map(len, spiketrains)))
    spike_max = np.max(lengths)
    return lengths / np.max(lengths) if spike_max > 0 else lengths

def spike_rate(simulation_time):
    def spiketrains_rate(trains):
        return np.array(list(map(lambda t: len(t) / simulation_time, trains)))
    return spiketrains_rate

def spike_count_linear(spiketrains):
    return LINEAR_TRANSLATION.from_spikes(spike_count(spiketrains))

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
