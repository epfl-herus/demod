"""Different metrics to help using states patterns.

A state is a discrete value, that can change
during a time period.
A state pattern records the value of the state over the time period.
States represent a number of different states patterns on the same
time period.
In demod states are numpy arrays containing the states patterns,
and is of shape n_patterns * n_times.
"""

from demod.utils.sim_types import States, TPMs
from typing import Any, Dict, List, Tuple, Union
import numpy as np


# Make available the functions of other modules
def get_states_durations(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the durations from the states matrix given as input.

    Will merge end and start of the diaries to compute the
    duration, if the starting and ending states or the same.

    Args:
        states: States patterns array

    Returns:
        * **durations**, a ndarray with the durations of each states
        * **corresponding_states**, a ndarray with the labels of the states
          for each duration

    .. note::

        The position of the durations in the returned
        arrays does not correspond to how
        they happened during time.
    """
    # find where the transitions between states occurs
    transitions_indices = np.where(states != np.roll(states, 1, axis=1))
    # split the indices
    household_indices = transitions_indices[0]
    time_indices = transitions_indices[1]

    # find in the indices the start and end of each household
    indices_first_household = np.where(
        household_indices != np.roll(household_indices, 1)
    )[0] if len(np.unique(household_indices)) > 1 else np.array([0], dtype=int)
    indices_last_household = np.roll(indices_first_household - 1, -1)

    # calculate the durations
    durations = np.roll(time_indices, -1) - time_indices
    # merging the beggining and end if possible they are the same activities
    durations[indices_last_household] = (
        states.shape[1]
        - time_indices[indices_last_household]
        + time_indices[indices_first_household]
    )

    # get the value of the new states
    corresponding_states = states[transitions_indices]

    # add housholds that never change states

    # first compute where the states are the same in a whole axis of a row
    no_transition_households = np.where(
        np.all(states == states[:, 0][:, None], axis=1)
    )[0]
    # their durations is the whole array
    durations_no_transition = [
        states.shape[1] for i in no_transition_households
    ]

    durations = np.concatenate((durations, durations_no_transition))
    corresponding_states = np.concatenate(
        (corresponding_states, states[no_transition_households, 0])
    )

    return np.array(durations, dtype=int), corresponding_states


def get_durations_by_states(states: np.ndarray) -> Dict[Any, np.ndarray]:
    """Return a dict of all the durations depending on the states.

    Args:
        states (ndarray(int, size=(n_households, n_times))):
            The matrix of the states for the housholds at all the times

    Returns:
        **durations_dict**, dictionary where you can access by giving the state and
        recieve an array containing
        all the durations for this state
    """
    durations, durations_labels = get_states_durations(states)
    # store the results in a dic
    dict_durations_by_states = {}
    for state in np.unique(durations_labels):  # iterates over the states
        # get the durations corresponding to that state
        dict_durations_by_states[state] = durations[durations_labels == state]

    return dict_durations_by_states


def count(array: np.ndarray) -> List[Tuple[Any, int]]:
    """Count the number of elements in the input array.

    Usefull to have a quick view of the values present in
    the array.

    Args:
        array: the input array to be counted

    Returns:
        A list of tuples contanining the elements and their counts
    """
    unique, counts = np.unique(array, return_counts=True)
    return [(u, c) for u, c in zip(unique, counts)]


def graph_metrics(
    states: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute and return 4 graph metrics for given States patterns.

    These metrics are an attempt to implement graph metrics as proposed
    by [McKenna2020]_. The following description of the metrics
    comes from the paper.


    Args:
        states: An array containing the states for which we want to
            compute the graph metrics

    Returns:
        * **network_size**, Refers to the number of unique nodes in a network.
          For states sequence networks the size describes the number of
          unique observations of states and the time they occur at.
          A larger network is one where more states were observed and
          gives an indication of the diversity of states that occurred.
        * **network_density**, The fraction of the total maximum number
          of possible connections that are actually observed in the network.
          Network density is calculated by counting the number of unique edges
          observed in the network and dividing it by this maximum.
          It is a measure of the diversity of sequences present in the network.
          A denser network is one where there are more alternative pathways
          between time periods by different people.
        * **centrality**, Refers to how dominant certain pathways or nodes
          are within a network. Central states nodes are those that are
          highly connected to others. Central connections or edges are those
          that appear in many individual sequences i.e. those with high
          ‘weight’. There are multiple ways of calculating centrality,
          here we use edge weight as a measure of (path) centrality and
          compute the mean and standard deviation of the distribution
          of edge weights for each network.
        * **homophily**, Refers to the likelihood that a node is connected
          to another node of the same type. Here this refers to when a state
          in one time period is connected to the same state in an
          adjacent time period. It is a measure of the extent to which
          states tend to endure in unbroken lasting sequences.
          A simple measure of homophily used here is the proportion of times
          a state in one time period is followed by the same state
          in the following time period.
          This is calculated for each sequence and the mean of
          the distribution is taken as the measure
          of the homophily of the network.
    """
    # Transpose the states as we will work over time
    states = states.T
    # gets the max states size to store the edges
    max_size = np.max(states) + 1
    # initialize the first edges with the last matrix
    old_s = np.array(states[-1])
    directed_edges = []

    for s in states:
        indices, counts = np.unique(
            np.c_[old_s, s], axis=0, return_counts=True
        )
        indices = tuple(indices.T)
        edges = np.zeros((max_size, max_size))
        edges[indices] = counts
        directed_edges.append(edges)

        old_s = s

    times, source_nodes, end_nodes = np.where(directed_edges)

    source_nodes_id = np.char.add(np.char.add(
        np.array(times, dtype=str),
        ['_' for i in range(len(times))]),
        np.array(source_nodes, dtype=str)
    )
    end_nodes_id = np.char.add(np.char.add(
        np.array(times, dtype=str),
        ['_' for i in range(len(times))]),
        np.array(end_nodes, dtype=str)
    )
    edge_weights = np.array(directed_edges)[times, source_nodes, end_nodes]

    network_size = len(np.unique(np.c_[source_nodes_id, end_nodes_id]))
    network_density = len(source_nodes_id) / (
        # used nodes / the max number of nodes possible
        (len(directed_edges)-1) * len(np.unique(source_nodes))**2
    )
    centrality = np.mean(edge_weights)
    homophily = (
        np.sum((end_nodes == source_nodes) * edge_weights)
        / np.sum(edge_weights)
    )

    return network_size, network_density, centrality, homophily


def sparsity(tpm: TPMs):
    """Return the proportion of 0 elements in the TPMs.

    | = 0. If all the elements are given
    | = 1. If the tpms are only 0

    Args:
        tpm: The transition probability matrices.
    """
    return 1. - (np.sum(tpm > 0) / np.prod(tpm.shape))


def average_state_metric(
    simulated_state: States, measured_state: States,
    average_over_timestep: bool = True,
) -> Union[List[float], float]:
    r"""Determine the average per time step state error.

    This is a generalization of the Average occupancy metric
    from [Flett2016]_ .

    It determines the average per time step state error between
    simulated data and real data from the TOU surveys, quantifying the
    quality of the calibration of the simulation.

    It is defined as

    .. math::
        A_{s} = \sum^{T}_{t=1} \frac{
            |\overline{N}^{sim}_s(t)
            - \overline{N}^{mes}_s(t)|
        }{T}

    where
    :math:`\overline{N}_s(t)` are the average number of subjects in state
    :math:`s`
    at time :math:`t`.

    Two means of analysis are possible with this metric.

        1. It can be used to calculate the prediction for the average
           per time step results of multiple profiles generated using the
           model. This determines how effectively the model converges to
           the population average.
        2. It can be used to calculate the prediction error for each
           individual profile. The mean of this error can be used to
           determine how effectively individual profiles replicate
           the input data.

    Args:
        simulated_state: The state that have been simulated.
            Array of shape = (n_subjects/n_households, n_times).
            The value should be the number of person performing
            the states at each time of the diaries.
        measured_state: The state that where measured.
            Same array as simulated_state.
            n_times must be the same as simulated_state, but
            n_subjects/n_households can be different.
        average_over_timestep: whether to average the result over the
            time steps. If false, will return an array of shape =
            (n_times).

    Returns:
        **average_state_error**, The average state error between
        simulated_state and measured_state.
    """
    # Average the states at each time step
    average_sim_state = np.mean(simulated_state, axis=0)
    average_mes_state = np.mean(measured_state, axis=0)

    # Compute the abolute error between sim and measured
    average_errors = np.abs(average_sim_state-average_mes_state)

    # Chooses the return
    if average_over_timestep:
        return np.mean(average_errors)
    else:
        return average_errors


def state_duration_distribution_metric(
    simulated_state: States, measured_state: States,
    average_over_timestep: bool = True,
) -> Dict[Any, Union[List[float], float]]:
    r"""Compare the difference in state durations.

    This is a generalization of the State duration distribution metric
    from [Flett2016]_ .
    This metric is commonly known as the Earth Movers Distance:
    a commonly used quantitative histogram similarity measure where
    the bin values are not independent and cross-bin analysis is required.

    The ‘error’ is
    the sum ofthe absolute difference between the simulated and mesured
    data CDFs at each duration value for each state.

    It is defined as

    .. math::
        DurDist_s =
        \sum^{T}_{t=1} \frac{
            |\sum^{t}_{d=1}\overline{P}^{sim}_s(d)
            - \sum^{t}_{d=1}\overline{P}^{mes}_s(d)|
        }{T}

    where :math:`\overline{P}_s(d)` is the probability of a state duration of
    :math:`d` for state :math:`s`.



    Note that this metric weights the same,
    erorrs in durations probs that are unlikely
    to happen as errors in duration probs that are very likely.

    It could be nice to do a weighted average instead, that uses
    the weight according to the pdf ?


    Args:
        simulated_state: The state that have been simulated.
            Array of shape = (n_subjects/n_households, n_times).
            The value should be the number of person performing
            the states at each time of the diaries.
        measured_state: The state that where measured.
            Same array as simulated_state.
            n_times must be the same as simulated_state, but
            n_subjects/n_households can be different.
        average_over_timestep: whether to average the cdf over the
            time steps. If false, will return an array of shape =
            (n_times) being the absolute difference of each cdf
            value.

    Returns:
        duration_distribution, The state duration distribution
        difference metric.
    """
    # Finds the durations of the states
    dur_dict_sim = get_durations_by_states(simulated_state)
    dur_dict_mes = get_durations_by_states(measured_state)

    n_times = measured_state.shape[1]

    dic_out = {}
    # Each state is handled separately
    for state in set((*dur_dict_sim.keys(), *dur_dict_mes.keys())):
        # Finds the cdf of the duration for that state
        cdf_sim = np.cumsum(
            np.bincount(dur_dict_sim[state], minlength=n_times + 1)
            / len(dur_dict_sim[state])
        )
        cdf_mes = np.cumsum(
            np.bincount(dur_dict_mes[state], minlength=n_times + 1)
            / len(dur_dict_mes[state])
        )
        # Error between cdfs
        err = np.abs(cdf_sim - cdf_mes)
        # Adds this state to the outputs
        dic_out[state] = (np.mean(err) if average_over_timestep else err)

    return dic_out


def levenshtein_edit_distance(profile1: np.ndarray, profile2: np.ndarray):
    """Compute the Levenshtein Edit Distance Method (LEDM) between 2 profiles.

    Can be used to compare the similarity between two discrete profiles,
    as suggested by [Flett2016]_ .

    You can use this distance combined with
    :py:func:`demod.metrics.loads.profiles_similarity`
    to compare many different profiles, by passing levenshtein_edit_distance
    as an arg of profiles_similarity().

    .. warning::

        Currently this is only implemented if the profiles contains
        single digit integers ranging from 0 to 10 (not included).

    .. warning::

        Requires an extra library to be installed
        `<https://pypi.org/project/python-Levenshtein/>`_

    Args:
        profile1, profile2: Two profiles given as arrays of size 1.
            They must be integers.

    Following description comes from [Flett2016]_

    This LEDM method is used to quantify the dissimilarity between
    two strings by quantifying the measures needed to transform one
    into the other. In the LEDM a ‘cost’ of 1 is assigned for each edit
    (insertions, deletions, and replacements) required in the trans-
    formation. For example, transforming 110111 to 001011 would
    require a minimum edit of a replacement of the first digit, deletion
    of the second, and insertion of the last digit – a total cost of 3.
    The
    approach can therefore be applied when comparing two numerical
    profiles. When two profiles are compared, for clarity,the total ‘cost’
    is converted from a per-time step to an hour equivalent by dividing
    the result by the number of time steps per hour.
    The metric can be used in two ways.

    1. It can be used to compare the output profiles with the input
       dataset. The smallest cost per profile, representative of the closest
       match, is determined and an average calculated across all mod-
       elled days. This is a measure of the average similarity between
       generated profiles and the closest real profile.

    2. Each profile in either the input dataset or model output
       dataset can be compared with other profiles in the same dataset
       quantifying the behavioural similarity within and between each
       dataset.
    """
    try:
        from Levenshtein import distance
    except ImportError as imp_err:
        raise ImportError(
            'You need to import Levenshtein distance module: \n'
            'pip install python-Levenshtein'
        ) from imp_err

    # Convert the arrays to strings
    return distance(
        ''.join(profile1.astype(str)),
        ''.join(profile2.astype(str)),
    )
