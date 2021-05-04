"""Different metrics to help using states patterns.

A States pattern represent a discrete value, that changes
during a time period.
States represent a number of different states patterns.
A States patterns array, is a numpy array containing the states patterns,
and in demod it is by convention a n_patterns * n_times array.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


# Make available the functions of other modules
def get_states_durations(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the durations from the states matrix given as input.

    Will merge end and start of the diaries to compute the
    duration, if the starting and ending states or the same.

    Args:
        states: States patterns array

    Returns:
        durations : a ndarray with the durations of each states
        corresponding_states : a ndarray with the labels of the states
            for each duration

    Note:
        The order of the durations is not the real one, as the
        unchanging states will be placed at the end.
    """
    # find where the transitions between states occurs
    transitions_indices = np.where(states != np.roll(states, 1, axis=1))
    # split the indices
    household_indices = transitions_indices[0]
    time_indices = transitions_indices[1]

    # find in the indices the start and end of each household
    indices_first_household = np.where(
        household_indices != np.roll(household_indices, 1)
    )[0]
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

    return durations, corresponding_states


def get_durations_by_states(states: np.ndarray) -> Dict[Any, np.ndarray]:
    """Return a dict of all the durations depending on the states.

    Args:
        states (ndarray(int, size=(n_households, n_times))):
            The matrix of the states for the housholds at all the times

    Returns:
        A dictionary where you can access by giving the state and
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
    by `McKenna et al (2020)
    <https://doi.org/10.1016/j.erss.2020.101572>`_ .


    Args:
        states: An array containing the states for which we want to
            compute the graph metrics

    Returns:
        network_size, network_density, centrality, homophily.
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
