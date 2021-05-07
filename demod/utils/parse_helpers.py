"""Helper functions for parsing Datasets."""

from typing import Dict, Tuple
import numpy as np
import pandas

from .sim_types import States, Any, Union


def states_to_transitions(
    states: States,
    return_duration: bool = False,
    include_same_state: bool = False,
    ignore_end_start_transitions: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert a state array to transitions array.

    For the durations of states between the nights, if the first
    states and the last states are the same, the duration is computed
    as the sum. If not, the two durations are kept.
    transition at time t: new_state = state[t], old_state = state[t-1]

    Args:
        states: The array containing the states.
            shape=(n_diaries, n_times), dtype=int or str
        return_duration: Wether to return the duration of the states.
            Defaults to False.
        include_same_state: Whether to also include transitions from
            a state to the same state.
        ignore_end_start_transitions: Whether to ignore the transitions
            that happens from the end of the diary to the start

    Returns:
        transitions_dict, containing the transitions

        * transitions_dict['times']
            the times at which the transitions
            occur, index of where the new state is.

        * transitions_dict['persons']
            the person or hh in the state
            array that performs the transition

        * transitions_dict['new_states']
            state after transition

        * transitions_dict['old_states']
            state before transition

    """
    states = np.array(states).T

    # 1rst state will have no transitions if ignore end start
    old_states = states[0 if ignore_end_start_transitions else -1]

    transition_times = []
    transition_person = []
    transition_new_state = []
    transition_old_state = []

    # iterate over time
    for i, s in enumerate(states):
        # check that the state has really disappeared, and not that it changed
        mask_transition = (
            old_states != s if not include_same_state
            else ~np.isnan(s)
        )
        transition_times.append(np.full(np.sum(mask_transition), i))
        transition_person.append(np.where(mask_transition)[0])
        transition_new_state.append(np.array(s[mask_transition]))
        transition_old_state.append(np.array(old_states[mask_transition]))

        old_states = s

    transitions_dict = {}
    transitions_dict["times"] = np.concatenate(transition_times)
    transitions_dict["persons"] = np.concatenate(transition_person)
    transitions_dict["new_states"] = np.concatenate(transition_new_state)
    transitions_dict["old_states"] = np.concatenate(transition_old_state)

    if return_duration:
        duration = np.zeros_like(transitions_dict["times"])
        for person in np.unique(transitions_dict["persons"]):
            mask_person = transitions_dict["persons"] == person
            person_times = transitions_dict["times"][mask_person]
            # times are already sorted from previous loop
            person_duration = np.roll(person_times, -1) - person_times
            # last state is merge with the first if same, else ignored
            person_last_state = transitions_dict["new_states"][mask_person][-1]
            person_first_state = transitions_dict["new_states"][mask_person][0]
            person_duration[-1] = (
                person_times[0] + (len(states) - person_times[-1])
                if person_first_state == person_last_state
                else 0
            )
            duration[mask_person] = np.array(person_duration)

        transitions_dict["durations"] = np.array(duration)

    return transitions_dict


def states_to_transitions_secondary(
    primary_states: States,
    secondary_states: States,
):
    """Convert a state array to transitions array.

    Args:
        states: The array containing the states. dtype=int or str
        return_duration: Wether to return the duration of the states.
            Defaults to False.

    Returns:
        transitions_dict, containing the transitions

        * transitions_dict['times']
            the times at which the transitions
            occur, index of where the new state is.

        * transitions_dict['persons']
            the person or hh in the state
            array that performs the transition

        * transitions_dict['new_states']
            state after transition

        * transitions_dict['old_states']
            state before transition

    """
    p_states = np.array(primary_states).T
    s_states = np.array(secondary_states).T

    old_p_states = p_states[0]
    old_s_states = s_states[0]

    transition_times = []
    transition_person = []
    transition_new_state = []
    transition_old_state = []

    for i, (p, s) in enumerate(zip(p_states, s_states)):
        # check that the state has really disappeared, and not that it changed
        mask_transition_p = (old_p_states != p) & (old_p_states != s)
        mask_transition_s = (old_s_states != p) & (old_s_states != s)
        transition_times.append(
            np.full(np.sum(mask_transition_p) + np.sum(mask_transition_s), i)
        )
        transition_person.append(np.where(mask_transition_s)[0])
        transition_person.append(np.where(mask_transition_p)[0])
        transition_new_state.append(np.array(p[mask_transition_p]))
        transition_new_state.append(np.array(s[mask_transition_s]))
        transition_old_state.append(np.array(old_p_states[mask_transition_p]))
        transition_old_state.append(np.array(old_s_states[mask_transition_s]))

        old_p_states = p
        old_s_states = s

    transitions_dict = {}
    transitions_dict["times"] = np.concatenate(transition_times)
    transitions_dict["persons"] = np.concatenate(transition_person)
    transitions_dict["new_states"] = np.concatenate(transition_new_state)
    transitions_dict["old_states"] = np.concatenate(transition_old_state)

    return transitions_dict


def group_hh_transitions(
    primary_states: np.ndarray,
    household_indexes: np.ndarray,
    days_indexes: np.ndarray,
    secondary_states: np.ndarray = None,
):
    """Group the states into household transitions.

    Converts the individual states to an household states that
    includes all the persons.

    Args:
        primary_states: The base states array.
        household_indexes: An array containing the household
            corresponding to each diary from the states array.
        days_indexes: Array containing the day index of the diaries
            in the states array.
        secondary_states: Optional array of secondary states.
            If None, no secondary states will be used.
            Defaults to None.

    Returns:
        transitions_dict, containing the transitions

        * transitions_dict['times']
            the times at which the transitions
            occur, index of where the new state is.

        * transitions_dict['households']
            the household from household_indexes

        * transitions_dict['new_states']
            state after transition

        * transitions_dict['old_states']
            state before transition

        * transitions_dict['counts']
            The number of persons in the household that do this transition.

        * transitions_dict['day_nrs']
            The days_indexes of this transition.

    """
    if secondary_states is not None:
        transitions = states_to_transitions_secondary(
            primary_states, secondary_states
        )
    else:
        transitions = states_to_transitions(primary_states)

    # find and counts the transitions in the housholds

    (
        hh_times,
        hh_new_states,
        hh_old_states,
        hh_nrs,
        day_nrs,
    ), hh_counts = np.unique(
        np.array(
            [
                transitions["times"],
                transitions["new_states"],
                transitions["old_states"],
                household_indexes[transitions["persons"]],
                days_indexes[transitions["persons"]],
            ]
        ),
        axis=1,
        return_counts=True,
    )

    hh_transitions = {}
    hh_transitions["times"] = hh_times
    hh_transitions["households"] = hh_nrs
    hh_transitions["new_states"] = hh_new_states
    hh_transitions["old_states"] = hh_old_states
    hh_transitions["counts"] = hh_counts
    hh_transitions["day_nrs"] = day_nrs

    return hh_transitions


def convert_states(raw_states: np.ndarray, merge_dic: dict = None):
    """Convert the states following the rule inscibed in merge_dic.

    Very useful to merge different states into a single one.

    Args:
        raw_states: an array containing the possible states
        merge_dic: dictionary that matches each state to a new state

    Returns:
        merged states: the new states
        u_lab: the label of those new states
    """
    u, inv = np.unique(raw_states, return_inverse=True)
    # get the integer states
    states = inv.reshape(raw_states.shape)

    # merge the activities
    if merge_dic is not None:
        assert isinstance(merge_dic, dict) | isinstance(
            merge_dic, np.ndarray
        ), "merge_dic must be a dictionary"
        to_labels = np.vectorize(lambda x: merge_dic[x])
    else:
        to_labels = np.vectorize(lambda x: x)
    states_labels = to_labels(u)
    # merge the labels together
    u_lab, inv_lab = np.unique(states_labels, return_inverse=True)
    # get the states once merged by category
    merged_states = inv_lab[states]
    return merged_states, u_lab


def translate_1d(array: Union[np.ndarray, list], translating_dict: dict):
    """Translate the values of the array according to the translating dict.

    Args:
        array: list or array of single dimension
        translating_dict: dict mapping values from array to what is returned.

    Returns:
        translated array
    """
    return [translating_dict[element] for element in array]


def make_jsonable(object: Any):
    """Tranforms some objects to ensure they are in json format.

    This is not 100% successful.
    At the moment only tranforms ndarrays to lists.

    Args:
        object: The object to be make jsonable.
    """
    if isinstance(object, dict):
        return {
            make_jsonable(key): make_jsonable(item)
            for key, item in object.items()
        }
    elif isinstance(object, (np.ndarray, list, pandas.Series)):
        return [make_jsonable(elem) for elem in object]
    elif isinstance(object, (str, int, float)):
        return object
    elif isinstance(object, np.generic):
        return object.item()
    else:
        raise NotImplementedError(
            " 'make_jsonable' is not defined for object of type: '{}' ".format(
                type(object)
            )
        )


def remove_spaces(object: Any):
    """Remove spaces from all the strings that can be found.

    This is not 100% successful.
    At the moment only tranforms ndarrays to lists.

    Args:
        object: The object to remove the spaces from it and subobjects.
    """
    if isinstance(object, str):
        return object.replace(" ", "")
    if isinstance(object, dict):
        return {
            remove_spaces(key): remove_spaces(item)
            for key, item in object.items()
        }
    elif isinstance(object, (list, pandas.Series)):
        return [remove_spaces(elem) for elem in object]
    elif isinstance(object, np.ndarray):
        # check if array of strings
        if object.dtype.type is np.string_:
            return remove_spaces(object.tolist)
        else:
            return object
    elif isinstance(object, (np.generic, float, int)):
        return object
    else:
        raise NotImplementedError(
            " 'remove_spaces' is not defined for object of type: '{}' ".format(
                type(object)
            )
        )


def lists_to_numpy_array(object: Any):
    """Tranforms some objects to ensure their list are numpy arrays.

    This is not 100% successful.
    At the moment only tranforms some list to numpy.

    Args:
        object: The object to be make jsonable.
    """
    if isinstance(object, dict):
        return {
            lists_to_numpy_array(key): lists_to_numpy_array(item)
            for key, item in object.items()
        }
    elif isinstance(object, (np.ndarray, list, pandas.Series)):
        return np.asarray(object)
    elif isinstance(object, (str, int, float)):
        return object

    else:
        raise NotImplementedError(
            " 'lists_to_numpy_array' is not defined for object of type: "
            " '{}' ".format(type(object))
        )


def bulbs_stats_from_config(
    config_array: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Transform the bulb config into, bubls and bulbs stats.

    Args:
        config_array: The light config array.

    Returns:
        (mean, std), (consumption, penetration)
    """
    # The light config is A 2-D array, where Dim0 is the different
    # houses and Dim1 the different bulbs of each house.

    mask_is_bulb = (config_array > 0) & (~np.isnan(config_array))

    n_bulbs = np.sum(mask_is_bulb, axis=1)

    consumptions, counts = np.unique(
        config_array[mask_is_bulb], return_counts=True
    )

    return (
        (np.mean(n_bulbs), np.std(n_bulbs)),
        (consumptions, counts/np.sum(counts))
    )


def states_to_tpms(
    states,
    first_tpm_modification_algo='last',
    labels=None
):
    """Convert the states to tranistions probability matrices.

    The output tpms have at time t, the transition from
    t-1 to t.

    Args:
        states: The states to convert.
        first_tpm_modification_algo: Algo to use to change the tpm from
            end of the diary to the start. Defaults to 'last'.
            See: :py:function:`first_tpm_change` .
        labels: Optional labels, (useful if some state are not visited).
            Defaults to None.

    Returns:
        Transition probability matrices.
    """
    states = states.T
    # this matrix will store the transitions probabilites
    tpms = []
    # initialize the first state with the previous state
    old_states = np.array(states[-1])
    n_states = (
        len(labels) if labels is not None
        else int(np.max(states) + 1)
    )  # get the number of states from the input
    for this_states in states:

        # define and counts the transitions
        states_indices, states_counts = np.unique(
            np.asarray((old_states, this_states)),
            axis=1, return_counts=True
        )
        # converts the indexes for accessing the matrix later
        states_indices = [(i) for i in states_indices]

        # compute the sum of the transitions for each states
        transition_matrice = np.full((n_states, n_states), 0)
        transition_matrice[states_indices] = states_counts
        tpms.append(transition_matrice)

        # save the state for old state
        old_states = np.array(this_states)

    # define what we should do with the first matrix that has false transitions
    tpms = first_tpm_change(tpms, algo=first_tpm_modification_algo)

    # converts to probs
    tpms = tpms / np.sum(tpms, axis=2)[:, :, None]
    # set to the same state when there are nan values
    tpms[np.isnan(tpms)] = 0.
    # for the cdfs that have no values, we set unchanging states
    times, rows = np.where(tpms.sum(axis=2) == 0)
    tpms[times, rows, rows] = 1.  # make them stay at the same states

    return np.asarray(tpms)


def first_tpm_change(tpms, algo='nothing'):
    """Change the first tpm based on an algorithm.

    Args:
        algo: The algorithm that should be used to change the first
            tpm values.

            * 'last', replaces the first by the last matrix
            * 'nothing', keeps the same

    Returns:
        The modified tpms.

    """
    # define what we should do with the first matrix that has false transitions
    if algo == 'last':
        tpms[0] = np.array(tpms[-1])
    elif algo == 'nothing':
        pass
    else:
        raise TypeError('Unknown algo {}.'.format(algo))

    return tpms
