"""Helpers for the appliances."""

from typing import Dict, List, Union
import warnings

import numpy as np
from demod.utils.sim_types import AppliancesDict

def merge_appliance_dict(
    dic1: AppliancesDict, dic2: AppliancesDict
) -> AppliancesDict:
    """Merges two appiance dictionaries together.

    They must have the same keys, or the keys that
    are not in both will be dropped.

    """
    special_keys = ['number']
    dict_merged = {}
    for key, item in dic1.items():
        if key in special_keys:
            continue
        # Check the key is also in 2 or drop
        if key in dic2:
            dict_merged[key] = np.append(item, dic2[key])
    # Sums the number of appliances
    dict_merged['number'] = dic1['number'] + dic2['number']
    return dict_merged


def remove_start(appliance_type: str) -> Union[bool, str]:
    """Remove the first element to get the parent appliance type.

    Args:
        appliance_type: The type of the appliance to find in types_list

    Return:
        The new name (without the start)
        or False if there is no start to remove.
    """
    parts = appliance_type.split("_")
    if len(parts) > 1:
        new_key = "_".join(parts[1:])
        return new_key
    else:
        return False


def find_closest_type(
    appliance_type: str, types_list: str
) -> Union[bool, str]:
    """Return the closest type in the list.

    Check the parented types.

    Args:
        appliance_type: The type of the appliance to find in types_list
        types_list: List of appliances types

    Return:
        The closest parent type or False if no parent was found.
    """
    temp_type = appliance_type
    while temp_type:
        if temp_type in types_list:
            return temp_type
        temp_type = remove_start(temp_type)
    return False


def get_ownership_from_dict(
    appliances_dict: AppliancesDict, ownership_dict: Dict[str, float]
) -> np.array:
    """Calculate the ownership of each appliance.

    Based on the ownership dictionary, detect the ownership of the
    appliances.
    The :py:obj:`appliance_dict['type']` is used to find the
    corresponding ownership.

    When two appliances of the same type are given, it will try to check
    the ownership for a second appliance of that type, and so on for
    more.

    If an appliance type is not found, it will check for a parent
    appliance, by removing the beggining of the appliance type.
    ex: chest_freezer is not found in ownership_dict,
    look for freezer instead.

    The two preceeding instructions can be combined, example::

        appliances_dict['type'] = ['hob', 'electric_hob']
        ownership_dict = {'hob': 0.9 , 'hob_2': 0.1]
        output = [0.9, 0.1]

    In the future, other algorithms could be used, as example using
    correlations of ownership. (you don't have a tv box if you have no
    tv)

    Args:
        appliances_dict: Dictonarry of the appliances.
        ownership_dict: Mapping
            :py:attr:`~demod.utils.cards_doc.Params.appliance_type`
            to a probability.

    Returns:
        np.array: The probability of ownership for each appliance.
    """

    def ensure_key_in_ownership(key: str):
        if find_closest_type(key, ownership_dict) is False:
            err_msg = (
                "appliance_type: '{}' from 'appliance_dict' cannot "
                "be found in the 'ownership_dict' with keys : '{}'."
            ).format(key, ownership_dict.keys())
            raise ValueError(err_msg)

    counter = {}
    pdf = []

    for i, key_name in enumerate(appliances_dict["type"]):

        try:  # Try to find the appliance type in the ownership dict
            ensure_key_in_ownership(key_name)
            # Get the closest type present in this dataset
            closest_type = find_closest_type(key_name, ownership_dict)
            app_type = closest_type

        except ValueError as val_err:
            warnings.warn(
                "Could not find the appliance type: '{}'. \n".format(key_name)
                + "This is due to an err with message: '{}'. \n".format(
                    val_err)
                + "Default values from appliances_dict['equipped_prob'] will "
                "be used instead."
            )
            prob = appliances_dict['equipped_prob'][i]

        else:  # If no exception is raised
            this_number = counter.get(app_type, 0)
            # Counts the i-eth occurence of this appliance in dict.
            if this_number == 0:
                # First occurance
                counter[app_type] = 1
                key_name = app_type
            else:
                # Multiple occurances
                counter[app_type] += 1
                key_name = app_type + "_" + str(counter[app_type])

            if key_name in ownership_dict:
                prob = ownership_dict[key_name]
            else:  # If not in, means that the i-eth occuance is to large
                prob = 0.

        finally:
            pdf.append(prob)

    return np.array(pdf)


def assign_ownership_from_prob1_and_number(
    prob_1: float, number: float, algo: str = 'floor'
) -> List[float]:
    r"""Assign the probability of owning multiple sample of appliances.

    You can choose different assignement algorithms.
    (Only one is currently implemented)

    :'floor':
        :math:`p_0 = prob_1`, :math:`n = number`
        and :math:`\sum_{i}^{} p_i = n`.
        which produces results like::

            [prob_1, 1., 1., n - i - prob_1]


    Args:
        prob_1: The probability of owning 1 of the appliance
        number: The average number of appliances owned by the hosehold
        algo: The algorithm name to use

    Returns:
        A variable length list, where the element i in the list
        corresponds to the prob of owning
        a i-eth copy of the appliance.
    """
    if number is np.nan:
        number = prob_1
    if (prob_1 > number) or (prob_1 < 0) or (prob_1 > 1) or (number < 0):
        raise ValueError(
                'Impossible values or, the probability of owning an'
                "appliance {} is greater thant the number owned {}.".format(
                    prob_1, number
                ))
    if algo == 'floor':
        probs = []
        # Assign ownership of the appliance
        prob = prob_1
        probs.append(prob)
        probs.append(min(number - prob, 1.))

        remaining_n = number - prob
        while remaining_n > 1:
            remaining_n -= 1.
            probs.append(min(remaining_n, 1.))

        return probs

    else:
        raise ValueError('Unkown value for agrument "algo" : {}.'.format(algo))


def get_target_from_dict(
    appliances_dict: AppliancesDict,
    target_dict: Dict[str, float],
    default_key: str = 'target_switchons'
) -> np.array:
    """Calculate the targets of each appliance.

    Based on the target_dict, find the target of the
    appliances from the app_dict.
    The :py:obj:`appliance_dict['type']` is used to find the
    corresponding target.
    The target can be any kind of target
    (switchons, consumption, duration, ...)

    When two appliances of the same type are given, it will try to check
    the target for a second appliance of that type, and so on for
    more.

    If an appliance type is not found, it will check for a parent
    appliance, by removing the beggining of the appliance type.
    ex: chest_freezer is not found in target_dict,
    look for freezer instead.

    The two preceeding instructions can be combined, example::

        appliances_dict['type'] = ['hob', 'electric_hob']
        ownership_dict = {'hob': 0.9 , 'hob_2': 0.1}
        output = [0.9, 0.1]

    Args:
        appliances_dict: Dictonary of the appliances.
        target_dict: Mapping
            :py:attr:`~demod.utils.cards_doc.Params.appliance_type`
            to a number.
        default_key: The key that is used for the target value in
            the appliances_dict.

    Returns:
        np.array: The target of each appliances.
    """

    def ensure_key_in_target_dict(key: str):
        if find_closest_type(key, target_dict) is False:
            err_msg = (
                "appliance_type: '{}' from 'appliance_dict' cannot "
                "be found in the 'target_dict' with keys : '{}'."
            ).format(key, target_dict.keys())
            raise ValueError(err_msg)

    counter = {}
    targets = []

    for i, key_name in enumerate(appliances_dict["type"]):

        try:  # Try to find the appliance type in the ownership dict
            ensure_key_in_target_dict(key_name)
            # Get the closest type present in this dataset
            closest_type = find_closest_type(key_name, target_dict)
            app_type = closest_type

        except ValueError as val_err:
            warnings.warn(
                "Could not find the appliance type: '{}'. \n".format(key_name)
                + "This is due to an err with message: '{}'. \n".format(
                    val_err)
                + "Default target values from appliance_dict['{}'] will "
                "be used instead.".format(default_key)
            )
            target_value = appliances_dict[default_key][i]

        else:  # If no exception is raised
            this_number = counter.get(app_type, 0)
            # Counts the i-eth occurence of this appliance in dict.
            if this_number == 0:
                # First occurance
                counter[app_type] = 1
                key_name = app_type
            else:
                # Multiple occurances
                counter[app_type] += 1
                key_name = app_type + "_" + str(counter[app_type])

            if key_name in target_dict:
                target_value = target_dict[key_name]
            else:  # If not in, means that the i-eth occuance is to large
                target_value = 0.

        finally:
            targets.append(target_value)

    return np.array(targets)
