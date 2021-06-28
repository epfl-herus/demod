"""Various helpers for subgroups.

More information can be found about the possible values for
:py:class:`~demod.utils.cards_doc.Params.subgroup`.
"""
from __future__ import annotations
import datetime

from typing import List, Tuple
import warnings

from .sim_types import Subgroups


class Subgroup(dict):
    """Represent a subgroup of the population.

    Addition to a dict is that it can be sorted.

    Added in v_0.2. Previously dict where used, which should be fine
    for compatibility in general.
    """

    def __gt__(self, other):
        """Implements this to be used by np.unique.

        The number of keys is the 1rst criteria, a larger number is more.
        The 2nd is by taking each key in a sorted order. If the first key
        exist in only one dict, it will be greater.
        Then if the keys are in both, 3rd is the value corresponding to
        the key.
        """
        if not isinstance(other, dict):
            raise TypeError(
                "'<' not supported between instances of '{}' and '{}'".format(
                    type(self), type(other)
                )
            )
        if len(other) > len(self):
            return True
        elif len(other) < len(self):
            return False
        else:
            for key in sorted(list(self.keys()) + list(other.keys())):
                if key not in other:
                    return True
                if key not in self:
                    return False
                if other[key] > self[key]:
                    return True
                if other[key] < self[key]:
                    return False
        return False

    def copy(self) -> Subgroup:
        """Returns a new Subgroup instead of dict."""
        return Subgroup(self)

def sort_subgroup(subgroup):
    """Sort the key values of a subgroup.

    Args:
        subgroup: subgroup to sort.

    Returns:
        sorted_subgroup
    """
    return {
        key: val
        for key, val in sorted(subgroup.items(), key=lambda ele: ele[0])
    }


def subgroup_string(subgroup):
    """Transform a subgroup to a string.

    The subgroup is sorted before transformation using
    :py:func:`~.sort_subgroup`.
    If any subgroup value is a :py:obj:`list`, it is extracted in the
    string.

    Args:
        subgroup: the subgroup to transform

    Returns:
        subgroup_string: The string matching the subgroup
    """
    # sort by key
    sorted_subgoup = sort_subgroup(subgroup)
    return "__".join(
        [
            str(i) + "_" + ",".join([str(k) for k in j])
            if isinstance(j, list)
            else str(i) + "_" + str(j)
            for i, j in sorted_subgoup.items()
        ]
    )


def remove_time_attributues(subgroup: Subgroup) -> Subgroup:
    """Return a new subgroup withtout the time attributes.

    Removes 'quarter', 'weekday', ...
    TODO: add all the ones that need removed

    Args:
        subgroup: The original subgroup
    Returns:
        The subgroup without time attributes.
    """
    new_subgroup = subgroup.copy()
    if "weekday" in new_subgroup:
        new_subgroup.pop("weekday")
    if "quarter" in new_subgroup:
        new_subgroup.pop("quarter")

    return new_subgroup


def check_weekend_day_format(subgroup: Subgroup):
    """Check the subgroup is weekday or weekend.

    subgroup['weekday'] must be [1, 2, 3, 4, 5] or [6, 7]
    which means either weekdays or weekends.

    Args:
        subgroup: the subgroup to be checked

    Raises:
        ValueError: If the subgroup is not weekend or weekday
    """
    if subgroup["weekday"] not in [[1, 2, 3, 4, 5], [6, 7]]:
        raise ValueError(
            " 'subgroup' is not weekdays or weekend format"
            " 'subgroup['weekday']' is {}, should be '[1, 2, 3, 4, 5]'"
            " or '[6, 7]' of type 'list' "
        )


def is_weekend(subgroup: Subgroup) -> bool:
    """Check that the subgroup correspond to the weekend definition.

    Args:
        subgroup: the subgroup to be checked

    Returns:
        whether subgroup is weekend
    """
    return subgroup["weekday"] == [6, 7]


def is_weekday(subgroup: Subgroup) -> bool:
    """Check that the subgroup correspond to the weekday definition.

    Args:
        subgroup: the subgroup to be checked

    Returns:
        whether subgroup is weekday
    """
    return subgroup["weekday"] == [1, 2, 3, 4, 5]


def subgroup_households_to_persons(
    hh_subgroups: Tuple[Subgroups, Subgroup],
) -> List[List[Subgroup]]:
    """Convert households subgroups to person subgroups.

    For each household subgroup given as input, return a list
    of all the persons in this household.

    More information can be found about
    :py:class:`~demod.utils.cards_doc.Params.subgroup`.

    Added in v_0.2.

    .. warning::
        This function migth not be suited to any kind of subgroups.
        It is designed to work for subgroups based on 'n_residents'
        or 'household_type'.
        Also this was designed for the
        :py:class:`~demod.datasets.GermanTOU.loader.GTOU` dataset.


    Args:
        hh_subgroups: The households subgroups that need to be
            converted. If a single subgroup is given, a single
            list with the person subgroups is returned.
        data: The dataset to use for the conversion.

    Returns:
        person_subgroups: The converted subgroups, as a list of person
            subgroups.
    """
    if isinstance(hh_subgroups, list):
        return [subgroup_households_to_persons(sg) for sg in hh_subgroups]

    hh_subgroup = Subgroup(hh_subgroups.copy())

    # Gets the households only relevant keys
    n_residents = hh_subgroup.pop('n_residents', None)
    hh_type = hh_subgroup.pop('household_type', None)

    if hh_type == 1 or n_residents == 1:  # Single person households
        if ('age' not in hh_subgroup) and ('hh_mean_age' in hh_subgroup):
            # Uses the mean age as age for the person
            hh_subgroup['age'] = hh_subgroup.pop('hh_mean_age')

        return [hh_subgroup]

    elif hh_type == 2:  # Couple without kid
        hh_subgroup_1, hh_subgroup_2 = hh_subgroup.copy(), hh_subgroup.copy()

        hh_subgroup_1['household_position'] = 1  # Main income
        hh_subgroup_2['household_position'] = 2  # Spouse or other

        return [hh_subgroup_1, hh_subgroup_2]

    elif hh_type == 3:
        # Single Parent with at least one kid under 18 and the other under 27
        subgroup_parent = hh_subgroup.copy()
        subgroup_child = hh_subgroup.copy()
        subgroup_parent['household_position'] = 1  # Main income
        subgroup_child['household_position'] = 3  # Kid
        return (
            [subgroup_parent] +
            [subgroup_child for _ in range(n_residents - 1)]
        )
    elif hh_type == 4:
        # Single Parent with at least one kid under 18 and the other under 27
        subgroup_parent = hh_subgroup.copy()
        subgroup_parent2 = hh_subgroup.copy()
        subgroup_child = hh_subgroup.copy()
        subgroup_parent['household_position'] = 1  # Main income
        subgroup_parent2['household_position'] = 2  # Spouse or other
        subgroup_child['household_position'] = 3  # Kid
        return (
            [subgroup_parent] + [subgroup_parent2] +
            [subgroup_child for _ in range(n_residents - 2)]
        )

    else:
        # Assume they are all the same
        warnings.warn((
            'subgroup_households_to_persons() is cannot convert hh subgroup'
            ' {} to person subgroup. It creates residents with the'
            ' from the same subgroup. '
        ).format(hh_subgroup))
        return [hh_subgroup for _ in range(n_residents)]


def add_time(
    subgroup: Subgroup, desired_datetime: datetime.datetime,
    use_week_ends_days: bool = True,
    use_week_sat_sun: bool = False,
    use_7days: bool = False,
    use_quarters: bool = False,
) -> Subgroup:
    """Add a time component to the subgroup depending on the parameters.

    Added in v_0.2.

    Args:
        use_week_ends_days: Distinguish the subgroups between
            weekdays and weekends. Defaults to True.
        use_week_sat_sun: Distinguish the subgroups between
            weekdays, saturday and sunday. Defaults to False.
        use_7days: Distinguish the subgroup between the 7 days of
            the weeks. Defaults to False.
        use_quarters: Distinguish the subgroup depending on the
            quarters of a year. Defaults to False.
    """
    if sum((use_7days, use_week_ends_days, use_week_sat_sun)) > 1:
        raise ValueError(
            "Can use only one of 'use_7days', 'use_week_ends_days',"
            " 'use_week_sat_sun' "
        )
    elif use_week_ends_days:
        if desired_datetime.isoweekday() <= 5:
            subgroup["weekday"] = [1, 2, 3, 4, 5]
        else:
            subgroup["weekday"] = [6, 7]
    elif use_week_sat_sun:
        if desired_datetime.isoweekday() <= 5:
            subgroup["weekday"] = [1, 2, 3, 4, 5]
        else:
            subgroup["weekday"] = desired_datetime.isoweekday()
    elif use_7days:
        subgroup["weekday"] = desired_datetime.isoweekday()
    else:  # No type of day was specifiy
        subgroup.pop("weekday", None)

    # update the quarter if required
    if use_quarters:
        subgroup["quarter"] = (desired_datetime.month - 1) // 3 + 1
    else:
        # Makes sure quarter is not in here
        subgroup.pop("quarter", None)

    return subgroup
