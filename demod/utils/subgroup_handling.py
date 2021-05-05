"""Various helpers for subgroups."""

from demod.utils.sim_types import Subgroup
from os.path import join


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
