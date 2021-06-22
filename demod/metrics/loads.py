"""Functions that compute various metrics for load profiles.

A load profile is an array with values equal to the power consumption
at various times.
Load profiles are by convention in demod arrays of size
n_profiles * n_times.
"""

import datetime
from typing import Union
import numpy as np

from scipy.spatial.distance import cdist, pdist


def total_energy_consumed(
    load_profiles: np.ndarray,
    step_size: datetime.timedelta = datetime.timedelta(minutes=1),
) -> np.ndarray:
    """Compute the total energy that was consumed in the profile.

    Sums up all the power comsumed by all time to find the total
    energy.
    Return the energy in Joules.

    Args:
        load_profiles: The load profiles, shape = n_profiles * n_times.
        step_size: the step size used in the load_profiles.
            Defaults to datetime.timedelta(minutes=1).

    Returns:
        total_energy: The total energy that was consumed in the load
            profiles (shape = n_profiles).
    """
    return np.sum(load_profiles, axis=-1) * step_size.total_seconds()


def profiles_similarity(
    simulated_profiles: np.ndarray,
    target_profiles: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Compare similarity of each simulated profiles with target profiles.

    It uses
    `scipy.cdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist>`_
    to compute the distances.
    Any keyword argument given to this function will be
    passed to the scipy function.

    Args:
        simulated_profiles: The simulated profiles.
            array of shape = (n_profiles_sim, n_times)
        target_profiles: The target profiles for the similarity.
            array of shape = (n_profiles_target, n_times)

    Return:
        similarity_metric: The similarity between the profiles, as an
            array of shape = (n_profiles_sim, n_profiles_target)
            where element[i,j] is the comparison of the i-eth simulated
            profile with the j-eth target profile.

    """
    return cdist(simulated_profiles, target_profiles, **kwargs)

def profiles_variety(
    profiles: np.ndarray,
    average: bool = True,
    **kwargs
) -> Union[np.ndarray, float]:
    """Compare variety in the given profiles.

    It uses
    `scipy.pdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.pdist>`_
    to compute the distances.
    Any keyword argument given to this function will be
    passed to the scipy function.

    Args:
        profiles: The profiles which should have their variety assessed.
            array of shape = (n_profiles, n_times)
        average: Whether to average the comparison or to give
            the distance from each profile to every other

    Return:
        similarity_metric: The similarity between the profiles, as an
            array of shape = (n_profiles_sim, n_profiles_target)
            where element[i,j] is the comparison of the i-eth simulated
            profile with the j-eth target profile.

    """
    distances = pdist(profiles, **kwargs)
    return np.mean(distances) if average else distances
