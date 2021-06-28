"""Functions that compute various metrics for load profiles.

A load profile is an array with values equal to the power consumption
at various times.
Load profiles are by convention in demod arrays of size
*n_profiles* * *n_times*. Where *n_profiles* is the number of load profiles
and *n_times* is the number of observations during the time, usually
with a constant step size.
"""

import datetime
from typing import Tuple, Union
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
        **total_energy**, The total energy that was consumed in the load
        profiles (shape = n_profiles).
    """
    return np.sum(load_profiles, axis=-1) * step_size.total_seconds()


def profiles_similarity(
    simulated_profiles: np.ndarray,
    target_profiles: np.ndarray,
    *args, **kwargs
) -> np.ndarray:
    """Compare similarity of each simulated profiles with target profiles.

    It uses
    `scipy.cdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist>`_
    to compute the distances.
    Any args, or kwargs given to this function will be
    passed to the scipy function.

    Args:
        simulated_profiles: The simulated profiles.
            array of shape = (n_profiles_sim, n_times)
        target_profiles: The target profiles for the similarity.
            array of shape = (n_profiles_target, n_times)

    Return:
        **similarity_metric**, The similarity between the profiles, as an
        array of shape = (n_profiles_sim, n_profiles_target)
        where element[i,j] is the comparison of the i-eth simulated
        profile with the j-eth target profile.

    """
    return cdist(simulated_profiles, target_profiles, *args, **kwargs)


def profiles_variety(
    profiles: np.ndarray,
    average: bool = True,
    *args, **kwargs
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
        **variety_metric**, The variety between the profiles, as an
        array of Shape = (n_profiles, n_profiles)
        where element[i,j] is the comparison of the i-eth
        profile with the j-eth profile.

    """
    distances = pdist(profiles, *args, **kwargs)
    return np.mean(distances) if average else distances


def diversity_factor(profiles: np.ndarray):
    """Compute the diversity factor for the given load profiles.

    The diversity factor is the ratio of the sum of the individual
    non-coincident maximum loads
    to the maximum demand of the aggregated loads.
    `More on wiki <https://en.wikipedia.org/wiki/Diversity_factor>`_ .

    Args:
        profiles: The load profiles. Shape = (n_profiles, n_times).
    """
    max_loads = np.max(profiles, axis=1)
    aggregated_loads = np.sum(profiles, axis=0)
    return np.sum(max_loads) / np.max(aggregated_loads)


def coincidence_factor(profiles: np.ndarray):
    """Compute the simultaneity factor for the given load profiles.

    The simultaneity factor is  the inverse of the
    :py:func:`.diversity_factor`.

    Args:
        profiles: The load profiles. Shape = (n_profiles, n_times).
    """
    return 1. / diversity_factor(profiles)


def simultaneity_factor(profiles: np.ndarray):
    """Compute the simultaneity factor for the given load profiles.

    The simultaneity factor is the same as the coincidence factor.
    :py:func:`.coincidence_factor`.

    Args:
        profiles: The load profiles. Shape = (n_profiles, n_times).
    """
    return coincidence_factor(profiles)


def cumulative_changes_in_demand(
    profiles: np.ndarray,
    bins: int = 100,
    normalize: bool = True,
    bin_edges: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the distribution of the changed in demand.

    Can be used as a measure of the 'spikiness' or 'volatility' of the loads.
    The distribution correspond to how often the load varies during the days.

    Args:
        profiles: The load profiles which are used to compute
            the demand. shape = (n_profiles, n_times).
        bins: The number of intervals to used for the cumulative
            distribution.
        normalize: Whether to normalize the demand by the maximum demand.
        bin_edges: If specified, these edges will be used instead.
            When using this, make sure that if
            kwarg :py:obj:`normalize` is True, the edges scale from 0 to 1.

    Returns:
        * **cdf**, array with the  values of the histogram.
          The value of cdf[i] corespond to the values between bin_edges[i] and
          bin_edges[i+1].
        * **bin_edges**, array of dtype float
          Return the bin edges (length(hist)+1).

    """
    # Take the absolute changes and remove the last change
    changes = np.abs(profiles - np.roll(profiles, -1, axis=1))[:, :-1]

    # Include all changes in the same array
    changes = changes.reshape(-1)

    max_change = np.max(changes)

    if normalize:
        changes /= max_change
    if bin_edges is None:
        # Use linearly spaced bin_edges
        hist, bin_edges = np.histogram(
            changes, bins=bins, range=(0, max_change)
        )
    else:
        # Use the user specified bin_edges
        if max_change >= bin_edges[-1]:
            raise ValueError((
                "bin_edges largest value is {}, which is too small for the "
                "largest value change {}. (must be striclty greater)."
            ).format(bin_edges[-1], max_change))
        if np.min(changes) < bin_edges[0]:
            raise ValueError((
                "bin_edges smallest value is {}, which is too large for the "
                "smallest value change {}. (must be smaller or equal)."
            ).format(bin_edges[0], np.min(changes)))
        # Counts where the transtions are in the bin_edges
        hist = np.bincount(
            np.searchsorted(bin_edges, changes)
        )[1:]
    # Convert to cdf
    return np.cumsum(hist/np.sum(hist)), bin_edges


def time_coincident_demand(profiles: np.ndarray) -> float:
    """Return the time coincident demand for the given profiles.

    The maximum demand per household during the whole time.

    For large number of profiles, this metric is equal to
    the *after diversity maximum demand* (ADMD) , as it is
    is "the maximum demand, per customer, as the number of
    customers connected to the network approaches infinity." [McQueen2004]_

    Args:
        profiles: The load profiles which are used to compute
            the demand. shape = (n_profiles, n_times).
    """
    return np.max(np.mean(profiles, axis=0))


def load_duration(
    profiles: np.ndarray,
    loads: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray] :
    """Compute the load duration of the given profiles.

    The outputs can be used to build the
    `load duration curve <https://en.wikipedia.org/wiki/Load_duration_curve>`_
    .

    *n_loads*, the number of load thresholds returned  is either
    given by :py:obj:`loads`
    or by the number of unique loads in profiles

    Args:
        profiles: The load profiles which are used to compute
            the demand. shape = (n_profiles, n_times).
        loads: The different loads thresholds at which the durations
            should be computed. size=n_loads

    Returns:
        * **loads**, Array of the loads (sorted), size = (n_loads)
        * **durations**, The durations correponding to the loads (in steps).
          size = (n_profiles, n_loads)
    """
    if loads is None:
        loads = np.sort(np.unique(profiles))
    else:
        loads = np.sort(loads)

    if len(profiles.shape) == 1:
        # One dimensional input gets rearranged
        profiles.reshape((1, -1))

    # First finds where the load is higher than loads
    mask_high = profiles[:, :, np.newaxis] >= loads[np.newaxis, np.newaxis, :]
    # Finds the proportion of time it is on
    durations = np.sum(mask_high, axis=1)
    return loads, durations
