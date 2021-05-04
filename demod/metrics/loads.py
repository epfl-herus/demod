"""Functions that compute various metrics for load profiles.

A load profile is an array with values equal to the power consumption
at various times.
Load profiles are by convention in demod arrays of size
n_profiles * n_times.
"""

import datetime
import numpy as np


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
