
from datetime import date, datetime
from ..utils.monte_carlo import monte_carlo_from_1d_cdf
from ..utils.distribution_functions import check_valid_cdf
from ..utils import sim_types

from typing import Callable, Union
import os
import numpy as np




OLD_DATASET_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..' + os.sep + '..'))
DATASETS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets'))


def assign_external_array(
        base_array:np.array, external_array:np.array) -> np.array:
    """Overwrite base array with the values from external array.

    If external array has some np.nan values, the np.nan values don't
    override and the value from base_array is used for these values.

    Args:
        base_array : The array to overwrite
        external_array : The overriding array, with nan values where it
            should not override.

    Returns:
        The new array.
    """
    if external_array is None:
        return base_array
    out = np.array(base_array)
    mask_not_nan = ~np.isnan(external_array)
    out[mask_not_nan] = external_array[mask_not_nan]
    return out


def assign_external_dict(base_dict, external_dict):
    """Overwrites the base dictionarry with the
    key values pair from the external dictionary

    Args:
        base_dict (dict): The dictionary to overwrite
        external_dict (dict): The dictionary overwriting

    Returns:
        dict: The base dictionary with the changes
    """

    if external_dict is not None:
        # overwrites the values coming from the external simulator
        for key, value in external_dict.items():
            base_dict[key] = assign_external_array(base_dict[key], value)
    return base_dict



def inherit_getters_docstring(cls):
    """Decorate method to pass the documentation.

    By default, inheritance in python overrides the docstring when you
    overrride the method. This makes the docstring inherited.
    """
    # Finds all the getter in the cls
    public_undocumented_getters = {
        name: getter for name, getter in vars(cls).items()
        if name.startswith('get_') and not getter.__doc__}

    # Check if we find the names in the parents of the class
    for name, getter in public_undocumented_getters.items():
        for parent in cls.mro()[1:]:
            parfunc = getattr(parent, name, None)
            if parfunc and getattr(parfunc, '__doc__', None):
                getter.__doc__ = parfunc.__doc__
                break
    return cls

def sample_population(
        n_samples: int , pdf: sim_types.PDF,
        population_sampling_algo: str = 'real_population', **kwargs
    ) -> np.ndarray:
    """Sample the population based on a pdf.

    Args:
        n_samples: The number of samples to be sampled.
        pdf: The probability of being in the population for each sample.
        population_sampling_algo: algorithm to be used,
            currently implemented:

                - 'real_population':
                    Based on the real population
                - 'monte_carlo':
                    Randomly assign the number based on a MC draw.

            Defaults to 'real_population'.

    Return:
        The number of samples from each value of the pdf.

    Raises:
        ValueError: if the :py:obj:`pdf` does not sum up to one.
    """
    cdf = np.cumsum(pdf)
    check_valid_cdf(cdf)
    if population_sampling_algo == 'real_population':
        lin_space = np.linspace(0, 1, n_samples)
        inds = np.searchsorted(cdf, lin_space)
    elif population_sampling_algo == 'monte_carlo':
        inds = monte_carlo_from_1d_cdf(cdf, n_samples=n_samples)
    else:
        raise NotImplementedError(
            "No implementation satifies algorithm '{}".format(
                population_sampling_algo)
            )

    counts = np.bincount(inds, minlength=len(cdf))

    return counts
    # if no algo is known



