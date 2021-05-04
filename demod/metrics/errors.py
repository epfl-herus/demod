"""Error metrics functions.

Implementation of different error metrics.
"""
import numpy as np


def RMSE(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the Root Mean Squared error between x1 and x2.

    `RMSE on Wikipedia
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_

    Args:
        x1: array 1
        x2: array 2, must have the same shape as x1

    Returns:
        rmse: The RMSE between x1 and x2.
    """
    return np.sqrt(np.mean((x1 - x2)**2))
