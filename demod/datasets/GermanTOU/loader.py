"""
Data loader for the german TOU survey.
"""
from datetime import time
import os
from typing import Tuple
import numpy as np

from ..tou_loader import LoaderTOU
from ...utils.sim_types import *
from ...utils.sparse import SparseTPM


class GTOU(LoaderTOU):
    """German Time-Of-Use Survey dataset loader.

    This loads data for different types of activity models.
    It can also split the households in different subgroups of the whole
    population.

    Currently implements activity_types:
        - 'Sparse9States'

    """

    DATASET_NAME = 'GermanTOU'
    refresh_time = time(4, 0, 0)
    def _parse_sparse_tpm(self, subgroup: Subgroup) -> Tuple[
        SparseTPM, StateLabels,
        ActivityLabels, np.ndarray]:

        if self.activity_type == 'Sparse9States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_sparse9states
            return get_data_sparse9states(subgroup)
        else:
            err = NotImplementedError(("No sparse parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

    def _parse_activity_profiles(self, subgroup: Subgroup) -> Dict[str, np.ndarray]:
        from .parser import create_data_activity_profile
        return create_data_activity_profile(subgroup)