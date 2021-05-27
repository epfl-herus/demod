"""
Data loader for the german TOU survey.
"""
from datetime import time
import os
from typing import Tuple
import numpy as np

from ..tou_loader import LoaderTOU
from ..base_loader import PopulationLoader
from ..DESTATIS.loader import Destatis
from ...utils.sim_types import *
from ...utils.monte_carlo import PDFs
from ...utils.sparse import SparseTPM


class GTOU(LoaderTOU, PopulationLoader):
    """German Time-Of-Use Survey dataset loader.

    This loads data for different types of activity models.
    It can also split the households in different subgroups of the whole
    population.

    Currently implements activity_types:
        - 'Sparse9States'
        - '4_States'
        - 'DemodActivities_0'
        - 'Bottaccioli2018' https://doi.org/10.1109/ACCESS.2018.2886201

    """

    DATASET_NAME = 'GermanTOU'
    refresh_time = time(4, 0, 0)

    def _parse_tpm(self, subgroup: Subgroup):
        if self.activity_type == '4_States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_4states
            return get_data_4states(subgroup)
        elif self.activity_type == 'DemodActivities_0':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_activity
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='last',
                add_away_state=True,
                add_durations=False,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_Bottaccioli_act
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_Bottaccioli_act,
                first_tpm_modification_algo='last',
                add_away_state=True,
                add_durations=False,
            )
        else:
            err = NotImplementedError(("No parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

    def _parse_tpm_with_duration(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, np.ndarray, StateLabels, PDF, PDFs, dict]:
        if self.activity_type == '4_States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_4states
            return get_data_4states(subgroup, add_durations=True)
        elif self.activity_type == 'DemodActivities_0':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_activity
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='nothing',
                add_away_state=True,
                add_durations=True,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_Bottaccioli_act
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_Bottaccioli_act,
                first_tpm_modification_algo='nothing',
                add_away_state=True,
                add_durations=True,
            )
        else:
            err = NotImplementedError(("No parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

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

    def _parse_activity_profiles(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        from .parser import create_data_activity_profile
        return create_data_activity_profile(subgroup)

    def load_population_subgroups(
        self, population_type: str
    ) -> Tuple[Subgroups, List[float], int]:
        # GTOU is the same as Germany so we can use Destatis for the population
        data = Destatis()
        return data.load_population_subgroups(population_type)
