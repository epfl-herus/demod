"""
Data loader for the german TOU survey.
"""
from datetime import time
from demod.utils.parse_helpers import convert_states, states_to_transitions
from demod.datasets.GermanTOU.parser import get_mask_subgroup
import os
from typing import Tuple
import numpy as np

from ..tou_loader import LoaderTOU
from ..base_loader import PopulationLoader
from ..DESTATIS.loader import Destatis
from ...utils.sim_types import *
from ...utils.monte_carlo import PDFs
from ...utils.sparse import SparseTPM
from ...metrics.states import get_durations_by_states
from .parser import (
    get_tpms_activity, GTOU_label_to_Bottaccioli_act, GTOU_label_to_activity,
    primary_states, occ
)


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

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(*args, **kwargs)
        # Define a dictionary of activities
        corresponding_dict = {
            'DemodActivities_0': GTOU_label_to_activity,
            'Bottaccioli2018': GTOU_label_to_Bottaccioli_act,
        }
        # Adds a states called 'away' when people are not at home
        add_away_state = {
            'DemodActivities_0': True,
            'Bottaccioli2018': True,
        }
        # Whether to replace 'other' activities by secondary states when possible
        use_secondary_states = {
            'DemodActivities_0': True,
            'Bottaccioli2018': True,
        }
        if self.activity_type in corresponding_dict:
            self.corresponding_dict = corresponding_dict[self.activity_type]
            self.add_away_state = add_away_state[self.activity_type]
            self.use_secondary_states = use_secondary_states[self.activity_type]
        else:
            self.corresponding_dict = None
            self.add_away_state = False
            self.use_secondary_states = False


    def _parse_tpm(self, subgroup: Subgroup):
        if self.activity_type == '4_States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_4states
            return get_data_4states(subgroup)
        elif self.activity_type == 'DemodActivities_0':
            # Imports the raw data parser only here to save time
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='last',
                add_away_state=True,
                add_durations=False,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
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
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='nothing',
                add_away_state=True,
                add_durations=True,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
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

    def get_states(
        self, subgroup: Subgroup
    ) -> np.ndarray:
        """Return the states from the data, labelled using activity dict.

        The activity dict is specifies in self.__init__()
        Also adds away state and the secodary states if it was specify
        in self.__init__()
        """

        if self.corresponding_dict is None:
            raise ValueError((
                "No activity dictionary was specifiy for activity type {}."
                "You can define on in the __init__ method of {}."
            ).format(self.activity_type, self))

        mask_subgroup = get_mask_subgroup(**subgroup)
        raw_states = primary_states[mask_subgroup]


        if self.add_away_state:
            # Away is the opposite of occupancy
            raw_states[~occ[mask_subgroup]] = 0
            activity_dict = self.corresponding_dict.copy()
            activity_dict[0] = 'away'


        states, states_label = convert_states(raw_states, activity_dict)
        # Makes sure '-' are replaced by 'other'
        states_label[states_label=='-'] = 'other'

        if self.use_secondary_states:
            # Gets the secondary states
            raw_sec_states = primary_states[mask_subgroup]
            sec_states, sec_states_label = convert_states(
                raw_sec_states, activity_dict
            )
            sec_states_label[sec_states_label=='-'] = 'other'
            # Find where to replace
            mask_can_replace = (
                (states_label[states] == 'other')
                & (sec_states_label[sec_states] != 'other')
            )
            # Recontruct a states array with the secondary states
            raw_act_array = states_label[states]
            raw_act_array[mask_can_replace] = sec_states_label[sec_states][mask_can_replace]
            # Finds back in states format
            states, states_label = convert_states(raw_act_array)


        # Adds missing activities from dict to states labels
        all_activities = np.array(list(activity_dict.values()))
        missing_act = all_activities[~np.isin(all_activities, states_label)]
        states_label = np.concatenate((states_label, missing_act))

        return states_label[states]

    def _parse_activity_duration(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:

        if 'n_residents' in subgroup:
            if subgroup['n_residents'] > 1:
                raise NotImplementedError(
            'Not implemented for households subgroups'
            )
        durations = get_durations_by_states(self.get_states(subgroup))
        return {
            # Makes a pdf from the counts of durations
            act: np.bincount(durs).astype(float) / len(durs)
            for act, durs in durations.items()
        }

    def _parse_daily_activity_starts(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        if 'n_residents' in subgroup:
            if subgroup['n_residents'] > 1:
                raise NotImplementedError(
            'Not implemented for households subgroups'
            )
        states = self.get_states(subgroup)
        transitions_dict = states_to_transitions(states)

        return {
            act: np.bincount(  # Counts the number of persons starting i times
                    np.bincount(  # Counts how many time each person start act
                        transitions_dict['persons'][
                            transitions_dict['new_states'] == act
                        ]
                    )
                ).astype(float) / len(states)  # Convert to probs
            for act in np.unique(transitions_dict['new_states'])
        }

    def _parse_activity_probabilities(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        states = self.get_states(subgroup)
        return {
            s: (states == s).mean(axis=0)
            for s in np.unique(states)
        }
