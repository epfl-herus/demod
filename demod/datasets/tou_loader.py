from datetime import datetime, time
from demod.utils.subgroup_handling import subgroup_string
import os
import json
from typing import Any, Tuple

import numpy as np

from .base_loader import DatasetLoader
from ..utils.sim_types import *
from ..utils.sparse import SparseTPM



class LoaderTOU(DatasetLoader):
    """Loader for Time Of Use survey data.

    Supports for different kind of activity parsing.

    Attributes:
        activity_type: the name of the activity loaded
        refresh_time: a time object specifying at which times the TOU
            is refreshed.

    """
    activity_type: str
    refresh_time: time

    def __init__(
            self, activity_type: str = '4_States',
            /, **kwargs) -> Any:
        super().__init__(**kwargs)

        # Adds the activity type to the path
        self.parsed_path_activity = os.path.join(
            self.parsed_path, activity_type)
        self.activity_type = activity_type
        # make the dir of the parsed path
        self._check_make_parsed_dir()
        # make the dir of the parsed path activity type
        if not os.path.exists(self.parsed_path_activity):
            os.mkdir(self.parsed_path_activity)

    def load_tpm(self, subgroup: Subgroup):
        """Loads a transition probability matrix for the requested
        subgroup.

        Args:
            subgroup: The desired subgroup

        Returns:
            the transition probability matrix, the labels and the initial pdf
        """
        raise NotImplementedError()

    def load_sparse_tpm(self, subgroup: Subgroup) -> Tuple[
            SparseTPM, StateLabels,
            ActivityLabels, np.ndarray]:
        """Loads a sparse transition probability matrix

        This can be used as data input by any
        :py:class:`SparseStatesSimulator`.

        Args:
            subgroup: requested suubgroup TPM

        Returns:
            sparse_TPM,
            states_labels,
            activity_labels,
            initial_pdf,
        """
        subgroup_str = subgroup_string(subgroup)

        file_path = os.path.join(self.parsed_path_activity, subgroup_str)
        file_name = os.path.join(self.activity_type, subgroup_str)
        try:
            # load the tpm

            sparse_tpm = SparseTPM.load(file_path)

            parsed = self._load_parsed_data(file_name)
            labels, activity_labels, initial_pdf = parsed

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            sparse_tpm, labels, activity_labels, initial_pdf, legend \
                = self._parse_sparse_tpm(subgroup)

            # saves the data
            sparse_tpm.save(file_path)
            parsed = (labels, activity_labels, initial_pdf)
            self._save_parsed_data(
                file_name,
                parsed, npz=True
            )
            # Saves a legend of the files created
            legend_path = (
                self.parsed_path_activity + os.sep
                + subgroup_str + '_dict_legend.json'
            )
            with open(legend_path, 'w') as fp:
                json.dump(legend, fp, indent=4)

        return sparse_tpm, labels, activity_labels, initial_pdf

    def _parse_sparse_tpm(self, subgroup: Subgroup) -> Tuple[
            SparseTPM, StateLabels,
            ActivityLabels, np.ndarray, dict]:
        """Abstract parsing method.
        You need to implement here the logic for parsing from raw data.

        Args:
            selfsubgroup: The subgroup to be parsed

        Returns:
            sparse_TPM,
            states_labels,
            activity_labels,
            initial_pdf,
            legend,
        """
        raise NotImplementedError()

    def load_activity_probability_profiles(
            self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        """Return the activity probability profiles for a subgroup.

        This can be used by an
        :py:class:`~demod.simulators.appliance_simulators.ApplianceSimulator`
        that requires the activity probability profiles.

        Activity profiles come as a dict key -> np.array
        *key: activity name
        *Array Shapes: DIM0: Time, DIM1:Active_Occupants

        Args:
            subgroup: requested subgroup activities profile

        Returns:
            activity_profiles_dict, A dictionary of daily activity
            profiles, where the key
            is the activity, and the profiles are arrays of shape
            DIM0:n_times, DIM1:active_occupancy.
        """
        subgroup_str = subgroup_string(subgroup)
        file_name = os.path.join(
            self.activity_type,
            'activity_profiles_' + subgroup_str)
        file_path = os.path.join(
            self.parsed_path_activity,
            'activity_profiles_' + subgroup_str)


        try:
            activity_profiles = dict(np.load(file_path + '.npz'))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            activity_profiles = self._parse_activity_profiles(
                subgroup
            )
            np.savez(file_path, **activity_profiles)

        return activity_profiles

    def _parse_activity_profiles(
                self, subgroup: Subgroup
            ) -> Dict[str, np.ndarray]:
        raise NotImplementedError()


