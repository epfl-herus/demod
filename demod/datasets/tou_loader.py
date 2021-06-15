from datetime import datetime, time
from demod.utils.monte_carlo import PDFs
from demod.utils.parse_helpers import make_jsonable
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

    def load_tpm(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, StateLabels, PDF]:
        """Loads a transition probability matrix for the requested
        subgroup.

        Args:
            subgroup: The desired subgroup

        Returns:
            the transition probability matrix, the labels and the initial pdf
        """
        subgroup_str = subgroup_string(subgroup)

        file_path = os.path.join(self.parsed_path_activity, subgroup_str)
        file_name = os.path.join(self.activity_type, subgroup_str)

        try:
            tpm_file = dict(np.load(file_path + '.npz'))
            tpm, labels, initial_pdf = (
                tpm_file['tpm'],
                tpm_file['labels'],
                tpm_file['initial_pdf']
            )

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            tpm, labels, initial_pdf, legend = self._parse_tpm(
                subgroup
            )
            tpm_dict = {}

            tpm_dict['tpm'] = tpm
            tpm_dict['labels'] = labels
            tpm_dict['initial_pdf'] = initial_pdf

            np.savez(file_path, **tpm_dict)

            # Saves a legend of the files created
            legend_path = (
                self.parsed_path_activity + os.sep
                + subgroup_str + '_dict_legend.json'
            )
            with open(legend_path, 'w') as fp:
                json.dump(make_jsonable(legend), fp, indent=4)

        return tpm, labels, initial_pdf

    def _parse_tpm(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, StateLabels, PDF, dict]:

        raise NotImplementedError()

    def load_tpm_with_duration(
        self, subgroup: Subgroup,
    ) -> Tuple[TPMs, np.ndarray, StateLabels, PDF, PDFs]:
        """Load tpms and durations for the requested subgroup.

        Args:
            subgroup: The desired subgroup

        Returns:
            the transition probability matrix, the durations pdfs,
            the labels, the initial state pdf and the intial durations pdf
        """
        subgroup_str = subgroup_string(subgroup) + '_with_duration'

        file_path = os.path.join(self.parsed_path_activity, subgroup_str)
        file_name = os.path.join(self.activity_type, subgroup_str)

        try:
            tpm_file = dict(np.load(file_path + '.npz'))
            (
                tpm, duration_pdfs,
                labels, initial_pdf, initial_duration_pdfs
            ) = (
                tpm_file['tpm'],
                tpm_file['duration_pdfs'],
                tpm_file['labels'],
                tpm_file['initial_pdf'],
                tpm_file['initial_duration_pdfs']
            )

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            (
                tpm, duration_pdfs, labels,
                initial_pdf, initial_duration_pdfs, legend
            ) = self._parse_tpm_with_duration(
                subgroup
            )
            tpm_dict = {}
            (
                tpm_dict['tpm'],
                tpm_dict['duration_pdfs'],
                tpm_dict['labels'],
                tpm_dict['initial_pdf'],
                tpm_dict['initial_duration_pdfs']
            ) = (
                tpm, duration_pdfs,
                labels, initial_pdf, initial_duration_pdfs
            )

            np.savez(file_path, **tpm_dict)

            # Saves a legend of the files created
            legend_path = (
                self.parsed_path_activity + os.sep
                + subgroup_str + '_dict_legend.json'
            )
            with open(legend_path, 'w') as fp:
                json.dump(make_jsonable(legend), fp, indent=4)

        return tpm, duration_pdfs, labels, initial_pdf, initial_duration_pdfs


    def _parse_tpm_with_duration(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, np.ndarray, StateLabels, PDF, PDFs, dict]:
        """Parse tpms and durations for the requested subgroup.

        Args:
            subgroup: The desired subgroup

        Returns:
            the transition probability matrix,
            the durations pdfs that depend only on new state,
            the durations pdfs that depend also on the previous state,
            the labels, the initial state pdf, the intial durations pdf,
            and a dictionary with metadata of the parsing
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
                json.dump(make_jsonable(legend), fp, indent=4)

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

        The probability profiles are based on how many active occupants are
        in the house. If you want only the probability that the activity
        is occuring, use :py:meth:`.load_activity_probabilities`

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

    def _check_make_parsed_act_stats(self):
        self.parsed_path_activity_stats = os.path.join(
            self.parsed_path_activity,
            'ActivityStats'
        )
        if not os.path.isdir(self.parsed_path_activity_stats):
            os.mkdir(self.parsed_path_activity_stats)

    def load_daily_activity_starts(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        """Return the probability of performing an activity in a day.

        Can vary depending on the subgroup.

        A dictionaries containing pdfs of how many times
        an activity is performed during a day.
        The i-eth element is the probability that activity
        is performed i times in a day.

        Arrays can be of variable length.

        ex:
        {
            'activity1': [0.3, 0.2, 0.5, 0.0],
            'activity2': [0.3, 0.2, 0.0, 0.3, 0.2],
            ...
        }
        """
        # Handles the storage of the data
        self._check_make_parsed_act_stats()
        subgroup_str = subgroup_string(subgroup)
        file_name = os.path.join(
            self.parsed_path_activity_stats,
            'act_occurence_' + subgroup_str)
        file_path = os.path.join(
            self.parsed_path_activity_stats,
            'act_occurence_' + subgroup_str)


        try:
            act_occurences = dict(np.load(file_path + '.npz'))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            act_occurences = self._parse_daily_activity_starts(
                subgroup
            )
            np.savez(file_path, **act_occurences)

        return act_occurences

    def _parse_daily_activity_starts(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def load_activity_duration(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        """Return the probability of activity duration.

        Can vary depending on the subgroup.

        A dictionaries containing pdfs of how long the activity last.
        The i-eth element means the duration is i*step_size.

        Element 0 means the the duration is smaller than step_size.
        ex:
        {
            'activity1': [0.0, 0.3, 0.2, 0.5, 0.0],
            'activity2': [0.0, 0.3, 0.2, 0.0, 0.3, 0.2],
            ...
        }
        """
        # Handles the storage of the data
        self._check_make_parsed_act_stats()
        subgroup_str = subgroup_string(subgroup)
        file_name = os.path.join(
            self.parsed_path_activity_stats,
            'act_duration_' + subgroup_str)
        file_path = os.path.join(
            self.parsed_path_activity_stats,
            'act_duration_' + subgroup_str)


        try:
            act_durations = dict(np.load(file_path + '.npz'))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            act_durations = self._parse_activity_duration(
                subgroup
            )
            np.savez(file_path, **act_durations)

        return act_durations

    def _parse_activity_duration(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def load_activity_probabilities(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        """Return the probability that the activity is performed.

        Proabilites are given at each step, during the day.
        At each step the probability means the probability of doing
        that activity instead of another.
        """
        self._check_make_parsed_act_stats()
        subgroup_str = subgroup_string(subgroup)
        file_name = os.path.join(
            self.parsed_path_activity_stats,
            'act_probs_' + subgroup_str
        )
        file_path = os.path.join(
            self.parsed_path_activity_stats,
            'act_probs_' + subgroup_str
        )

        try:
            act_proabs = dict(np.load(file_path + '.npz'))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            act_proabs = self._parse_activity_probabilities(
                subgroup
            )
            np.savez(file_path, **act_proabs)
        return act_proabs

    def _parse_activity_probabilities(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
