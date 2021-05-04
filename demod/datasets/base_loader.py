"""Module implementing base classes for loading datasets."""
from __future__ import annotations
from datetime import timedelta
import datetime
import inspect
from os.path import isfile
from ..utils.parse_helpers import lists_to_numpy_array, make_jsonable
from ..utils.sim_types import AppliancesDict, StateLabels, Subgroup, Subgroups, TPM, TPMs
from ..utils.appliances import get_ownership_from_dict
from ..utils.subgroup_handling import subgroup_string

import os
import json
import shutil
from typing import Any, Dict, List, Tuple, Union
import warnings

import numpy as np
from numpy.lib.npyio import NpzFile
import pandas as pd


DATASET_PATH = os.path.dirname(__file__)


class DatasetLoader:
    """Base class for loading Datasets.

    Can be used for helping parsing, saving and loading files.
    It helps retrieving access path to datasets.
    It also provides methods for warning and errors.

    Attributes:
        DATASET_NAME: The name of the dataset folder, should always be
            specified.
        raw_path: The path of the raw data folder, can be used to help
            accessing raw files
        parsed_path: The path of the parsed data folder,
            can be used to help accessing parsed files.
        version: The version name of the data.

    """

    DATASET_NAME: str
    DATASET_PATH: str
    raw_path: str
    parsed_path: str
    version: str

    def __init__(
        self, /, allow_pickle: bool = False, version: str = None,
        clear_parsed_data: bool = False
    ) -> Any:
        """Initailize a Dataset Loader.

        Children of this class should have their attribute
        DATASET_NAME defined as it will be used for the handling
        of data files.

        .. warning::    (from Numpy) Loading files that contain object
                        arrays uses the
                        ``pickle`` module, which is not secure against
                        erroneous or maliciously constructed data.
                        Consider passing ``allow_pickle=False`` to load
                        data that is known not to contain object arrays
                        for the safer handling of untrusted sources.

        Args:
            allow_pickle : Whether to allow pickle. (see warning).
                default: False.
            version: Optional version of the dataset.
                parsed data will contain several version
                None if there is only a single version.
            clear_parsed_data: Whether to clear the parsed data.


        Returns:
            Any: [description]
        """
        if not hasattr(self, "DATASET_NAME"):
            raise ValueError(
                (
                    "You must set the attribute DATASET_NAME "
                    "to {} as it is mandatory for children of DatasetLoader. "
                    "DATASET_NAME should be the exact same name as you name "
                    "the dataset folder."
                ).format(inspect.getmodule(self))
            )
        self.DATASET_PATH = os.path.join(DATASET_PATH, self.DATASET_NAME)
        self.raw_path = os.path.join(self.DATASET_PATH, "raw_data")
        self.parsed_path = os.path.join(self.DATASET_PATH, "parsed_data")

        if version is not None:
            self.parsed_path = os.path.join(self.parsed_path, version)
        self.version = version

        if clear_parsed_data:
            self._clear_parsed_data()

        self.allow_pickle = allow_pickle

    def _clear_parsed_data(self):
        user_input = None

        # Warns about the destruction of all the versions
        if self.version is None:
            while user_input not in ['yes', 'no']:
                user_input = input((
                    'You are about to erase the parsed data for '
                    'all the versions of {ds} dataset. \n'
                    'You can specifiy a version name using '
                    "version = 'v_name'. \n"
                    "Do you want to procced with erasing all version of"
                    " {ds} ? Enter yes or no: ").format(ds=self.DATASET_NAME)
                )

        if user_input == 'no':
            print('No data was erased.')
            return

        # clear the parsed directory
        shutil.rmtree(self.parsed_path)
        os.mkdir(self.parsed_path)

    def _raise_missing_raw(
        self,
        file_name: str,
        optional_download_website: str = None,
        full_path=False,
    ):
        """Raise custom error for missing raw data files.

        Args:
            file_name: The file that is missing
            optional_download_website: a website where the data could be
                 found
            full_path: whether the :py:attr:`file_name` given is acutally
                the full path to the file

        Raises:
            FileNotFoundError: The error specifying the missing file.
        """
        # Creates a custom errormessage.
        msg = "".join(
            (
                "Dataset '{}' has no data file '{}'. If you have it,",
                " you can place it in '{}' .",
            )
        ).format(
            self.DATASET_NAME,
            file_name,
            file_name
            if full_path
            else os.path.join(  # Full data path
                self.DATASET_PATH, "raw_data", file_name
            ),
        )
        if optional_download_website:
            msg = "".join(
                (
                    msg,
                    " You can download the data from : ",
                    optional_download_website,
                )
            )
        raise FileNotFoundError(msg)

    def _warn_could_not_load_parsed(
        self,
        exception_raised: Exception,
        data_name: str,
        warning_type: str = "warning",
    ):
        """Send a could not load parsed warning message.

        Contains the exception raised while
        loading the raw data.

        Args:
            exception_raised: The exception that was raised during the
                parsing of the data
            data_name: The name of the data that was loaded
            warning_type: The type of warning to raise.
                Curently implemented:
                    - 'warning' : using the warning module
                    - 'print' : standart print()
                    - 'all', : all the above
                If warning_type is not registerd, will use 'print'.
                Defaults to 'warning'.
        """
        warn_msg = "".join(
            (
                "Could not load parsed data for '{}', due to: \n '{}'",
                "with message: '{}'.\nGenerating now from raw_data.",
            )
        )
        msg = warn_msg.format(
            data_name, type(exception_raised).__name__, exception_raised
        )

        if (warning_type == "warning") or (warning_type == "all"):
            warnings.warn(msg)
        else:
            print(msg)

    def _load_npz(self, file_name: str):
        np_file_pathname = os.path.join(self.parsed_path, file_name + ".npz")
        npz_file = np.load(np_file_pathname, allow_pickle=self.allow_pickle)
        return [value for value in npz_file.values()]

    def _load_npy(self, file_name: str):
        np_file_pathname = os.path.join(self.parsed_path, file_name + ".npy")
        out = np.load(np_file_pathname, allow_pickle=self.allow_pickle)
        return out

    def _save_step_size(self, file_name: str, step_size: timedelta) -> None:
        file_path = os.path.join(self.parsed_path, file_name + ".json")
        # Only days, seconds and microseconds are stored internally,
        # in timedelta objects
        with open(file_path, "w+") as f:
            json.dump(
                {
                    "days": step_size.days,
                    "seconds": step_size.seconds,
                    "microseconds": step_size.microseconds,
                },
                f,
            )

    def _load_step_size(self, file_name: str) -> timedelta:
        file_path = os.path.join(self.parsed_path, file_name + ".json")
        with open(file_path, "r") as f:
            dic = json.load(f)
        return timedelta(**dic)

    def _load_parsed_data(
        self,
        file_name: str,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Load the file name from parsed data.

        Args:
            file_name : The name you want the file to have

        Raises:
            FileNotFoundError: if the desired file was not found

        Returns:
            The array or arrayfile to be loaded.
        """
        try:
            out = self._load_npy(file_name)
        except FileNotFoundError as e_1:
            # if npy does not work, try npz
            try:
                out = self._load_npz(file_name)
            except FileNotFoundError as e_2:
                msg = "Not found npy or npz file for: {}*".format(
                    e_1.filename[:-1]
                )  # replace z or y by *
                raise FileNotFoundError(msg) from e_2
        except ValueError as v_e:
            pkl_msg = "Object arrays cannot be loaded when allow_pickle=False"
            # Raise special message for allow_pickle error
            if (str(v_e) == pkl_msg) and (not self.allow_pickle):
                err_msg = "".join(
                    (
                        pkl_msg,
                        ". You can allow pickle using: '",
                        type(self).__name__,
                        "(allow_pickle=True)'",
                    )
                )
                raise ValueError(err_msg)
            else:
                raise v_e

        return out

    def _save_parsed_data(
        self,
        file_name: str,
        array: Union[np.ndarray, List[np.ndarray]],
        npz: bool = False,
        compress: bool = False,
    ):
        """Save an array or a list of array to the parsed_data folder.

        If the array given is a List, it will be saved as a single npy
        array if (npz or compress) is not True.
        If the array given is a ndarray, it will be saved to .npy or to
        .npz if npz is specified.

        Args:
            file_name: The name of the .npy-z file
            array: Array of list of arrays to be saved.
            npz: Whether to save in npz file. Defaults to False.
            compress: Whether to use compression while saving. (only
                available for npz format.)

        Note:
            Compression can only use npz format, so that when compress
            is True, kwarg 'npz' is ignored.
        """
        file_path = os.path.join(self.parsed_path, file_name)

        if isinstance(array, np.ndarray):
            if compress or npz:
                array_list = [array]  # Packs for unpacking later
        elif isinstance(array, list) or isinstance(array, tuple):
            array_list = array
            if not (compress or npz):
                array = np.asarray(array)
        else:
            raise TypeError("Arg 'array' must be ndarray or list or tuple.")

        if compress:
            np.savez_compressed(file_path, *array_list)
        elif npz:
            np.savez(file_path, *array_list)
        else:
            np.save(file_path, array)

    def _check_make_parsed_dir(self):
        """Create the parsed path dir if it does not exists."""
        if not os.path.isdir(self.parsed_path):
            os.mkdir(self.parsed_path)


class PopulationLoader(DatasetLoader):
    """Loader for population data.
    """
    def __init__(self, /, **kwargs) -> Any:
        super().__init__(**kwargs)
        self.parsed_population_folder = os.path.join(
            self.parsed_path, 'population'
        )
        if not os.path.isdir(self.parsed_population_folder):
            os.mkdir(self.parsed_population_folder)

    def load_population_subgroups(
        self, population_type: str,
    ) -> Tuple[Subgroups, List[float], int]:
        """Loads the subgroups and their numbers of a population.

        The population refers to the households population.
        Returns the list of subgroups, the proportion of each
        subgroups in the population and the total number of
        households for this population.
        Different splitting can be specified using the the
        :py:attr:`population_type` argument.

        Returns:
            subgroups_list, subgroup_prob, total_population
        """
        app_file = (
            self.parsed_population_folder + os.sep + population_type + ".json"
        )
        try:
            with open(app_file, "r") as f:
                population_dict = dict(json.load(f))

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, app_file)

            population_dict = {}
            (
                population_dict['subgroups_list'],
                population_dict['subgroups_probs'],
                population_dict['total_number']
             ) = self._parse_population_subgroups(population_type)

            with open(app_file, "w+") as f:
                json.dump(make_jsonable(population_dict), f, indent=2)

        return (
            population_dict['subgroups_list'],
            population_dict['subgroups_probs'],
            population_dict['total_number']
        )

    def _parse_population_subgroups(
        self, population_type: str,
    ) -> Tuple[Subgroups, List[float], int]:
        raise NotImplementedError(
            "'_parse_population_subgroups' requires overriding in "
            "{}.".format(type(self).__name__)
        )

class ApplianceLoader(DatasetLoader):
    """Loader that provide methods for loading appliances data.

    Children of this class need to implement the following methods:
    * :py:meth:`_parse_appliance_dict`
    """

    def load_appliance_dict(self) -> AppliancesDict:
        """Load the appliance dictionary.

        Try to call self. :py:meth:`_parse_appliance_dict` if the
        parsed data is not available.

        Returns:
            The appliance dictionary.
        """
        app_file = self.parsed_path + os.sep + "appliance_dict.json"
        try:
            with open(app_file, "r") as f:
                appliance_dict = dict(json.load(f))

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, app_file)

            appliance_dict = self._parse_appliance_dict()

            if "type" not in appliance_dict:
                raise ValueError(
                    (
                        "'type' is not in the appliance_dict you have just"
                        "parsed. Check '{}._parse_appliance_dict' and make "
                        "sure you add 'type' key to the appliance_dict."
                    ).format(type(self).__name__)
                )

            with open(app_file, "w+") as f:
                json.dump(make_jsonable(appliance_dict), f, indent=2)

        return lists_to_numpy_array(appliance_dict)

    def _parse_appliance_dict(self):
        raise NotImplementedError(
            "'_parse_appliance_dict' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def _parse_appliance_ownership_dict(
        self, subgroup: Subgroup
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "'_parse_appliance_ownership_dict' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_appliance_ownership_dict(
        self,
        subgroup: Subgroup = {},
    ) -> Dict[str, float]:
        """Return the dictionary with probability of owning appliances.

        A subgroup can be specifies for datasets that differentiate
        different subgroups.

        :py:func:`~demod.utils.appliances.get_ownership_from_dict`
        can then be used to sample the ownership
        using an appliance dictionary.

        Args:
            subgroup: The subgroup of the desired ownership
                probabilities.

        Return:
            pdf of ownership for each appliance
        """
        subgroup_str = subgroup_string(subgroup)

        # put in a dedicated folder
        folder_name = "appliance_ownership"
        parsed_path_ownership = self.parsed_path + os.sep + folder_name
        if not os.path.exists(parsed_path_ownership):
            os.mkdir(parsed_path_ownership)

        file_name = folder_name + os.sep + "ownership__" + subgroup_str
        file_path = os.path.join(self.parsed_path, file_name)

        try:
            ownership_dict = dict(np.load(file_path + ".npz"))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_name)
            ownership_dict = self._parse_appliance_ownership_dict(subgroup)
            np.savez(file_path, **ownership_dict)

        return ownership_dict

    def _parse_appliance_ownership(
        self,
        appliance_dict: AppliancesDict,
        subgroup: Subgroup = None,
    ) -> Union[np.ndarray, Dict[str, float]]:

        raise NotImplementedError(
            "'_parse_appliance_ownership' requires overriding in "
            "{}.".format(type(self).__name__)
        )


class ClimateLoader(DatasetLoader):
    """Loader providing methods for loading climate data.

    Attributes:
        step_size: The time between two different data points from
            the historical data.
    """

    step_size: timedelta

    def __init__(self, /, **kwargs) -> Any:
        """Create a climate loader.

        Args:
            version: The version of the dataset used.
            allow_pickle: Wheter to allow pickle. Keep it to false unless
                you know what you are doing.
        """
        super().__init__(**kwargs)
        self.parsed_path_climate = os.path.join(self.parsed_path, "climate")
        if not os.path.isdir(self.parsed_path_climate):
            os.mkdir(self.parsed_path_climate)

    def load_clearness_tpms(self) -> Tuple[TPM, StateLabels, timedelta]:
        """Return TPM for the clearness of the sky, with the labels.

        The tpm containains the probability that the sky clearness
        changes at each step.

        Returns:
            1. The TPM of clearness
            2. Labels containing the clearness value of each TPM states
            3. The step size of the tpm, resolution of the transitions.
        """
        file_path = os.path.join(self.parsed_path_climate, "clearness_tpms")
        labels_path = os.path.join(
            self.parsed_path_climate, "clearness_labels"
        )
        step_size_file_name = "climate" + os.sep + "clearness_step_size"

        try:
            clearness_tpms = np.load(file_path + ".npy")
            labels = np.load(labels_path + ".npy")
            step_size = self._load_step_size(step_size_file_name)

        except FileNotFoundError as f_err:
            self._warn_could_not_load_parsed(f_err, file_path)
            clearness_tpms, labels, step_size = self._parse_clearness_tpms()
            np.save(file_path, clearness_tpms)
            np.save(labels_path, labels)
            self._save_step_size(step_size_file_name, step_size)

        return clearness_tpms, labels, step_size

    def _parse_clearness_tpms(self) -> Tuple[TPMs, StateLabels]:
        raise NotImplementedError(
            "'_parse_clearness_tpms' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_temperatures_arma(self) -> Dict[str, float]:
        """Load the parameters for a temperature arma model.

        Returns:
            arma_dict: contains the parameters of the arma model.
        """
        file_path = os.path.join(self.parsed_path, "temperatures_arma.json")
        try:
            with open(file_path, "r") as f:
                geo_dic = json.load(f)
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "temperatures_arma.json")
            geo_dic = self._parse_temperatures_arma()
            with open(file_path, "w+") as f:
                json.dump(geo_dic, f)

        return geo_dic

    def _parse_temperatures_arma(self) -> Dict[str, float]:
        raise NotImplementedError(
            "'_parse_temperatures_arma' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_geographic_data(self) -> Dict[str, Union[str, float]]:
        """Return a dictionary with geographic information on the dataset.

        Returns:
            geo_dic, The geographic data dictionary with available keys

                * 'country': the country where the data is collected
                * 'latitude': in degree
                * 'longitude': in degree
                * 'meridian': in degree
                * 'use_daylight_saving_time': whether to use the time shift

        """
        file_path = os.path.join(self.parsed_path, "geographic_data.json")
        try:
            with open(file_path, "r") as f:
                geo_dic = json.load(f)
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "geographic_data.json")
            geo_dic = self._parse_geographic_data()
            with open(file_path, "w+") as f:
                json.dump(geo_dic, f)

        return geo_dic

    def _parse_geographic_data(self) -> Dict[str, Union[str, float]]:
        raise NotImplementedError(
            "'_parse_geographic_data' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_historical_climate_data(
        self, start_datetime: datetime.datetime
    ) -> Dict[str, np.ndarray]:
        """Load historical data starting from the requested time.

        Provides the data point at the start of the simulation
        using :py:attr:`start_datetime`
        , or the closest one before the start.

        Args:
            start_datetime: a datetime object, specifying the start of
                the required data.

        Returns:
            climate_dict, a dictionary with the following possible keys

            * 'datetime': the time stored as numpy 'datetime64', only mandatory key. the datetime array should be in utc format time.
            * 'temperature': the temperature of the air [C]
            * 'radiation_diffuse': diffuse radiation at surface [W/m^2]
            * 'radiation_direct': direct radiation at surface [W/m^2]
            * 'radiation_global': global radiation at surface [W/m^2]
            * 'radiation': same as radiation_global [W/m^2]
        """
        # put in a dedicated folder
        file_path = self.parsed_path_climate + os.sep + "historical_climate"

        try:
            climate_dict = dict(np.load(file_path + ".npz"))
        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, file_path)
            climate_dict = self._parse_historical_climate_data()
            np.savez_compressed(file_path, **climate_dict)

        # Check for aware datetime
        if start_datetime.tzinfo is not None:
            # Convert the datetime to a naive utc datetime
            start_datetime_utc = datetime.datetime.utcfromtimestamp(
                start_datetime.timestamp()
            )
        else:
            # Assume the input is already utc
            start_datetime_utc = start_datetime

        # Checks that the requested start is not too late.
        if start_datetime_utc > climate_dict["datetime"][-1]:
            raise ValueError(
                "Requested start_datetime is : {}, but dataset {} "
                "ends at {}".format(
                    start_datetime_utc,
                    self,
                    climate_dict["datetime"][-1],
                )
            )

        mask = np.where(
            climate_dict["datetime"] > start_datetime_utc - self.step_size
        )[0]

        # Return only the values starting from datetime
        return {key: value[mask] for key, value in climate_dict.items()}

    def _parse_historical_climate_data(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError(
            "'_parse_historical_climate_data' requires overriding in "
            "{}.".format(type(self).__name__)
        )

class LightingLoader(DatasetLoader):
    """Loader for lighting simulators components.
    """
    def __init__(self, /, **kwargs) -> Any:
        super().__init__(**kwargs)
        self.parsed_bulbs_folder = os.path.join(
            self.parsed_path, 'bulbs'
        )
        if not os.path.isdir(self.parsed_bulbs_folder):
            os.mkdir(self.parsed_bulbs_folder)

    def load_fisher_lighting(self) -> Dict[str, Any]:
        """Load data for
        :py:class:`~demod.simulators.lighting_simulators.FisherLighitingSimulator`

        Returns:
            Fisher lighting sim parameters dict.
        """
        file_path = os.path.join(self.parsed_path, "fisher_lighting.json")
        try:
            with open(file_path, "r") as f:
                fisher_dic = json.load(f)
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "fisher_lighting.json")
            fisher_dic = self._parse_fisher_lighting()
            with open(file_path, "w+") as f:
                json.dump(fisher_dic, f)

        return fisher_dic

    def _parse_fisher_lighting(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "'_parse_fisher_lighting' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_crest_lighting(self) -> Dict[str, Any]:
        """Load data for
        :py:class:`~demod.simulators.lighting_simulators.CrestLightingSimulator`

        Returns:
            crest lighting sim parameters dict.
        """
        file_path = os.path.join(self.parsed_path, "crest_lighting.json")
        try:
            with open(file_path, "r") as f:
                crest_dic = json.load(f)
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "crest_lighting.json")
            crest_dic = self._parse_crest_lighting()
            with open(file_path, "w+") as f:
                json.dump(make_jsonable(crest_dic), f)

        return lists_to_numpy_array(crest_dic)

    def _parse_crest_lighting(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "'_parse_crest_lighting' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_bulbs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data for each light bulb, consumption and penetration.

        Returns are arrays where each bulb type is an element of the array.
        Consumption is in Watts.
        Penetration is in probability.

        Returns:
            consumption, penetration
        """
        file_path = os.path.join(self.parsed_path, "bulbs.json")
        try:
            with open(file_path, "r") as f:
                bulbs_dic = json.load(f)
                bulbs_consumption, bulbs_penetration = (
                    bulbs_dic['bulbs_consumption'],
                    bulbs_dic['bulbs_penetration'],
                )
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "bulbs.json")
            bulbs_consumption, bulbs_penetration = self._parse_bulbs()
            bulbs_dic = {
                'bulbs_consumption': bulbs_consumption,
                'bulbs_penetration': bulbs_penetration,
            }
            with open(file_path, "w+") as f:
                json.dump(make_jsonable(bulbs_dic), f)

        return (
            np.array(bulbs_consumption),
            np.array(bulbs_penetration)
        )

    def _parse_bulbs(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "'_parse_bulbs' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_bulbs_config(self, subgroup: Subgroup = {}) -> np.ndarray:
        """Return the light config of some houses.

        The light config is A 2-D array,
        where Dim0 is the different houses and
        Dim1 the different bulbs of each house. The values
        correspond to the bulb consumption in watts.

        Args:
            subgroup: The subgroup corresponding to the config. Defaults to {}.

        Returns:
            config: The light bulbs config.
        """
        file_path = os.path.join(
            self.parsed_bulbs_folder,
            'bulbs_config_' + subgroup_string(subgroup)
        )
        try:
            bulbs = np.load(file_path + '.npy')
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, file_path)
            bulbs = self._parse_bulbs_config(subgroup)
            np.save(file_path, bulbs)

        return bulbs

    def _parse_bulbs_config(self, subgroup) -> np.ndarray:
        raise NotImplementedError(
            "'_parse_bulbs_config' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_installed_bulbs_stats(
        self, subgroup: Subgroup = {}
    ) -> Tuple[float, float]:
        """Load the mean and std of the number of bulbs installed.

        Args:
            subgroup: The requested subgroup.

        Returns:
            mean, std
        """
        file_path = os.path.join(
            self.parsed_bulbs_folder,
            'bulbs_' + subgroup_string(subgroup) + '.json'
        )

        try:
            with open(file_path, "r") as f:
                dic = json.load(f)
        except FileNotFoundError as fnf_err:
            self._warn_could_not_load_parsed(fnf_err, "bulbs")
            mean, std  = self._parse_installed_bulbs_stats(subgroup)
            dic = {
                    'mean': mean,
                    'std': std,
                }
            with open(file_path, "w+") as f:
                json.dump(dic, f)

        return dic['mean'], dic['std']

    def _parse_installed_bulbs_stats(self, subgroup: Subgroup):
        raise NotImplementedError(
            "'_parse_installed_bulbs_stats' requires overriding in "
            "{}.".format(type(self).__name__)
        )

class HeatingLoader(DatasetLoader):
    """Loader for heating simulators components.
    """
    def __init__(self, /, **kwargs) -> Any:
        super().__init__(**kwargs)
        self.parsed_heating_folder = os.path.join(
            self.parsed_path, 'heating'
        )
        if not os.path.isdir(self.parsed_heating_folder):
            os.mkdir(self.parsed_heating_folder)
    def load_buildings_dict(
        self, subgroup: Subgroup = {}
    ) -> Dict[str, np.ndarray]:
        """Load the buildings dictionary.

        Try to call self. :py:meth:`_parse_buildings_dict` if the
        parsed data is not available.

        Returns:
            The buildings dictionary.
        """
        app_file = self.parsed_heating_folder + os.sep + subgroup_string(subgroup) + "_buildings_dict.json"
        try:
            with open(app_file, "r") as f:
                buildings_dict = dict(json.load(f))

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, app_file)

            buildings_dict = self._parse_buildings_dict(subgroup)

            with open(app_file, "w+") as f:
                json.dump(make_jsonable(buildings_dict), f, indent=2)

        return lists_to_numpy_array(buildings_dict)

    def _parse_buildings_dict(self, subgroup: Subgroup):
        raise NotImplementedError(
            "'_parse_buildings_dict' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_heating_system_dict(
        self, subgroup: Subgroup = {}
    ) -> Dict[str, np.ndarray]:
        """Load the heating_system dictionary.

        Try to call self. :py:meth:`_parse_heating_system_dict` if the
        parsed data is not available.

        Returns:
            The heating_system dictionary.
        """
        app_file = self.parsed_heating_folder + os.sep + subgroup_string(subgroup) + "_heating_system_dict.json"
        try:
            with open(app_file, "r") as f:
                heating_system_dict = dict(json.load(f))

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, app_file)

            heating_system_dict = self._parse_heating_system_dict(subgroup)

            with open(app_file, "w+") as f:
                json.dump(make_jsonable(heating_system_dict), f, indent=2)

        return lists_to_numpy_array(heating_system_dict)

    def _parse_heating_system_dict(self, subgroup: Subgroup):
        raise NotImplementedError(
            "'_parse_heating_system_dict' requires overriding in "
            "{}.".format(type(self).__name__)
        )

    def load_thermostat_dict(
        self, subgroup: Subgroup = {}
    ) -> Dict[str, np.ndarray]:
        """Load the thermostat dictionary.

        Try to call self. :py:meth:`_parse_thermostat_dict` if the
        parsed data is not available.

        Returns:
            The thermostat dictionary.
        """
        app_file = self.parsed_heating_folder + os.sep + subgroup_string(subgroup) + "_thermostat_dict.json"
        try:
            with open(app_file, "r") as f:
                thermostat_dict = dict(json.load(f))

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, app_file)

            thermostat_dict = self._parse_thermostat_dict(subgroup)

            with open(app_file, "w+") as f:
                json.dump(make_jsonable(thermostat_dict), f, indent=2)

        return lists_to_numpy_array(thermostat_dict)

    def _parse_thermostat_dict(self, subgroup: Subgroup):
        raise NotImplementedError(
            "'_parse_thermostat_dict' requires overriding in "
            "{}.".format(type(self).__name__)
        )