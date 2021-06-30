"""Loader for CREST data.

Loads data from the crest spreadsheet.
"""

from datetime import time, timedelta
from sys import version
import shutil
import warnings
import os
from urllib import request
from typing import Any, List, Tuple, Dict, Union

import pandas as pd
import numpy as np

from ..base_loader import (
    ApplianceLoader,
    ClimateLoader,
    DatasetLoader,
    HeatingLoader,
    LightingLoader,
    PopulationLoader,
)
from ...utils.sim_types import (
    AppliancesDict,
    StateLabels,
    Subgroup,
    Subgroups,
    TPMs,
)
from ...utils.subgroup_handling import (
    check_weekend_day_format,
    is_weekday,
    is_weekend,
    subgroup_string,
)
from ...utils.distribution_functions import rescale_pdf
from ...utils.parse_helpers import remove_spaces, translate_1d, bulbs_stats_from_config
from ..tou_loader import LoaderTOU
from .utils import (
    crest_act_to_demod,
    crest_appname_to_demod_type,
)
from ...utils.error_messages import (
    DATASET_CANNOT_DISTINGUISH_ON_SUBGROUPS,
    NOT_IMPLEMENTED_IN_DATASET_FOR_VERSION,
    UNKOWN_POPULATION_TYPE,
)


DOWNLOAD_URL_BY_VERSION = {
    '2.2': 'https://repository.lboro.ac.uk/ndownloader/files/4969096',
    '2.3.3': 'https://repository.lboro.ac.uk/ndownloader/files/22693271',
}

class Crest(
    ApplianceLoader, LoaderTOU, ClimateLoader, LightingLoader,
    PopulationLoader, HeatingLoader
):
    """Crest Data.

    CREST model was developped by Loughborough University, Leicestershire, UK

    The CREST demand model can be accessed
    `here
    <https://repository.lboro.ac.uk/articles/dataset/CREST_Demand_Model_v2_0/2001129>`_

    Args:
        version: The version of CREST to use. Defaults to '2.2'.
    """

    DATASET_NAME = "CREST"
    refresh_time = time(0, 0, 0)
    # Step size of the load simulation
    step_size = timedelta(minutes=1)

    def __init__(self, version: str = "2.2", /, **kwargs) -> Any:
        """Create a data loader for CREST.

        Args:
            version: [description]. Defaults to '2.2'.
            allow_pickle: [description]. Defaults to False.

        Returns:
            Any: [description]
        """
        version_name = "v_" + version
        self.version = version
        super().__init__(version=version_name, **kwargs)

        raw_file_name = "CREST_Demand_Model_v" + version + ".xlsm"
        self.raw_file_path = os.path.join(self.raw_path, raw_file_name)


        # Downloads the raw file if it does not exist
        if not os.path.isfile(self.raw_file_path):
            if version not in DOWNLOAD_URL_BY_VERSION:
                raise ValueError('Unkonw Version {} of CREST.'.format(version))
            # Reads the url and  download the
            print('Downloading CREST Model to : {}'.format(self.raw_file_path))
            with request.urlopen(DOWNLOAD_URL_BY_VERSION[version]) as response:
                with open(self.raw_file_path, 'wb') as f:
                    shutil.copyfileobj(response, f)

    def load_population_subgroups(
        self, population_type: str = "crest",
    ) -> Tuple[Subgroups, List[float], int]:
        # Defaults the population type
        return super().load_population_subgroups(population_type)

    def _parse_population_subgroups(
        self, population_type: str
    ) -> Tuple[Subgroups, List[float], int]:
        """Parse the subgroups.

        Read crest data to get the distribution of the population
        in dwellings depending on the number of residents in them.
        """
        if population_type == "crest" or population_type == "resident_number":
            try:
                df = pd.read_excel(
                    self.raw_file_path,
                    "ActivityStats",
                    header=11,
                    nrows=5,
                    usecols=[1, 2],
                    engine="openpyxl",
                )
            except FileNotFoundError:
                self._raise_missing_raw()

            if self.version == "v_2.3.3":
                pdf = df[(
                    "Probability of dwelling having this number of residents,"
                    " in the UK"
                )].to_numpy()
            elif self.version == "v_2.2":
                pdf = df[
                    ("Probability of dwelling having this number of residents")
                ].to_numpy()
            else:
                raise ValueError(
                    NOT_IMPLEMENTED_IN_DATASET_FOR_VERSION.format(
                        not_implemented="_parse_population_subgroups",
                        dataset=self,
                        version=self.version,
                    )
                )
            n_of_residents = df["Number of residents"].to_numpy()

            subgroups = [{"n_residents": n} for n in n_of_residents]

        else:
            raise ValueError(
                UNKOWN_POPULATION_TYPE.format(population_type, self)
            )

        return subgroups, pdf, 25e6  # estimate

    def _raise_missing_raw(self, *args, **kwargs):
        return super()._raise_missing_raw(
            self.raw_file_path,
            full_path=True,
            optional_download_website="https://repository.lboro.ac.uk/articles"
            "/dataset/CREST_Demand_Model_v2_0/2001129",
        )

    def _check_crest_subgroup(self, subgroup):
        """Ensure the subgroup is valid for CREST."""
        if ("n_residents" not in subgroup) or ("weekday" not in subgroup):
            raise ValueError(
                "'CREST' data requires two subgroup keys: "
                " 'weekday' and 'n_residents'."
            )
        elif subgroup["weekday"] != [1, 2, 3, 4, 5] and subgroup[
            "weekday"
        ] != [6, 7]:
            raise ValueError(
                "'CREST' data only differentiate weekdays and weekends."
                " 'subgroup['weekday']' must therefore be "
                "'[1, 2, 3, 4, 5]' or '[6, 7]' not"
                " {}".format(subgroup["weekday"])
            )
        elif subgroup["n_residents"] not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "'CREST' data only accepts 1,2,3,4,5,6 for "
                "'subgroup['n_residents']'"
            )
        return

    def _subgroup_to_daytype(self, subgroup):
        """Return 'd' or 'e' depending on the daytype from subgroup."""
        if subgroup["weekday"] == [1, 2, 3, 4, 5]:
            day_type = "d"
        elif subgroup["weekday"] == [6, 7]:
            day_type = "e"
        else:
            raise ValueError("unkonw day type {}".format(subgroup["weekday"]))

        return day_type

    def load_tpm(self, subgroup: Subgroup):
        """Load transition probability matrices for CREST subgroups.

        Args:
            subgroup: A subgroup containing 'n_residents' and 'weekday'.
        """
        self._check_crest_subgroup(subgroup)

        tpm_path = self.parsed_path + os.sep + "4_States"
        if not os.path.exists(tpm_path):
            os.mkdir(tpm_path)
        tpm_file_name = "".join(
            ("4_States", os.sep, "tpm__", subgroup_string(subgroup))
        )
        labels_file_name = "".join(
            ("4_States", os.sep, "labels__", subgroup_string(subgroup))
        )
        start_pdf_file_name = "".join(
            ("4_States", os.sep, "start_pdf__", subgroup_string(subgroup))
        )

        try:
            tpm = self._load_parsed_data(tpm_file_name)
            labels = self._load_parsed_data(labels_file_name)
            start_pdf = self._load_parsed_data(start_pdf_file_name)

        except FileNotFoundError as err:
            self._warn_could_not_load_parsed(err, tpm_file_name)

            tpm, labels, start_pdf = self._parse_tpm(subgroup)

            self._save_parsed_data(tpm_file_name, tpm)
            self._save_parsed_data(labels_file_name, labels)
            self._save_parsed_data(start_pdf_file_name, start_pdf)

        return tpm, labels, start_pdf

    def _parse_tpm(self, subgroup: Subgroup):
        day_type = self._subgroup_to_daytype(subgroup)

        sheet_name = "tpm{}_w{}".format(int(subgroup["n_residents"]), day_type)
        n_states = int((subgroup["n_residents"] + 1) ** 2)

        try:
            df = pd.read_excel(
                self.raw_file_path, sheet_name, skiprows=9, engine="openpyxl"
            )
        except FileNotFoundError:
            self._raise_missing_raw()

        cols = df.columns[2: n_states + 2]
        tpms = df[cols].to_numpy().reshape(144, n_states, n_states)

        # Replaces the missing values by 1 where alls are 0 in the pdf
        # Note that this is only required to have a valid cdf
        # but those states should never be reached by any household,
        # if the data was correctly interpreted
        times, rows = np.where(tpms.sum(axis=2) == 0)
        # make the stay at the same states
        tpms[times, rows, rows] = 1.0

        # Finally rescale the pdf values that don't make 1 when summed.
        tpms = rescale_pdf(tpms)

        # now get the starting staes
        if day_type == "d":
            starting_states = pd.read_excel(
                self.raw_file_path,
                "Starting states",
                header=4,
                skiprows=[5],
                nrows=49,
                engine="openpyxl",
            )
        elif day_type == "e":
            starting_states = pd.read_excel(
                self.raw_file_path,
                "Starting states",
                header=58,
                skiprows=[59],
                nrows=49,
                engine="openpyxl",
            )

        mask = np.concatenate(
            [
                np.arange(subgroup["n_residents"] + 1) + 7 * n
                for n in range(subgroup["n_residents"] + 1)
            ]
        )
        start_pdf = np.array(starting_states[subgroup["n_residents"]])[mask]

        labels = np.array(
            starting_states["Number of residents in the house"], dtype=int
        )[mask]

        return tpms, labels, start_pdf

    def _parse_appliance_dict(self) -> AppliancesDict:
        if self.version == "v_2.2":

            df = pd.read_excel(
                self.raw_file_path,
                "AppliancesAndWaterFixtures",
                header=0,
                skiprows=[0, 1, 2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44],
                nrows=41,
                engine="openpyxl",
            )

        else:
            raise NotImplementedError(
                "'_parse_appliance_dict' is not implemented for CREST "
                "version {}.".format(self.version)
            )

        appliances = {}

        appliances["name"] = np.array(df["Unnamed: 3"])

        appliances["type"] = crest_appname_to_demod_type(df["Unnamed: 3"])

        appliances["equipped_prob"] = np.array(df["Unnamed: 4"])

        associated_activity_use_profile = np.array(df["Unnamed: 5"])
        appliances["related_activity"] = crest_act_to_demod(
            associated_activity_use_profile
        )

        appliances["inactive_switch_off"] = np.where(
            np.logical_or(
                associated_activity_use_profile == "ACT_WASHDRESS",
                associated_activity_use_profile == "LEVEL",
                appliances["name"] == "DISH_WASHER",
            ),
            False,
            True,
        )

        appliances["mean_duration"] = np.array(df["(min).1"])

        appliances["after_cycle_delay"] = np.array(df["(min).2"])

        appliances["switch_on_prob_crest"] = np.array(df["Unnamed: 28"])

        appliances["mean_elec_consumption"] = np.array(df["(W)"])
        appliances["mean_elec_consumption"][-4:] = 0.0  # remove water

        appliances["mean_water_consumption"] = np.array(
            df["(W)"]  # will be flow for water
        )
        appliances["mean_water_consumption"][:-4] = 0.0  # remove elec

        appliances["standby_consumption"] = np.array(df["(W).1"])

        mpf = np.array(df["Unnamed: 30"])  # needs to set some nan values to 1
        appliances["mean_power_factor"] = np.where(np.isnan(mpf), 1.0, mpf)

        appliances["number"] = len(appliances["name"])

        if self.version in (["v_2.2", "v_2.3.3"]):
            appliances["poisson_sampling_lambda_liters"] = np.zeros_like(
                appliances["mean_water_consumption"]
            )
            appliances["poisson_sampling_lambda_liters"][-1] = 73.3  # bath
            appliances["poisson_sampling_lambda_liters"][-2] = 25.7  # shower
            appliances["poisson_sampling_lambda_liters"][-3] = 2.3  # sink
            appliances["poisson_sampling_lambda_liters"][-4] = 2.3  # basin

            appliances["uses_water"] = np.zeros(
                appliances["number"], dtype=bool
            )
            appliances["uses_water"][-4:] = True

        return appliances

    def _parse_activity_profiles(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        check_weekend_day_format(subgroup)
        if is_weekday(subgroup):
            head = 28
        elif is_weekend(subgroup):
            head = 28 + 36
        else:
            raise ValueError(
                "Invalid subgroup['day_type']: {} ".format(
                    subgroup["day_type"]
                )
            )

        df = pd.read_excel(
            self.raw_file_path,
            "ActivityStats",
            # For a wierd reasons, some lines are ignored in v2.2
            header=head,
            nrows=36,
            usecols=np.arange(144) + 4,
            engine="openpyxl",
        )

        pdf = np.array(df).T.reshape((144, 6, 6))

        labels_df = pd.read_excel(
            self.raw_file_path,
            "ActivityStats",
            header=28,
            nrows=6,
            usecols=[3],
            engine="openpyxl",
        )

        labels = [
            lab.upper() for lab in labels_df["Unnamed: 3"]
        ]  # converts to list

        # adds an activity that can always happen (pdf) = 1
        labels.append("LEVEL")
        pdf = np.concatenate((pdf, np.ones((144, 6, 1))), axis=2)
        # adds an activity that happens when there is an active occupant
        labels.append("ACTIVE_OCC")
        act_occ_pdf = np.ones((144, 6, 1))
        act_occ_pdf[:, 0, :] = 0
        pdf = np.concatenate((pdf, act_occ_pdf), axis=2)
        # Axis0: time
        # Axis1: act_occ
        # Axis2: activity

        demod_labels = crest_act_to_demod(labels)

        activity_profiles = {
            lab: profile
            for lab, profile in zip(demod_labels, np.moveaxis(pdf, -1, 0))
        }
        return activity_profiles

    def _parse_appliance_ownership_dict(
        self, subgroup: Subgroup
    ) -> Union[np.ndarray, Dict[str, float]]:
        crest_dic = self.load_appliance_dict()
        warnings.warn(
            "CREST cannot distinguish appliances ownership based on subgroups."
            "Ownership will be the same for all subgroups."
        )
        app_ownership = {}
        for app_type, ownership in zip(
            crest_dic["type"], crest_dic["equipped_prob"]
        ):

            # checks for multiply represented appliances
            counter = 1
            temp_type = app_type
            while temp_type in app_ownership:
                counter += 1
                temp_type = app_type + "_" + str(counter)

            app_ownership[temp_type] = ownership

        return app_ownership

    def _parse_clearness_tpms(self) -> Tuple[TPMs, StateLabels, timedelta]:
        if self.version in ("v_2.2", "v_2.3.3"):

            df = pd.read_excel(
                self.raw_file_path,
                "ClearnessIndexTPM",
                header=8,
                nrows=101,
                usecols=np.arange(2, 2 + 101),
                engine="openpyxl",
            )

            # step size is always one minute
            step_size = timedelta(minutes=1)

            labels = np.array(df.columns)
            labels[labels == "Clear"] = 1.0
            labels = np.array(labels, dtype=float)

            clearness_tpm = df.to_numpy()
            # corrects a bit the values of clearness TPM for round off erros
            # and non existing state
            clearness_tpm[0, 1] = 1
            clearness_tpm = rescale_pdf(clearness_tpm)

        else:
            raise NotImplementedError(
                "'_parse_clearness_tpms' is not implemented for CREST "
                "version {}.".format(self.version)
            )

        return clearness_tpm, labels, step_size

    def _parse_geographic_data(self) -> Dict[str, Union[str, float]]:
        if self.version == "v_2.2":
            return {
                "country": "england",
                "latitude": 52.8,
                "longitude": -1.2,
                "meridian": 0.0,
                "use_daylight_saving_time": True,
            }
        elif self.version == "v_2.3.3":
            return {
                "country": "india",
                "latitude": 13.1,
                "longitude": 80.3,
                "meridian": 82.5,
                "use_daylight_saving_time": False,
            }
        else:
            raise NotImplementedError(
                "'_parse_geographic_data' is not implemented for CREST "
                "version {}.".format(self.version)
            )

    def _parse_temperatures_arma(self) -> Dict[str, float]:
        if self.version == "v_2.2":
            arma_dic = {
                "T_mean": 9.3,
                "T_std": 6.5,
                "T_shift": -115,
                "AR": 0.81,
                "MA": 0.62,
                "SD": 0.5,
            }
        else:
            raise NotImplementedError(
                "'_parse_temperatures_arma' is not implemented for CREST "
                "version {}.".format(self.version)
            )
        return arma_dic

    def _parse_bulbs_config(self, subgroup: Subgroup) -> np.ndarray:
        if len(subgroup) > 0:
            # Warns that subgroup won't be used.
            warnings.warn(
                "Crest does not distinguish bulbs config" "based on subgroups."
            )
        if self.version in ("v_2.2", "v_2.3.3"):
            df = pd.read_excel(
                self.raw_file_path,
                header=9,
                sheet_name="bulbs",
                engine="openpyxl",
            )
            # Select the columns of interest
            bulbs_config = df.to_numpy()[:100, 2:40]
        else:
            raise NotImplementedError(
                "'_parse_bulbs_config' is not implemented for CREST "
                "version {}.".format(self.version)
            )
        return bulbs_config

    def _parse_crest_lighting(self) -> Dict[str, Any]:
        crest_dict = {}

        df = pd.read_excel(
            self.raw_file_path,
            header=23,
            sheet_name="light_config",
            engine="openpyxl",
        )

        if self.version == "v_2.2":
            crest_dict["calibration_scalar"] = float(df.columns[5])
        elif self.version == "v_2.3.3":
            crest_dict["calibration_scalar"] = float(df.columns[6])
        else:
            raise NotImplementedError(
                "'_parse_crest_lighting' is not implemented for CREST "
                "version {}.".format(self.version)
            )

        df = pd.read_excel(
            self.raw_file_path,
            header=35,
            sheet_name="light_config",
            engine="openpyxl",
        )
        crest_dict["effective_occupancy"] = np.array(
            df["occupancy"][:6], dtype=float
        )

        df = pd.read_excel(
            self.raw_file_path,
            sheet_name="light_config",
            header=53,
            engine="openpyxl",
        )
        crest_dict["durations_cdf"] = np.array(
            df["probability"][:9], dtype=float
        )
        crest_dict["durations_minutes_low"] = np.array(
            df["(minutes)"][:9], dtype=float
        )
        crest_dict["durations_minutes_high"] = np.array(
            df["(minutes).1"][:9], dtype=float
        )

        df = pd.read_excel(
            self.raw_file_path,
            sheet_name="light_config",
            header=3,
            engine="openpyxl",
        )
        crest_dict["irradiance_threshold_mean"] = float(df.columns[5])
        crest_dict["irradiance_threshold_std"] = float(df.columns[6])

        return crest_dict

    def _parse_installed_bulbs_stats(self, subgroup: Subgroup):
        warnings.warn(
            DATASET_CANNOT_DISTINGUISH_ON_SUBGROUPS.format(
                dataset=self,
                not_distinguishable='bulbs stats',
            )
        )
        stats, _ = bulbs_stats_from_config(self.load_bulbs_config(subgroup))
        return stats

    def _parse_bulbs(self):
        _, bulbs = bulbs_stats_from_config(self.load_bulbs_config())
        return bulbs

    def _parse_heating_system_dict(self, subgroup: Subgroup = {}):
        warnings.warn(
            DATASET_CANNOT_DISTINGUISH_ON_SUBGROUPS.format(
                dataset=self,
                not_distinguishable='heating system',
            )
        )
        if self.version == "v_2.2":
            rows = 3
        elif self.version == "v_2.3.3":
            rows = 5
        else:
            raise NotImplementedError(
                NOT_IMPLEMENTED_IN_DATASET_FOR_VERSION.format(
                    not_implemented=self._parse_heating_system_dict,
                    dataset=self,
                    version=self.version
                )
            )

        df = pd.read_excel(
            self.raw_file_path,
            sheet_name='PrimaryHeatingSystems',
            header=1, nrows=rows, skiprows=[2, 3]
        )

        heating_systems = {}
        heating_systems["equipped_prob"] = np.array(
             df["Proportion of dwellings with this heating system"]
        )
        heating_systems["name"] = np.array(df["Type of heating unit"])
        heating_systems["is_combi"] = np.array(df["Type of system"]) == 2
        heating_systems["fuel_type"] = remove_spaces(df["Type of fuel"])
        heating_systems["fuel_flow_rate"] = np.array(
            df["Fuel flow rate (nominal)"]
        )
        heating_systems["flow_rate_to_W"] = np.array(
            df["Calorific value of fuel flow rate"]
        )
        # Same heat outputs sh and dhw
        heating_systems["heat_output_sh"] = np.array(
            df["Heat output of unit"]
        )
        heating_systems["heat_output_dhw"] = np.array(
            df["Heat output of unit"]
        )
        heating_systems["standby_power"] = np.array(df["Standby power"])
        heating_systems["pump_power"] = np.array(df["Pump power"])
        heating_systems["cyl_volume"] = np.array(
            df["DHW cylinder volume"]
        )
        heating_systems["cyl_loss"] = np.array(
            df["DHW Tank Heat Loss"]
        )

        return heating_systems


    def _parse_thermostat_dict(self, subgroup: Subgroup):
        warnings.warn(
            DATASET_CANNOT_DISTINGUISH_ON_SUBGROUPS.format(
                dataset=self,
                not_distinguishable='thermostat dict',
            )
        )

        df_home = pd.read_excel(
            self.raw_file_path,
            sheet_name='HeatingControls',
            header=3, nrows=15
        )
        df_water = pd.read_excel(
            self.raw_file_path,
            sheet_name='HeatingControls',
            header=22, nrows=12
        )

        thermostat = {}
        thermostat["home_temperatures_cdf"] = np.cumsum(
            np.array(df_home["Percentage of homes"])
        )
        thermostat["home_temperatures_values"] = np.array(
            df_home["Demand temperature"], dtype=float
        )
        thermostat["water_temperatures_cdf"] = np.cumsum(
            np.array(df_water["Percentage of homes"])
        )
        thermostat["water_temperatures_values"] = np.array(
            df_water["Hot water delivery temperature"], dtype=float
        )



        # crest value for the emitters
        thermostat["emitter_setpoints"] = 50.0

        # Set thermostat deadbands
        deadbands = {}
        deadbands["space_heating"] = 2
        deadbands["space_cooling"] = 2
        deadbands["hot_water"] = 5
        deadbands["emitter"] = 5
        thermostat["deadband"] = deadbands

        return thermostat

    def _parse_buildings_dict(self, subgroup: Subgroup):

        df = pd.read_excel(
            self.raw_file_path,
            sheet_name='Buildings',
            header=1, skiprows=[2, 3]
        )

        buildings = {}
        buildings["equipped_prob"] = np.array(
            df["Proportion of dwellings of this building type"]
        )
        buildings["name"] = np.array(df["Description"])
        buildings["out_build_transfer_coef"] = np.array(
            df[
                "Thermal transfer coefficient between outside air and external building thermal capacitance"
            ]
        )
        buildings["build_int_transfer_coef"] = np.array(
            df[
                "Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance"
            ]
        )
        buildings["ventilation_transfer_coef"] = np.array(
            df[
                "Thermal transfer coefficient representing ventilation heat loss between outside air and internal building thermal capacitance"
            ]
        )
        buildings["ext_capacitance"] = np.array(
            df["External building thermal capacitance"]
        )
        buildings["int_capacitance"] = np.array(
            df["Internal building thermal capacitance"]
        )
        buildings["irradiance_multiplier"] = np.array(
            df["Global irradiance multiplier"]
        )
        buildings["floor_area"] = np.array(df["Floor area, living space"])
        buildings["height"] = np.array(df["Height, living space"])

        buildings["emitters_target_temperature"] = np.array(
            df["Nominal temperature of emitters "]
        )
        buildings["emitters_transfer_coef"] = np.array(
            df["Heat transfer coefficient of emitters"]
            if self.version.startswith('v_2.')
            else df["Heat transfer coefficient of heat emitters"]
        )
        buildings["emitters_capacitance"] = np.array(
            df["Thermal capacitance of emitters"]
        )

        return buildings


    def _parse_controls_tpm(self):
        # TODO: OlD implementation need to change
        path = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.3.3.xlsm - HeatingControlsTPM.csv"
        )
        df_heating = pd.read_csv(
            path, header=6, nrows=96, usecols=[2, 3, 4, 5]
        )
        thermostat["transitions cdf wd"] = np.cumsum(
            df_heating.to_numpy()[:, :2].reshape((48, 2, 2)), axis=-1
        )
        thermostat["transitions cdf we"] = np.cumsum(
            df_heating.to_numpy()[:, 2:].reshape((48, 2, 2)), axis=-1
        )

    def _parse_real_profiles_dict(self, profiles_type: str):
        """Parse the real profiles.

        Returns:
            The appliance dictionary, of the form
            {app_type: {app_name: array}}, such that it is easy to retrieve
            the profiles based on the type of the appliances
        """
        if profiles_type != 'switchedON':
            raise NotImplementedError('CREST only has switched on profiles.')
        return {
            'washingmachine': {
                'CREST_washingmachine':
                self._compute_washing_machine_power(np.arange(138, dtype=int))
            },
            'washer_dryer': {
                'CREST_washer_dryer':
                self._compute_washing_machine_power(np.arange(198, dtype=int))
            }
        }

    def _compute_washing_machine_power(self, current_time):
        """Washing machine power for a minute based simulation."""
        power = np.zeros_like(current_time, dtype=float)
        # define the values of power depending on n_times left, form CREST
        power[current_time <= 8] = 73  # Start-up and fill
        power[
            np.logical_and(current_time > 8, current_time <= 29)
        ] = 2056  # Heating
        power[
            np.logical_and(current_time > 29, current_time <= 81)
        ] = 73  # Wash and drain
        power[
            np.logical_and(current_time > 81, current_time <= 92)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 92, current_time <= 94)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 94, current_time <= 105)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 105, current_time <= 107)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 107, current_time <= 118)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 118, current_time <= 120)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 120, current_time <= 131)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 131, current_time <= 133)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 133, current_time <= 138)
        ] = 568  # Fast spin
        power[
            np.logical_and(current_time > 138, current_time <= 198)
        ] = 2500  # Drying cycle
        return power
