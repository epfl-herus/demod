"""Dataset loaders for Germany."""
import datetime
import os
from urllib import request

import numpy as np
import pandas as pd

from ...utils.sim_types import AppliancesDict, Subgroup, Subgroups
from ...utils.parse_helpers import remove_spaces
from typing import Any, Dict, List, Tuple
from ..base_loader import (
    ApplianceLoader,
    ClimateLoader,
    HeatingLoader,
    LightingLoader,
    PopulationLoader,
)
from ..ExcellInputFile.loader import InputFileLoader


from ..RenewablesNinja.loader import NinjaRenewablesClimate
from ..GermanTOU.loader import GTOU
from ..DESTATIS.loader import Destatis
from ..tracebase.loader import Tracebase


class GermanDataHerus(
    ApplianceLoader,
    LightingLoader,
    HeatingLoader,
    PopulationLoader,
    ClimateLoader,
):
    """Data for demod for Germany.

    This data originates from a Master project at HERUS lab, (EPFL).
    The data merges different sources:

    * :py:meth:`~demod.datasets.CREST.loader.Crest`
    * :py:meth:`~demod.datasets.GermanTOU.loader.GTOU`
    * :py:meth:`~demod.datasets.OpenPowerSystems.loader.OpenPowerSystemClimate`
    * :py:meth:`~demod.datasets.DESTATIS.loader.Destatis`

    The raw data is in some parts in an excell sheet.

    Parameters:
        version: A versionning system that can be used for
            modifications of the data.

    """

    DATASET_NAME = "Germany"
    # Assign a 1 min step size as it is the target of the simulation
    # Though GTOU is 10 minutes
    step_size = datetime.timedelta(minutes=1)

    def __init__(self, /, version: str = "v0.1", **kwargs) -> Any:
        """Create a dataset loader for HERUS data.

        Args:
            version: The version of the datset to load. Defaults to "v0.1".
        """
        if version == "v0.1":
            self.activity_data = GTOU("Sparse9States")
        elif version == "vBottaccioli":
            self.activity_data = GTOU("Bottaccioli2018")
        else:
            raise ValueError("Unkown version :'{}' for '{}'".format(
                version, self
            ))
        self.destatis = Destatis()
        self.climate = NinjaRenewablesClimate("germany")

        super().__init__(version=version, **kwargs)

        self.raw_file_path = (
            self.raw_path + os.sep + "data_" + version + ".xlsx"
        )

        if not os.path.isfile(self.raw_file_path):
            # Download file if does not exist
            url = (
                'https://raw.githubusercontent.com/epfl-herus/demod/master'
                '/demod/datasets/Germany/raw_data/data_{}.xlsx'.format(version)
            )
            print('Downloading {} excell sheet for {} from {}.'.format(
                self.DATASET_NAME, version, url
            ))
            response = request.urlopen(url)
            datatowrite = response.read()
            with open(self.raw_file_path, 'wb') as f:
                f.write(datatowrite)

        self.refresh_time = self.activity_data.refresh_time

    def load_historical_climate_data(
        self, start_datetime: datetime.datetime
    ) -> Dict[str, np.ndarray]:
        return self.climate.load_historical_climate_data(start_datetime)

    def load_population_subgroups(
        self,
        population_type: str = "household_types_2019",
    ) -> Tuple[Subgroups, List[float], int]:

        return self.destatis.load_population_subgroups(population_type)

    def load_activity_probability_profiles(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        return self.activity_data.load_activity_probability_profiles(subgroup)

    def load_tpm(self, *args, **kwargs):
        return self.activity_data.load_tpm(*args, **kwargs)

    def load_tpm_with_duration(self, *args, **kwargs):
        return self.activity_data.load_tpm_with_duration(*args, **kwargs)

    def load_sparse_tpm(self, subgroup: Subgroup):
        return self.activity_data.load_sparse_tpm(subgroup)

    def load_activity_probabilities(self, subgroup: Subgroup):
        return self.activity_data.load_activity_probabilities(subgroup)

    def load_daily_activity_starts(self, subgroup: Subgroup):
        return self.activity_data.load_daily_activity_starts(subgroup)

    def load_appliance_ownership_dict(self, subgroup: Subgroup) -> np.ndarray:
        return self.destatis.load_appliance_ownership_dict(subgroup)

    def load_real_profiles_dict(self, *args, **kwargs):
        self.tracebase_profiles = Tracebase()
        return self.tracebase_profiles.load_real_profiles_dict(*args, **kwargs)

    def _parse_appliance_dict(self) -> AppliancesDict:
        if self.version in ["v0.1", "vBottaccioli"]:

            df = pd.read_excel(
                self.raw_file_path,
                "Appliances",
                skiprows=14,
                engine="openpyxl",
            )

        else:
            raise NotImplementedError(
                "'_parse_appliance_dict' is not implemented for "
                "GermanDataHerus version {}.".format(self.version)
            )

        appliances = {
            key: df[key]
            for key in df.columns
            if not str(key).startswith("Unnamed:")
        }
        appliances = remove_spaces(appliances)

        if "inactive_switch_off" in appliances:
            appliances["inactive_switch_off"] = np.array(
                appliances["inactive_switch_off"], dtype=bool
            )
        if "uses_water" in appliances:
            appliances["uses_water"] = np.array(
                appliances["uses_water"], dtype=bool
            )
        if "probabilistic" in appliances:
            appliances["probabilistic"] = np.array(
                appliances["probabilistic"], dtype=bool
            )

        appliances["number"] = len(appliances["name"])

        return appliances

    def _parse_fisher_lighting(self) -> Dict[str, Any]:
        if self.version == "v0.1":
            return {
                "irradiation_threshold_min": 30,  # W/m^2
                "irradiation_threshold_max": 60,  # W/m^2
                "individual_light_use": 12,  # Watts
            }
        else:
            raise NotImplementedError(
                (
                    "'_parse_fisher_lighting' is not implemented for {}"
                    ", version '{}'."
                ).format(type(self).__name__)
            )

    def _parse_crest_lighting(self) -> Dict[str, Any]:
        return InputFileLoader.load_crest_lighting(self)

    def _parse_bulbs(self) -> Dict[str, Any]:
        return InputFileLoader.load_bulbs(self)

    def _parse_installed_bulbs_stats(self, subgroup: Subgroup):
        return InputFileLoader.load_installed_bulbs_stats(self)

    def _parse_buildings_dict(self, subgroup: Subgroup):
        return InputFileLoader.load_buildings_dict(self, subgroup)

    def _parse_heating_system_dict(self, subgroup: Subgroup):
        return InputFileLoader.load_heating_system_dict(self, subgroup)

    def _parse_yearly_target_switchons(
        self, subgroup: Subgroup
    ) -> Dict[str, float]:
        return InputFileLoader.load_yearly_target_switchons(self, subgroup)

    def _parse_thermostat_dict(self, subgroup: Subgroup):
        """Parse data from CREST."""
        thermostat = {}
        thermostat["home_temperatures_cdf"] = np.cumsum(
            np.array(
                [
                    0.01,
                    0.02,
                    0.01,
                    0.03,
                    0.04,
                    0.08,
                    0.16,
                    0.14,
                    0.16,
                    0.16,
                    0.09,
                    0.05,
                    0.03,
                    0.01,
                    0.01,
                ]
            )
        )
        thermostat["home_temperatures_values"] = np.array(
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            dtype=float,
        )
        thermostat["water_temperatures_cdf"] = np.cumsum(
            np.array(
                [
                    0.04,
                    0.04,
                    0.11,
                    0.08,
                    0.13,
                    0.09,
                    0.14,
                    0.11,
                    0.08,
                    0.10,
                    0.04,
                    0.04,
                ]
            )
        )
        thermostat["water_temperatures_values"] = np.array(
            [
                42,
                43,
                45,
                47,
                49,
                51,
                53,
                55,
                57,
                59,
                61,
                62,
            ],
            dtype=float,
        )

        # crest value for the emitters
        thermostat["emitter_setpoints"] = 50.0

        # Set thermostat deadbands
        deadbands = {}
        deadbands["space_heating"] = 1
        deadbands["hot_water"] = 5
        deadbands["emitter"] = 5
        thermostat["deadband"] = deadbands

        return thermostat
