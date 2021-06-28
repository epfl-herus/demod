"""Loader for the excell input file that can be customized by the user.

Reads the excell file provided.
Each sheet correspond to a part of the load simulation.
"""


from demod.utils.sparse import SparseTPM
from demod.datasets.tou_loader import LoaderTOU
import numpy as np
from demod.utils.parse_helpers import remove_spaces
import os
from typing import Any, Dict, Tuple

import pandas as pd
from demod.utils.sim_types import (
    ActivityLabels,
    AppliancesDict,
    StateLabels,
    Subgroup,
)
from ..base_loader import ApplianceLoader, HeatingLoader, LightingLoader


class InputFileLoader(ApplianceLoader, LoaderTOU, LightingLoader, HeatingLoader):
    """Dataset loader for an input excel spreadsheet.

    An example of the requested file is provided in this folder as
    inputs.xlsx .

    Attrs:
        raw_file_path: The path at which the excell file is located.
    """

    raw_file_path: str
    tou_loader: LoaderTOU

    def __init__(self, raw_file_path=None) -> Any:
        if raw_file_path is None:
            raw_file_path = os.path.dirname(__file__) + os.sep + "inputs.xlsx"

        if not os.path.isfile(raw_file_path):
            raise FileNotFoundError(
                "Could not find excel input file at {}".format(raw_file_path)
            )

        self.raw_file_path = raw_file_path

    def _read_tou_loader(self):

        df = pd.read_excel(
            self.raw_file_path,
            "Appliances",
            skiprows=14,
            engine="openpyxl",
        )

    def load_appliance_dict(self) -> AppliancesDict:

        df = pd.read_excel(
            self.raw_file_path,
            "Appliances",
            skiprows=14,
            engine="openpyxl",
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

        appliances["number"] = len(appliances["name"])

        return appliances

    def load_crest_lighting(self) -> Dict[str, Any]:
        crest_dict = {}

        df = pd.read_excel(
            self.raw_file_path,
            header=23, sheet_name="Lighting",
            engine="openpyxl",
        )

        crest_dict['calibration_scalar'] = float(df.columns[5])

        df = pd.read_excel(
            self.raw_file_path,
            header=35, sheet_name="Lighting",
            engine="openpyxl",
        )
        crest_dict['effective_occupancy'] = (
            np.array(df["occupancy"][:6], dtype=float)
        )


        df = pd.read_excel(
            self.raw_file_path,
            sheet_name="Lighting",
            header=53,
            engine="openpyxl",
        )
        crest_dict['durations_cdf'] = (
            np.array(df["probability"][:9], dtype=float)
        )
        crest_dict['durations_minutes_low'] = (
            np.array(df["(minutes)"][:9], dtype=float)
        )
        crest_dict['durations_minutes_high'] = (
            np.array(df["(minutes).1"][:9], dtype=float)
        )

        df = pd.read_excel(
            self.raw_file_path,
            sheet_name="Lighting",
            header=3,
            engine="openpyxl",
        )
        crest_dict['irradiance_threshold_mean'] = float(df.columns[5])
        crest_dict['irradiance_threshold_std'] = float(df.columns[6])

        return crest_dict

    def load_bulbs(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_excel(
            self.raw_file_path, header=75,
            sheet_name="Lighting", engine="openpyxl",
        )
        bulbs_penetration = df["Penetration"].to_numpy()
        bulbs_consumption = df["Consumption[W]"].to_numpy()
        return bulbs_consumption, bulbs_penetration

    def load_bulbs_config(self) -> np.ndarray:
        df = pd.read_excel(
            self.raw_file_path, header=9,
            sheet_name="bulbs", engine="openpyxl",
        )
        return df.to_numpy()[:, 2:]

    def load_installed_bulbs_stats(self) -> Tuple[float, float]:
        df = pd.read_excel(
            self.raw_file_path, header=68,
            sheet_name="Lighting", engine="openpyxl",
        )
        return df["Unnamed: 3"][0], df["Unnamed: 3"][1]

    def load_buildings_dict(self, subgroup: Subgroup) -> Dict[str, np.ndarray]:
        df = pd.read_excel(
            self.raw_file_path,
            "Buildings",
            skiprows=4,
            engine="openpyxl",
        )

        buildings = {
            key: df[key]
            for key in df.columns
            if not str(key).startswith("Unnamed:")
        }
        buildings = remove_spaces(buildings)

        buildings["number"] = len(buildings["name"])

        return buildings

    def load_heating_system_dict(self, subgroup: Subgroup) -> Dict[str, np.ndarray]:
        df = pd.read_excel(
            self.raw_file_path,
            "Heating Systems",
            skiprows=4,
            engine="openpyxl",
        )

        heating_systems = {
            key: df[key]
            for key in df.columns
            if not str(key).startswith("Unnamed:")
        }
        heating_systems = remove_spaces(heating_systems)

        heating_systems["number"] = len(heating_systems["name"])

        return heating_systems


    def load_sparse_tpm(
        self, subgroup: Subgroup
    ) -> Tuple[SparseTPM, StateLabels, ActivityLabels, np.ndarray]:
        return self.tou_loader.load_sparse_tpm(subgroup)

    def load_tpm(self, subgroup: Subgroup):
        return self.tou_loader.load_tpm(subgroup)

    def load_appliance_ownership_dict(
        self, subgroup: Subgroup
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def load_yearly_target_switchons(
        self, subgroup: Subgroup = {},
    ) -> Dict[str, float]:
        app_dict = self.load_appliance_dict()
        return {
            app_type: target_switchon for app_type, target_switchon
            in zip(app_dict['type'], app_dict['target_cycle_year'])
        }
