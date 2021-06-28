"""Open Power System Data https://open-power-system-data.org/."""

from datetime import timedelta
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from ..base_loader import ClimateLoader
from ...utils.countries import country_name_to_code


class OpenPowerSystemClimate(ClimateLoader):
    """Loader of the climate.

    Deprectaed in demod version 0.2.
    Should use RenewablesNinja dataset instead.

    Data comes from
    `Open Power System Data  <https://open-power-system-data.org/>`_
    and the raw dataset can be downloaded
    `here <https://data.open-power-system-data.org/weather_data/>`_

    Loaders:
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`

    """

    DATASET_NAME = 'OpenPowerSystems'
    step_size = timedelta(hours=1)

    def __init__(self, country_name) -> Any:
        """Initialize the climate loader for the country."""
        super().__init__(allow_pickle=False, version=None)
        # Creates the path for the requested country
        self.parsed_path_climate = os.path.join(
            self.parsed_path_climate, country_name)
        # creates the parsed path
        if not os.path.exists(self.parsed_path_climate):
            os.mkdir(self.parsed_path_climate)

        self.country = country_name

    def _parse_historical_climate_data(self) -> Dict[str, np.ndarray]:
        """Parse the historical climate data.

        Returns:
            climate_dict: climate_dict with keys:
                - 'temperature'
                - 'radiation_global'
        """
        raw_file_path = os.path.join(self.raw_path, 'weather_data.csv')

        code = country_name_to_code(self.country)

        df = pd.read_csv(raw_file_path)

        out_dict = {}

        out_dict['datetime'] = np.array(
            df['utc_timestamp'],
            dtype='datetime64'
        )

        out_dict['outside_temperature'] = np.array(df[code + '_temperature'])

        radiation_direct = np.array(
            df[code + '_radiation_direct_horizontal']
        )
        radiation_diffuse = np.array(
            df[code + '_radiation_diffuse_horizontal']
        )

        # https://meteonorm.meteotest.ch/en/faq/definition-of-direct-and-diffuse-radiation
        out_dict['irradiance'] = radiation_diffuse + radiation_direct

        return out_dict
