"""Ninja Renewable  https://www.renewables.ninja/.

Thanks to Open-Power-System-Data for their code, which is used here.
https://github.com/Open-Power-System-Data/weather_data/blob/master/download.ipynb
"""

from datetime import timedelta
import os
from typing import Any, Dict
import urllib.request
import shutil
import zipfile

import numpy as np
import pandas as pd
from ..base_loader import ClimateLoader
from ...utils.countries import country_name_to_code, is_country_code


class NinjaRenewablesClimate(ClimateLoader):
    """Loader of the climate.

    Data comes from
    `Ninja Renewable <https://www.renewables.ninja/>`_
    The raw datasets are downloaded on demand by this dataloader.
    It corresponds to MERRA-2(global).

    Attributes:
        weighted_type: The method used to weight the climate.
            This was performed by Renewables.ninja. Can be
            'population' or 'land_area'.

    Loaders:
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`

    """

    DATASET_NAME = 'RenewablesNinja'
    step_size = timedelta(hours=1)

    def __init__(
        self, country_name,
        clear_parsed_data: bool = False,
        update_raw_data: bool = False,
        weighted_type: str = 'population',
        **kwargs
    ) -> Any:
        """Initialize the climate loader for the country."""
        super().__init__(**kwargs)
        # Creates the path for the requested country
        self.parsed_path_climate = os.path.join(
            self.parsed_path_climate, country_name)
        # creates the parsed path
        if not os.path.exists(self.parsed_path_climate):
            os.mkdir(self.parsed_path_climate)
        # Creates the raw path
        if not os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)

        self.country = country_name

        self._check_download_raw_file(update_raw_data, weighted_type)

    def _check_download_raw_file(self, update_raw_data, weighted_type):
        country_code = (
            country_name_to_code(self.country)
            if not is_country_code(self.country) else self.country
        )
        raw_file_path = os.path.join(
            self.raw_path, country_code + '_' + weighted_type + '.csv'
        )
        # Check if the raw file already exists and should not be updated
        if os.path.isfile(raw_file_path) and not update_raw_data:
            return
        # Else download it
        base_url = 'https://www.renewables.ninja/country_downloads/'
        country_url_template = (
            '{country}/ninja_weather_country_{country}_'
            'merra-2_{weighted_type}_weighted.csv'
        )
        country_url = base_url + country_url_template.format(
            country=country_code,
            weighted_type=weighted_type
        )
        print('Downloading tracebase raw data from {}.'.format(country_url))
        print('This can take some time.')
        # Creates the request
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent':user_agent,}
        request = urllib.request.Request(country_url, None, headers)

        # Reads the url and  download the
        with urllib.request.urlopen(request) as response:
            with open(raw_file_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        # Now the file is downloaded
        print('downloaded')



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