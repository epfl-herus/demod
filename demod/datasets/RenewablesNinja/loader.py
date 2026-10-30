"""Ninja Renewable  https://www.renewables.ninja/.

Thanks to Open-Power-System-Data for their code, which is used here.
https://github.com/Open-Power-System-Data/weather_data/blob/master/download.ipynb
"""

from datetime import timedelta
import os
from typing import Any, Dict
import urllib.request

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

    Available data:

    - 'datetime' in UTC
    - 'precipitation'
    - 'snowfall'
    - 'snow_mass'
    - 'clearness'
    - 'air_density'
    - 'outside_temperature'
    - 'irradiance'

    Parameters:
        weighted_type: The method used to weight the climate.
            This was performed by Renewables.ninja. Can be
            'population' or 'land_area'.
        update_raw_data: Wether the raw data file should be updated.
            As time goes by, new data might be collected by
            Renewable Ninjas.

    Loaders:
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`

    """

    DATASET_NAME = 'RenewablesNinja'
    # Default step size from ninja
    step_size = timedelta(hours=1)

    # Renewables.ninja now serves one csv file per climate variable.
    # These are downloaded and merged into a single raw file whose
    # columns are named after these variables.
    VARIABLES = [
        'temperature',
        'precipitation',
        'snowfall',
        'snow_mass',
        'cloud_cover',
        'air_density',
        'irradiance_surface',
        'irradiance_toa',
    ]
    # Maps the weighting type name used in demod to the one used in the
    # Renewables.ninja urls.
    _WEIGHTING_URL_NAMES = {
        'population': 'pop_wtd',
        'land_area': 'area_wtd',
    }

    def __init__(
        self, country_name,
        update_raw_data: bool = False,
        weighted_type: str = 'population',
        **kwargs
    ) -> Any:
        """Initialize the climate loader for the country.

        If update_raw_data, the raw data will be acutualized and parsed
        again, only for the selected country.
        """
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
        self.raw_file_path = os.path.join(
            self.raw_path, country_code + '_' + weighted_type + '.csv'
        )
        # Check if the raw file already exists and should not be updated
        if os.path.isfile(self.raw_file_path) and not update_raw_data:
            return
        # Else download every variable and merge them in a single file
        weighting = self._WEIGHTING_URL_NAMES.get(
            weighted_type, weighted_type
        )
        base_url = 'https://www.renewables.ninja/country_downloads/'
        variable_url_template = (
            '{country}/ninja-weather-country-{country}-{variable}'
            '_{weighting}-merra2.csv'
        )
        print('Downloading {} raw data. This can take some time.'.format(
            self.DATASET_NAME
        ))
        merged = None
        for variable in self.VARIABLES:
            country_url = base_url + variable_url_template.format(
                country=country_code,
                variable=variable,
                weighting=weighting,
            )
            variable_df = self._download_variable(country_url, country_code)
            variable_df = variable_df.rename(
                columns={variable_df.columns[1]: variable}
            )
            merged = (
                variable_df if merged is None
                else merged.merge(variable_df, on='time', how='outer')
            )
        merged.to_csv(self.raw_file_path, index=False)
        print('download finished')
        # Now the file is downloaded, we can remove old parsed data
        for f in os.listdir(self.parsed_path_climate):
            os.remove(os.path.join(self.parsed_path_climate, f))

    def _download_variable(
        self, url, country_code
    ) -> pd.DataFrame:
        """Download a single climate variable and keep the country column.

        Renewables.ninja serves one csv file per climate variable. These
        files start with three commented metadata lines, followed by a
        header row and the hourly values for every country zone. Only the
        ``time`` column and the country level column are kept.

        Args:
            url: The url of the csv file to download.
            country_code: The ISO2 code of the country.

        Returns:
            A dataframe with a ``time`` column and a single value column.
        """
        user_agent = (
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) '
            'Gecko/2009021910 Firefox/3.0.7'
        )
        headers = {'User-Agent': user_agent}
        try:
            request = urllib.request.Request(url, None, headers)
            with urllib.request.urlopen(request) as response:
                return pd.read_csv(
                    response, skiprows=3, usecols=['time', country_code]
                )
        except ValueError:
            # The country code is not a column of this file: fall back on
            # the first value column, the country level aggregate.
            request = urllib.request.Request(url, None, headers)
            with urllib.request.urlopen(request) as response:
                df = pd.read_csv(response, skiprows=3)
            return df[['time', df.columns[1]]]





    def _parse_historical_climate_data(self) -> Dict[str, np.ndarray]:
        """Parse the historical climate data.

        Returns:
            climate_dict: climate_dict with keys:
                - 'datetime' in UTC
                - 'precipitation'
                - 'snowfall'
                - 'snow_mass'
                - 'clearness'
                - 'air_density'
                - 'outside_temperature'
                - 'irradiance'
        """
        df = pd.read_csv(self.raw_file_path)

        out_dict = {}

        out_dict['datetime'] = np.array(
            df['time'],
            dtype='datetime64'
        )

        out_dict['precipitation'] = np.array(df['precipitation'])
        out_dict['snowfall'] = np.array(df['snowfall'])
        out_dict['snow_mass'] = np.array(df['snow_mass'])

        out_dict['clearness'] = np.array(df['cloud_cover'])

        out_dict['air_density'] = np.array(df['air_density'])

        out_dict['outside_temperature'] = np.array(df['temperature'])

        radiation_direct = np.array(df['irradiance_surface'])
        radiation_diffuse = np.array(df['irradiance_toa'])
        # https://meteonorm.meteotest.ch/en/faq/definition-of-direct-and-diffuse-radiation
        out_dict['irradiance'] = radiation_diffuse + radiation_direct

        return out_dict
