"""This folder contains the dataset from http://www.tracebase.org/ .

Contains information from the tracebase data set,
which is made available at http://www.tracebase.org
under the Open Database License (ODbL)

Andreas Reinhardt, Paul Baumann, Daniel Burgstahler,
Matthias Hollick, Hristo Chonov, Marc Werner, Ralf Steinmetz:
On the Accuracy of Appliance Identification Based on
Distributed Load Metering Data.
roceedings of the 2nd IFIP Conference on Sustainable Internet and
ICT for Sustainability (SustainIT), 2012.
`link
<https://www.areinhardt.de/publications/2012/Reinhardt_SustainIt_2012.pdf>`_
"""
from datetime import datetime, timedelta
import os
import urllib.request
import shutil
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from ..base_loader import ApplianceLoader


traceback_type_to_demod = {
    'Alarmclock': 'alarm_clock',
    'Amplifier': 'speaker',
    'BeanToCupCoffeemaker': 'beans_coffemachine',
    # 'Breadcutter': '',
    'CdPlayer': 'cd_speaker',
    # 'Charger-PSP': '',
    'Charger-Smartphone': 'smart_phone',
    'Coffeemaker': 'coffemachine',
    'Cookingstove': 'electric_hob',
    'DigitalTvReceiver': 'tv_box',
    'Dishwasher': 'dishwasher',
    'DvdPlayer': 'dvd_console',
    'EthernetSwitch': 'internet_box',
    'Freezer': 'freezer',
    'Iron': 'iron',
    'Lamp': 'lamp',
    'LaundryDryer': 'dryer',
    'MicrowaveOven': 'microwave',
    'Monitor-CRT': 'crt_monitor',
    'Monitor-TFT': 'tft_monitor',
    # 'Multimediacenter': '',
    'PC-Desktop': 'fixed_computer',
    'PC-Laptop': 'laptop_computer',
    'Playstation3': 'gaming_console',
    'Printer': 'printer',
    'Projector': 'projector',
    'Refrigerator': 'fridge',
    # 'RemoteDesktop': '',
    'Router': 'wifi_box',
    # 'SolarThermalSystem': '',
    'Subwoofer': 'hifi_speaker',
    'Toaster': 'toaster',
    'TV-CRT': 'crt_tv',
    'TV-LCD': 'lcd_tv',
    # 'USBHarddrive': '',
    # 'USBHub': '',
    'VacuumCleaner': 'vacuum_cleaner',
    'VideoProjector': 'projector',
    'Washingmachine': 'washingmachine',
    'WaterBoiler': 'waterboiler',
    # 'WaterFountain': '',
    'WaterKettle': 'kettle',
    'XmasLights': 'christmas_lamp',
}

class Tracebase(ApplianceLoader):
    """Dataset loader for the trace base dataset.

    The dataset is automatically downloaded form the github repo.

    It then parses the profiles to demod.

    It also resample the data to the desired step size using interpolation.

    Contains information from the tracebase data set,
    which is made available at http://www.tracebase.org
    under the Open Database License (ODbL).
    """

    DATASET_NAME = 'tracebase'

    def __init__(
        self, /,
        version: str = 'master',
        clear_parsed_data: bool = False,
        update_raw_data: bool = False,
        step_size: timedelta = timedelta(seconds=60)
    ) -> Any:
        """Create the dataset.

        Will check whether it should download the tracebase raw data.
        """
        super().__init__(
            version=version + '_step_size_{}_seconds'.format(
                int(step_size.total_seconds())
            ),
            # If the raw data must be cleared
            clear_parsed_data=clear_parsed_data or update_raw_data
        )
        self.step_size = step_size
        if not os.path.isdir(self.raw_path):
            os.mkdir(self.raw_path)
        raw_zip_filepath = self.raw_path + os.sep + 'raw_github.zip'
        # Check if it needs to download the raw data
        if update_raw_data:
            os.remove(raw_zip_filepath)
        if not os.path.isfile(raw_zip_filepath):
            download_url = 'https://github.com/areinhardt/tracebase/archive/refs/heads/master.zip'
            print('Downloading the raw data from {}.'.format(download_url))
            print('This can take some time.')
            with (  # Reads the url and
                urllib.request.urlopen(download_url) as response,
                open(raw_zip_filepath, 'wb') as f
            ):
                shutil.copyfileobj(response, f)
            # Now the file is downloaded
            with zipfile.ZipFile(raw_zip_filepath, 'r') as zip_obj:
                zip_obj.extractall(self.raw_path)

    def _parse_real_profiles_dict(self, profiles_type: str):
        if profiles_type != 'full':
            raise NotImplementedError(profiles_type)

        profiles_dict = {}
        appliances_path = os.path.join(
            self.raw_path, 'tracebase-master', 'complete'
        )

        traceback_app_types = os.listdir(appliances_path)
        # Iterates over the profiles
        for type_name in traceback_app_types:
            # Check that the type exists in demod
            if type_name in traceback_type_to_demod:
                # Finds the demod name
                demod_name = traceback_type_to_demod[type_name]
                if demod_name not in profiles_dict:
                    # Assign an empty dict
                    profiles_dict[demod_name] = {}
                profiles_filenames = os.listdir(
                    appliances_path + os.sep + type_name
                )
                for profile_name in profiles_filenames:
                    profiles_dict[demod_name][profile_name] = (
                        self._parse_single_profile(
                            os.path.join(
                                appliances_path, type_name, profile_name
                            ),
                            profiles_type
                        )
                    )

        return profiles_dict

    def _parse_single_profile(self, file_path, profiles_type):
        if profiles_type == 'full':
            df = pd.read_csv(
                file_path,
                parse_dates=[0], infer_datetime_format=True,
                sep=';', usecols=[0, 1], header=None
            )
            # sample on a step size
            seconds = np.array(
                3600 * df[0].dt.hour
                + 60 * df[0].dt.minute
                + df[0].dt.second,
                dtype=int
            )
            # desired axis
            seconds_interp = np.arange(
                seconds[0],
                seconds[-1],
                step=self.step_size.total_seconds(),
                dtype=int
            )
            return np.interp(seconds_interp, seconds, df[1])

