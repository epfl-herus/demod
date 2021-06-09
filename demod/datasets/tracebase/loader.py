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

    It also resample the data to the desired step size using interpolation
    and averageing over time.

    Note that the 'switchedON' profiles are not available for all
    appliances, only for
    dishwasher, washing machine, dryer, iron, microwave

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
            print('Downloading tracebase raw data from {}.'.format(download_url))
            print('This can take some time.')
            # Reads the url and  download the zip archive
            with  urllib.request.urlopen(download_url) as response:
                with open(raw_zip_filepath, 'wb') as f:
                    shutil.copyfileobj(response, f)
            # Now the file is downloaded
            with zipfile.ZipFile(raw_zip_filepath, 'r') as zip_obj:
                zip_obj.extractall(self.raw_path)

    def _parse_real_profiles_dict(self, profiles_type: str):
        if profiles_type == 'switchedON':
            return self._parse_real_profiles_dict_ON()

        elif profiles_type != 'full':
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
                step=1, # one second
                dtype=int
            )
            # Interpolate to get the load at each second step
            load_second = np.interp(seconds_interp, seconds, df[1])
            if int(self.step_size.total_seconds()) == 1:
                return load_second
            else:
                # Need to average over the load
                sec_avg = int(self.step_size.total_seconds())
                # Pad the array for resahping
                n_pad = sec_avg - (len(load_second) % sec_avg)
                padded_load = np.append(load_second, np.zeros(n_pad))
                # Compute the mean over the step size intervals
                return padded_load.reshape((-1, sec_avg)).mean(axis=-1)



    def _parse_real_profiles_dict_ON(self):
        # Those load profiles where looked at if it was possible to
        # crop them using only the start and end of consumption
        # Also they were selected by different devices or cycle pattern
        load_profiles_satisfying = [
            # dishwashers
            'dev_995BAC_2012.06.11.csv',
            'dev_B7E6F4_2012.02.03.csv',
            'dev_B7E6FA_2012.01.18.csv',
            'dev_B81D04_2012.05.23.csv',
            'dev_B82F81_2011.08.15.csv',
            # washing machines
            'dev_11F01E_2011.12.10.csv',
            'dev_7297E3_2012.01.16.csv',
            'dev_B8121D_2012.02.01.csv',
            'dev_D31FFD_2012.06.11.csv',
            'dev_D3230E_2011.12.18.csv',
            'dev_D338C9_2012.05.16.csv',
            'Washingmachine_2011.11.30.csv',
            # Dryer, (only one seemed to fit sadly)
            'dev_B7E43D_2012.01.24.csv',
            # Iron
            'dev_D337C2_2011.12.26.csv',
            # Microvawe
            'dev_995FCC_2012.01.21.csv',
            'dev_D32309_2011.12.25.csv',
            'dev_D32309_2012.01.08.csv',
            'Microwave_2011.12.28.csv',
            'Microwave_2011.12.30.csv',
            'Microwave_2012.01.02.csv',

        ]
        profiles_dict = self.load_real_profiles_dict('full')
        dict_ON_profiles = {}
        for app_type, load_dict in profiles_dict.items():
            for load_name, load in load_dict.items():
                if load_name in load_profiles_satisfying:
                    # Assume the load is where there are enough watts in profils
                    mask = np.where(load > 6)[0]
                    # Take the boundaries of where the on load is
                    a, b = mask[[0, -1]]
                    if app_type not in dict_ON_profiles:
                        dict_ON_profiles[app_type] = {}
                    # Adds the profile to the dictionary
                    dict_ON_profiles[app_type][load_name] = load[int(a):int(b+1)]
        return dict_ON_profiles
