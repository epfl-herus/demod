"""
Data loader for the german TOU survey.
"""
from datetime import time
import os
from typing import Tuple
import numpy as np
from urllib import request, error
import shutil
import zipfile

from ..tou_loader import LoaderTOU
from ..base_loader import GITHUB_REPO_URL, PopulationLoader
from ..DESTATIS.loader import Destatis
from ...utils.sim_types import *
from ...utils.monte_carlo import PDFs
from ...utils.sparse import SparseTPM
from ...utils.subgroup_handling import subgroup_string



class GTOU(LoaderTOU, PopulationLoader):
    """German Time-Of-Use Survey dataset loader.

    This loads data for different types of activity models.
    It can also split the households in different subgroups of the whole
    population.

    Currently implements activity_types:
        - 'Sparse9States'
        - '4_States'
        - 'DemodActivities_0'
        - 'Bottaccioli2018' https://doi.org/10.1109/ACCESS.2018.2886201

    """

    DATASET_NAME = 'GermanTOU'
    refresh_time = time(4, 0, 0)

    def __init__(
        self, activity_type: str = '4_States',
        **kwargs
    ) -> Any:
        self.activity_type = activity_type
        LoaderTOU.__init__(self, activity_type, **kwargs)
        PopulationLoader.__init__(self, **kwargs)

        if all((  # Downloads parsed folder if data does not exist
            (not self._raw_file_exists()),
            (not os.path.isfile(
                self.raw_path + os.sep + self.activity_type + '.zip'
            ))
        )):
            self._download_parsed_folder_github()

    def _clear_parsed_data(self):
        # clear the parsed directory
        shutil.rmtree(self.parsed_path + os.sep + self.activity_type)
        os.mkdir(self.parsed_path + os.sep + self.activity_type)
        # Also cleans the zip file
        parsed_zip_filepath = (
            self.raw_path + os.sep + self.activity_type + '.zip'
        )
        if os.path.isfile(parsed_zip_filepath):
            os.remove(parsed_zip_filepath)

    def _raw_file_exists(self):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_name, 'raw_data', 'zve13_puf_')
        return (
            os.path.isfile(file_path + 'hh.csv')
            & os.path.isfile(file_path + 'takt.csv')
            & os.path.isfile(file_path + 'pers.csv')
        )

    def _check_parsed_file_hosted_on_github(self):
        """Return True if the file is hosted on github."""
        url = ''.join((
            GITHUB_REPO_URL, '/datasets/GermanTOU/parsed_data/',
            str(self.activity_type), '.zip'
        ))
        try:
            request.urlopen(url)
        except error.HTTPError as e:
            if e.code == 404:
                return False
            else:
                raise e
        return True

    def _download_parsed_folder_github(self):

        url = ''.join((
            'https://raw.githubusercontent.com/epfl-herus/demod/master',
            '/demod/datasets/GermanTOU/parsed_data/',
            self.activity_type, '.zip'
        ))
        parsed_zip_filepath = (
            self.raw_path + os.sep + str(self.activity_type) + '.zip'
        )
        print('Downloading {} parsed data for {} from {}.'.format(
            self.DATASET_NAME, self.activity_type, url
        ))
        # Reads the url and  download the zip archive
        response = request.urlopen(url)
        datatowrite = response.read()
        with open(parsed_zip_filepath, 'wb') as f:
            f.write(datatowrite)
        print('end_download')
        # Now the file is downloaded
        with zipfile.ZipFile(parsed_zip_filepath, 'r') as zip_obj:
            zip_obj.extractall(self.parsed_path_activity)


    def _parse_tpm(self, subgroup: Subgroup):
        if self.activity_type == '4_States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_4states
            return get_data_4states(subgroup)
        elif self.activity_type == 'DemodActivities_0':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_activity
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='last',
                add_away_state=True,
                add_durations=False,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_Bottaccioli_act
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_Bottaccioli_act,
                first_tpm_modification_algo='last',
                add_away_state=True,
                add_durations=False,
            )
        else:
            err = NotImplementedError(("No parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

    def _parse_tpm_with_duration(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, np.ndarray, StateLabels, PDF, PDFs, dict]:
        if self.activity_type == '4_States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_4states
            return get_data_4states(subgroup, add_durations=True)
        elif self.activity_type == 'DemodActivities_0':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_activity
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_activity,
                first_tpm_modification_algo='nothing',
                add_away_state=True,
                add_durations=True,
            )
        elif self.activity_type == 'Bottaccioli2018':
            # Imports the raw data parser only here to save time
            from .parser import get_tpms_activity, GTOU_label_to_Bottaccioli_act
            return get_tpms_activity(
                subgroup,
                activity_dict=GTOU_label_to_Bottaccioli_act,
                first_tpm_modification_algo='nothing',
                add_away_state=True,
                add_durations=True,
            )
        else:
            err = NotImplementedError(("No parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

    def _parse_sparse_tpm(self, subgroup: Subgroup) -> Tuple[
        SparseTPM, StateLabels,
        ActivityLabels, np.ndarray]:

        if self.activity_type == 'Sparse9States':
            # Imports the raw data parser only here to save time
            from .parser import get_data_sparse9states
            return get_data_sparse9states(subgroup)
        else:
            err = NotImplementedError(("No sparse parsing defined for" +
                "'{}' in dataset '{}'").format(
                    self.activity_type, self.DATASET_NAME
                ))
            raise err

    def _parse_activity_profiles(
        self, subgroup: Subgroup
    ) -> Dict[str, np.ndarray]:
        from .parser import create_data_activity_profile
        return create_data_activity_profile(subgroup)

    def load_population_subgroups(
        self, population_type: str
    ) -> Tuple[Subgroups, List[float], int]:
        # GTOU is the same as Germany so we can use Destatis for the population
        data = Destatis()
        return data.load_population_subgroups(population_type)
