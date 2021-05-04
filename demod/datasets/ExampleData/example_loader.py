from typing import Union, Tuple, List, Dict, Any
import warnings
from ..base_loader import DatasetLoader

import os
import pandas as pd
import numpy as np


class ExampleLoader(DatasetLoader):
    """Example loader object.

    You can follow this example to create your new data loader.

    There are only 2 types of methods you need to create:
    - get_ methods, which return the requested data
    - _parse_ methods, (underscore before means private to the loader)
        that load the data from the raw files and transform it in
        numpy arrays

    You can define your own logic for logic for loading the data, but 
    try to follow these principles:
        - The data is split between raw_data and parsed_data
        - raw_data is the data set you have and it won't be added to git
        - parsed_data stores np.ndarrays
        - create specific get_ methods for different parts of the
            dataset
        - when getting some data, try first to read the parsed_data,
            and if not available, try to parse from the raw_data
        - after you have parsed the raw_data, save it as parsed_data
        - Use the helper methods from parent class 'DatasetLoader'
    """
    # Name of the folder containing your data
    DATASET_NAME:str = 'ExampleData'  
    # Names used by your data loader to load/store files
    example1_file_name:str = 'example_1'
    example2_file_name:str = 'example_2'
    
    def get_example1_data(
            self, return_ex2=False,
            ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[Any, Any]]]]:
        """Load the example data.

        You can customize the loading of the data using additional
        keyword arguments.

        
        """
        
        try:
            # first try to load from the parsed data
            array1 = self._load_parsed_data(self.example1_file_name)
            # loads ex2 only on condition
            if return_ex2:
                array2 = self._load_parsed_data(self.example2_file_name)
        except Exception as e:
            # if there was an issue loading parsed data, show warning
            self._warn_could_not_load_parsed(e, self.example1_file_name)
            # parse the data from the raw file
            array1, array2 = self._parse_example()

        # Here you can perform some operation on the data
        # ... (converting, reshaping, ...)


        # returns requested data
        if return_ex2:
            return array1, array2
        else:
            return array1
    


    def _parse_example(self):
        """Parsing the data should follow this process

        1. Defines the location of the raw_data
        2. Load the raw data
        3. Transfrom the raw data to parse data
        4. Saving the parsed data 
        5. Returning the data

        Returns:
            Any: The data
        """
        # name of the raw file
        raw_file_name = 'ex_one_1_file_txt.txt'
        try:
            # Creates the path of the raw data file
            data_path = os.path.join(self.raw_path, raw_file_name)
            # Loads the data, you can choose the most adapted for your data
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            # Raise a special error for raw_file missing, you can add downlad
            # instructions if your raw data is available online.
            self._raise_missing_raw(
                raw_file_name,
                optional_download_website='www.github.com',)

        # extract the data from the raw data
        array1 = data['col1'].to_numpy()
        array2 = data['col2'].to_numpy()

        # Save the data in the parsed data folder
        self._save_parsed_data(
            self.example1_file_name,
            array1,
            )
        self._save_parsed_data(
            self.example2_file_name,
            array2)

        return array1, array2
