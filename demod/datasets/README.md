This folder contains all the datasets.

All dataset contain two folder: one with raw data and one with parsed data
Raw data contains the basic data set.
The loaders will first try to load the parsed data, and if it does not work
it will then try to generate the data from the dataset.
The raw_data folder is not uploaded on the github repository.
The parsed_data should contain only the necessary infromation and can be
uploaded, to be used by the simulators at runtime.

base_loader.py contains a base_loader class that can be used as skeleton
to implement a loader for a data set.

