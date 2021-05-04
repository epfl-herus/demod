============
Datasets
============

This tutorial will show you how to create your custom dataset loader.

It requires that you know
:ref:`how to use a dataset in a simulation <changing_the_dataset>`.



Dataset API
-----------


Demod provide an API that allows to very simply choose which Dataset
is used during the simulation. This flexibility requires the
DatasetLoader objects to follow some specific rules.

A DatasetLoader must implement some loading methods that
are called by the Simulators.
The loading methods must return the data as specified
in the
:doc:`load methods documentation <../api/data_api>`
, which lists all the existing load
methods.

A load methods follows this principles:

.. code-block:: python

    data = ExampleDatasetLoader()
    # The method can be called from the loader object
    # The method name should start with 'load_'
    # The method returns the requested data
    example_data = data.load_example()

A simulator will typically raise an error if a loading method
does not exist in the DatasetLoader.

Raw data vs Parsed data
~~~~~~~~~~~~~~~~~~~~~~~

Demod does not contain the full raw datasets, but instead uses
parsed datasets which contains only the data useful for the simulation.

The reasons behind this are multiples:

* Reading the data is much faster with parsed data
* The memory required by the parsed data can be smaller than the raw dataset
* A raw datset might be to heavy to be hosted on GitHub
* Some dataset do not allow being made public, but it is in some case possible to publish parsed data

The parsed data is usally in a .npy or .npz, which are numpy arrays,
or in json files if the data is quite small.

Adding your dataset to Demod
----------------------------


Though you could simply create a dataset loader by
implementing different load methods to a python class,
Demod has many helpers available.

You can follow the following instructions.

Make sure you have
:ref:`installed Demod from source <installation_from_source>`


Create a folder in path_of_source/demod/datasets/ with the name you want
your data to be called. You will put all your required files
into this folder.
ExampleData folder in presents you an example of
how you could structure your folder.

Inside your dataset folder,
you can create  .py file where you create your Dataloader.

Make you Dataloader inherit from a parent Dataloader.
Parent Dataloader class provides different load methods with specific inputs and
outputs. When you inherit methods from a parent Dataloader,
you should not implement the :py:obj:`load_` methods, but instead
you should override its corresponding :py:obj:`_parse_` method.

As an example, here we create a dataloader for the appliances.

.. code-block:: python

    from demod.datasets.base_loader import ApplianceLoader
    # inherits from the ApplianceLoader from base_loader that provides
    # a load_appliance_dict method
    class MyApplianceLoader(ApplianceLoader):
        # The name of the folder of this data
        DATASET_NAME = 'ExampleData'
        def _parse_appliance_dict():
            # Here you can define the logic for generating the
            # appliance dictionary from you raw data
            return {
                'name': ['Washing Machine MK3', 'Elec. Hob MK2'],
                'type': ['washingmachine', 'electric_hob'],
            }
            # Note that in a real appliance_dict you might want
            # to add more fields, this is only for example

Now, as you loader inherits form a base loader, the load_ method of the
base loader will create the parsed data and save it using this _parse_
method.

If you want to recreate the parsed data, from the raw_data using
the parse methods, you can set the keyword argument
:py:obj:`clear_parsed_data=True` when instantiating the DatasetLoader.

Sharing your Dataset
~~~~~~~~~~~~~~~~~~~~

We are very happy to see new dataset joining Demod.

At the moment we include only the parsed data
and the python script contaning the dataset loader inside the github
repository, as well as a small readme file where you can
explain what is in your dataset, alternatively you can
also give instructions on how to download and use the raw data.

You will also need to add a documentation to your dataset,
which you can do in the docstring of your Dataloader class.
Please follow
:doc:`Demod documentation guidelines <../contributing/documentation>`.

Once your dataset is ready to be published in Demod,
you can simply create a pull request to the Demod github repository.

If you meet any issue, you are welcomed to contact us at
demod@groupes.epfl.ch .




