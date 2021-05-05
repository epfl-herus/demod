
.. _appliances-support:

===================
Appliances Support
===================

Modelling appliances can become tedious, especially when working with
different incompatible dataset.

We provide some support that aims at solving compatibility issues.
When designing your dataset, you might be interested to read the
following.


Appliances Dictionary
----------------------

When simulating appliances, you can use the appliance dictionary:
:py:attr:`~demod.utils.cards_doc.Params.appliances_dict`
It stores the information of all appliances that you require for the
simulation.
You might want to make sure that the dict you use has all the
properties required by the simulator.

Appliances Types
~~~~~~~~~~~~~~~~

To ensure compatibility between the datasets, appliances all have a property
defined in the dict
:py:obj:`appliances_dict['type']`.
You can check the possible types implemented at:
:py:attr:`~demod.utils.cards_doc.Params.appliance_type`.



Appliances Excell Sheet
-----------------------

A spreadsheet modified from CREST is available for helping you designing
appliances setting for you simulation.
Instructions for creating can be directly found in the
`downloadable excell file
<https://github.com/epfl-herus/demod/blob/main/demod/datasets/ExcellInputFile/inputs.xlsx?raw=true>`_.


Helper functions
----------------

.. automodule:: demod.utils.appliances
    :members:


