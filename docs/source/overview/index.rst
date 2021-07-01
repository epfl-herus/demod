===============
Overview
===============

In this section demod is presented, showing main functionalities
and available modules.
By reading this section you should have a general understanding of what demod
can do, what are the theoretical and empirical foundations of the
different modules and how to proceed with the design of a domestic
energy demand model that meets your needs.


What's demod supposed to do?
----------------------------

Demod has been developed with the overarching aim of providing
a flexible and easily customizable tool for
simulating domestic energy demand (e.g., electrical and thermal).

Indeed, you can select numerous household characteristics
(e.g., appliances, heating system, socio-demographic characteristics)
or replace entire simulation modules to generate
suitable energy demand profiles for your research interests.

Currently, demod's modules are parametrized by default using data for Germany.
It is also possible to select a UK version, which makes use of data
from the CREST_ model.
You can also add a dataset from another country by creating a custom dataset
module to parametrize demod's simulation modules.

The data employed are usually made available
for other European and non-European countries.
This together with the complete documentation of all the modules
and the data processing methods allows the application of demod
to contexts other than Germany.



Socio-technical modeling
--------------------------

As discussed in several studies, domestic energy demand is the outcome
of diverse individual and shared practices in diverse situations.
Therefore, investigating how consumers access energy services and undertaken
daily activities is key for improving domestic energy demand models.

In this regard, demod's modules are based on microsimulation of human behavior.
In other words, they explicitly take into account household occupancy and
activity behavior to reconstruct their thermal and electrical demand profiles.
This allows to simulate thermal and electrical demand in an integrated and
consistent manner, including dependencies between individual loads
and obtaining adequate time diversification.

However, this approach requires certain assumptions and simplifications
to be made, e.g. regarding the association between human behaviour and
appliance use, behavioural heterogeneity or variable energy service expectation.
These assumptions should be considered to make informed choice about the
most suitable modules according to the specific application, but also encourage
the improvement of those currently available.

For further details, you may be interested in the following readings:
[McKenna2017]_


Examples of application
-----------------------

Thanks to its properties of modularity, scalability and complete transparency,
demod can be used in various applications:

- **Direct use** for generating occupancy, activity, thermal and electrical
  demand profiles with high temporal resolution (see :ref:`quickstart-example`).
- Integration of demod-based domestic energy demand model within
  **co-simulation ecosystems** for the study of larger scale scenarios
  at district, urban and national level (see :ref:`co-simulation`).
- **Improve, change modules** or **extend** demod's modules to perform
  cross-analysis and validation of entire models or individual components 
  (see :ref:`create-sim_label`)


demod's main components
-----------------------

This section presents the modules available in demod, their operation,
their input and output data, compatibility with other demod modules,
the current available parameterisations and
the data required to parameterise them in case they are to be used
for a specific case study other than those available.
For more details on their implementation,
you will also find the link of the dedicated API section.

In :numref:`model-framework`, a possible demod-based domestic energy demand
model framework is reported.


.. figure:: OverviewFigures/ModelFramework.png
  :width: 600
  :align: center
  :name: model-framework


  Domestic energy demand model framework using demod: a possible configuration


.. toctree::
    :maxdepth: 2

    Occupany and activity (green) <occupancy_overview>
    Appliance usage and lighting (yellow) <electricdemand_overview>
    Heating demand and supply (red) <thermaldemand_overview>
    mobility_overview
    Wather simulators (blue) <weather_overview>

.. initialization_overview


.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COLORs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |green| image:: OverviewFigures/green.png
.. |red| image:: OverviewFigures/red.png
.. |yellow| image:: OverviewFigures/yellow.png
.. |blue| image:: OverviewFigures/blue.png