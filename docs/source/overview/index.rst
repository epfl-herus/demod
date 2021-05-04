===============
Overview
===============

.. https://docutils.sourceforge.io/docs/user/rst/quickref.html#comments
.. https://docutils.sourceforge.io/docs/user/rst/quickstart.html 

In this section Demod is presented, showing main functionalities 
and available modules.
By reading this section you should have a general understanding of what Demod 
can do, what are the theoretical and empirical foundations of the 
different modules and how to proceed with the design of a domesitc
energy demand model that meets your needs. 


What's Demod supposed to do?
----------------------------

Demod has been developed with the overarching aim of providing
a flexible and easily customizable tool for 
simulating domestic energy demand (i.e., electrical and thermal). 

Indeed, you can select numerous household characteristics 
(e.g., appliances, heating system, socio-demographic characteristics) 
or replace entire simulation modules to generate 
suitable energy demand profiles for your research interests. 

.. Activity-based model

In general, Demod's modules are based on microsimulation of human behavior. 
In other words, they explicitly take into account household occupancy and 
activity profiles to reconstruct their thermal and electrical demand profiles. 
This allows to simulate energy demand profiles in an integrated and
consistent manner, including dependencies between individual loads 
and obtaining adequate time diversification. 


.. German focus

Currently, Demod's modules are parametrized by default using data for Germany. 
It is also possible to select a UK version, which makes use of data 
from the CREST_ model.

The data employed are usually made available 
for other European and non-European countries.
This together with the complete documentation of all the modules
and the data processing methods allows the application of Demod 
to contexts other than Germany. 



Examples of application
-----------------------

Thanks to its properties of modularity, scalability and complete transparency, 
Demod can be used in various applications: 

- **Direct use** for generating occupancy, activity, thermal and electrical 
  demand profiles with high temporal resolution. 
- Integration of Demod-based domestic energy demand model within 
  **co-simulation ecosystems** for the study of larger scale scenarios 
  at district, urban and national level.  
- **Improve, change modules** or **extend** Demod's modules to perform 
  cross-analysis and validation of entire models or individual components.  
  
  
.. figure:: OverviewFigures/ModelFramework.png
  :width: 400
  :alt: Model framework
  :align: center
  :name: model-framework
  
  NAME framework


NAME's main components
-----------------------

This section presents the modules avialble in Demod, their operation, 
their input and output data, and the data needed to parameterize them in case
they are to be used for a specific case study other than those available.
In :numref:`model-framework`, a typical Demod-based domestic energy demand 
model framework is reported. 


.. figure:: OverviewFigures/ModelFramework.png
  :width: 400
  :alt: Model framework
  :align: center
  :name: model-framework
  
  NAME framework
  

.. warning:: reference the name of the variables within the text


.. toctree::
    :maxdepth: 2

    initialization_overview
    occupancy_overview
    electricdemand_overview
    thermaldemand_overview
    mobility_overview
    
    
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/ 