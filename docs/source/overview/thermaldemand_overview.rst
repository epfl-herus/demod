============================================
Heating demand and supply
============================================

Domestic heating demand and supply simulation requires a set of module to
be integrated and jointly executed. 
Below the list of modules available in demod for simulating 
household indoor temperature settings:

- :ref:`overview_CREST_thermostat_setting`
- :ref:`overview_LivingLab_thermostat_setting`

for building thermal behavior:

- :ref:`overview_4R3C_building_thermal_model`

for domestic hot water demand:

- :ref:`overview_CREST_dhw_demand`

for hot water tank thermal behavior:

- :ref:`overview_1R1C_hot_water_tank`

for integrated heat demand for space heating and domestic hot water:

- :ref:`overview_heat_demand`

for heating system control:

- :ref:`overview_thermostats`
- :ref:`overview_system_control`

for heating system operation:

- :ref:`overview_CREST_heating_system`

for integrated heating demand and supply simulation:

- :ref:`overview_FiveModulesHeatingSimulator` 



   
Household indoor temperature settings
-------------------------------------

Two modules are currently available to simulate 
how each household set indoor temperature set point and switch on/off periods
of the heating system. 

.. _overview_CREST_thermostat_setting:

CREST thermostat setting simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details of the implementation of this simulator you can visit
  :py:class:`~demod.simulators.hetaing_simulators.CRESTcontrols`.

:Description: This module implements the approach developed in CREST_.
  First, indoor air temperature set point is stochastically assigned based on 
  empirical discrete distributions.
  Then, timer setting (i.e., on and off periods) are stochastically simulated
  using a first order Markov chain model, which uses empirical data 
  for weekdays and weekends.  
  If the timer is set on, the heating system keep indoor air temperature 
  within the deadband of :math:`+ \: - 2°C`.

:Availability: This module uses empirical data from CREST, which are derived 
  from a UK study. No equivalent data are currently available for Germany. 


.. _overview_LivingLab_thermostat_setting:

Living Lab thermostat setting simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API:  For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.VariableThermostatTemperatureSimulator`.

:Description: This module is inspired by [Sovacool2020]_ and
  attempts to simulate heating system control by defining 
  six different usage patterns. These six different patterns aim 
  to give relevance to the heterogeneous behaviour of different households 
  in terms of heating periods 
  (i.e., regularity and dependence on the presence of active residents) 
  and target temperatures. 
  
  The six profiles can be briefly described as follows: 

  * *Cool Conservers*, often adjust temperature to try and cut bills.
  * *Steady and Savvy*, rarely adjust their heating as they are fine with 18-20°C.
  * *Hot and Cold Fluctuators*, often adjust temperature to get comfortable.
  * *On-Demand Sizzlers*, some like it hotter or want to spend more than others in their home.
  * *On-off Switchers*, turn it on and off to try and make sure home is only warm when someone is in.
  * *Toasty Cruisers*,  love feeling cosy and prefer not to put clothes on if they are cold.

:Availability: This module is inspired by empirical observations 
  of an UK-based research [Sovacool2020]_, but makes use of guessed parameters.

:Compatibility: This module is compatible with all heating system simulators 
  that accept exogenous indoor temperature set point profiles as inputs. 




Building thermal behavior
--------------------------

Demod employs simplified lumped-capacitance models
to simulate building and heating system thermal behaviour.

.. _overview_4R3C_building_thermal_model:

Low-order building thermal model (4R3C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.BuildingThermalDynamics`.

:Description:
  This module simulates the thermal behavior of the building using an 
  equivalent low-order electric circuit as in CREST_
  (see :numref:`4R3C-building-thermal-model`).
  Six building typology are available: detached house,
  semi-detached house and apartment both in the renovated version and not. 
  
  The name 4R3C refers to three thermal capacitances representing
  the thermal masses of the building, indoor air, and heat emitters and
  the four thermal resistences account for heat transfer between 
  (i) walls and indoor air, (ii) walls and outdoor air, 
  (iii) emitters and indoor air, 
  and (iv) air ventilation between indoor and outdoor.

  Here are the equivalent equations:

  :math:`T_{ia}^{t+1}=T_{ia}^t + \frac{dt}{C_{ia}}[u_{ia,em}(T_{em}^t-T_{ia}^t)-u_{ia,b}(T_{ia}^t-T_{b}^t)-u_{v}(T_{ia}^t-T_{oa}^t)+g^t]`

  :math:`T_{b}^{t+1}=T_{b}^t + \frac{dt}{C_{b}}[u_{ia,b}(T_{ia}^t-T_{b}^t)-u_{oa,b}(T_{b}^t-T_{oa}^t)]`

  :math:`T_{em}^{t+1}=T_{em}^t + \frac{dt}{C_{em}}[Q^t-u_{ia,em}(T_{em}^t-T_{ia}^t)]`

  The emitters currently available in demod are a radiator system. 
  More details on their sizing and characteristics can be found 
  in [McKenna2016]_.   

:Availability: The parameters for the capacitance and resistences are 
  taken from CREST_, and they refer to the UK building stock.
  An updated parameters for the German case will be released 
  in future versions. 



.. figure:: OverviewFigures/4R3Cbuildingthermalmodel.png
    :width: 700
    :alt: 4R3C low-order building thermal model
    :align: center
    :name: 4R3C-building-thermal-model

    4R3C low-order building thermal model  
        
.. 6R2C building thermal model
    
.. Alternatively can be selected the model 6R2C, that starting from 
.. the model 5R1C of EN ISO 13790 integrates the resistance (1R) 
.. and the capacity (1C) of the radiator system (see :numref:`ISO13790-thermal-model`).  
    
.. For this model, parameters are currently available for a single apartment
.. configuration, as presented in [Vivian2017]_.




Domestic hot water demand
-------------------------

Currently in demod there is a module for simulating the demand for 
domestic hot water. 

.. _overview_CREST_dhw_demand:

CREST domestic hot water demand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: Simulation of domestic hot water demand is part of
  :ref:`overview_occupancy_based_appliance_usage` and for details 
  about the implementation you can visit
  :py:class:`~demod.simulators.appliance_simulators.SubgroupApplianceSimulator`.

:Description: Currently demod simulates domestic hot water demand 
  following the approach of CREST_, which simulates the use of water fixtures
  in the same way as household appliances: 

  1. first the number of water fixtures in the house is initialized; 
  2. then, the pdf of the activities *washing* or *cooking*
     is multiplied by a calibration scalar, 
     whose value is assigned such that the simulated annual water consumption 
     of each fixture matches a target value;
  3. the water withdrawal event occurs if the probability exceeds a 
     random draw; 
  4. finally, when a water withdrawal event occurs, the  temperature  of  
     hot  water  and  withdrawn  volume  are determined stochastically.


:Availability: This module uses empirical data from CREST, which are derived 
  from a UK study. No equivalent data are currently available for Germany.



Hot water tank thermal behavior
--------------------------------

Intro

.. _overview_1R1C_hot_water_tank:

Low-order hot water tank thermal model (1R1C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.BuildingThermalDynamics`.

:Description:
  This module simulates the thermal behavior of the hot water tank using an 
  equivalent low-order electric circuit as in CREST_
  (see :numref:`1R1C-hot-water-tank-thermal-model`).
  
  The name 1R1C refers to thermal capacitance representing
  the thermal masses of hot water and 
  the thermal resistences of the hot water tank insulation between 
  hot water and indoor air. 

  Here is the equivalent equation:

  :math:`T_{dhw}^{t+1}=T_{dhw}^t + \frac{dt}{C_{tank}}[Q_{dhw}-m_{dhw}^{t}(T_{dhw}^t-T_{dhw}^{in})-u_{tank}(T_{dhw}^t-T_{ia}^t)]` 

:Availability: The parameters for the capacitance and resistences are 
  taken from CREST_.

.. figure:: OverviewFigures/1R1Chotwatertankthermalmodel.png
    :width: 700
    :alt: 1R1C low-order hot water tank thermal model
    :align: center
    :name: 1R1C-hot-water-tank-thermal-model

    1R1C low-order hot water tank thermal model 



Heat demand
-------------------------

Intro 

.. _overview_heat_demand:

Heat demand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.HeatDemand`.

:Description: It computes the heat demand for both:
  domestic hot water and space heating.
  
  There exist different algorithm for computing the heat demand.

:Compatibility: This module is flexible and allows to use alternative 
  comfort temperature and heating switch on profiles. 
  Once these profiles are generated or empirically measured, 
  they can be given as imput to the thermal building model 
  to estimate the heating demand. 




Heating system control
------------------------

Intro

.. _overview_thermostats:

Thermostats
~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.Thermostats`.

:Description: Simulates the state of different thermostats (can be ON or OFF
  = True or False).
  Thermostat control the temperature of different component.
  They are switched to on once the temperature of a component is
  below its target_temperature minus a dead_band.

:Availability:

:Compatibility:

.. _overview_system_control:

Heating system control
~~~~~~~~~~~~~~~

:API:  For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.SystemControls`.

:Description: It checks which controls should be sent to the
  :py:class:`.HeatingSystem`, based on the heat demand and
  on the thermostats.

  It can handle combi boilers, which means that the boiler does not
  stay on to keep the cylinder at a high temperature.

  The heating control model simulates an integrated system 
  with a timer and thermostat (see :numref:`heating-control`). 
  It allows to manage in an integrated way the supply of heating 
  for domestic hot water and space heating, prioritizing the first 
  and ensuring that the heating system works 
  within the recommended operating conditions. 


  This unit takes the indoor temperature of the building as input and 
  compares it to thermostat setting 
  to estimate the space heating thermal demand. Moreover, thanks to 
  the temperature monitoring of the emitters, 
  the controller avoids that they can reach temperatures higher than 
  the safety temperature of 55 °C.

:Availability:

:Compatibility:


    
Heating systems
------------------------

Currently demod implements a set of heating systems, following the 
approach developed in [McKenna2016]_.

.. _overview_CREST_heating_system:

CREST hetaing system
~~~~~~~~~~~~~~~~~~~~~
   
:API:  For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.HeatingSystem`.

:Description: It simulates the energy consumption (i.e., gas or electricity)
  of the heating system for providing requested heat demand.

:Availability:

:Compatibility:    



Integrated heating demand and supply 
-------------------------------------

Intro

.. _overview_FiveModulesHeatingSimulator:

Five modules heating simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API:  For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.heating_simulators.FiveModulesHeatingSimulator`.

:Description: It simulates the energy consumption of an household required for
  heating.
  The five components simulated:

  * The heating system (boiler, heat pump, ...)
  * The controls of the heating system.
  * The heat demand of the house.
  * The thermostats of different components
  * The temperatures of the building components.

  The implementation is based on CREST model, with a simplification of
  the thermostats and controls.

  This simulator is also compatible with external simulated components.

  * External thermostat
      the desired indoor temperature can be passed in the
      step method through
      :py:attr:`~demod.utils.cards_doc.Inputs.external_target_temperature`

:Availability:

:Compatibility:

(insert a figure)



.. note::
  







 
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/ 