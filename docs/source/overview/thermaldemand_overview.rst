============================================
Thermal demand, control and heating system 
============================================

Demod structures the thermal demand and heating system model into multiple 
components: building thermal model, space heating control, 
consumer thermostat settings, domestic hot water demand and hot water tank.
   
Consumer thermostat settings
----------------------------

Currently there are available two modueles for simulating 
consumer thermostat settings.


CREST stochastic setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module implements the approach developed in CREST_.
In this case, timer setting are stochastically simulated based on empirical 
distributions for weekdays and weekends.  
(speak about temperature anm switch on periods)


Living Lab thermostat setting simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
The second method is inspired by the work [Sovacool2020]_ and
attempts to simulate the operation of the heating system by defining 
six different usage patterns. These six different patterns aim to give 
more relevance to the heterogeneity of behavior of different users in 
terms of heating periods and target temperatures. 

For further details about the implementation of this simulator, you can 
read about 
:py:class:`~demod.simulators.demod.simulators.heating_simulators.VariableThermostatTemperatureSimulator`.


.. note::
    Demod is flexible and allows to use alternative comfort temperature and 
    heating switch on profiles. Once these profiles are generated or 
    empirically measured, they can be given as imput to the 
    thermal building model to estimate the heating demand. 

Building thermal model
------------------------

Demod employs simplified lumped-capacitance models
to simulate building and heating system thermal behaviour.

4R3C building thermal model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model currently available in demod is a 4R3C model 
(see :numref:`CREST-thermal-model`) and it consider the same 
parameters used in CREST_ for the UK building stock. 
Here the six configuration of building are reported: detached house,
semi-detached house and apartment both in the renovated version and not. 
 
The 4R3C models implements the three thermal capacitances representing
the thermal masses of the building, indoor air, and heat emitters. 
While, the four thermal resistences account for heat transfer between 
(i) walls and indoor air, (ii) walls and outdoor air, 
(iii) emitters and indoor air, 
and (iv) air ventilation between indoor and outdoor.

The emitters currently available in demod are a radiator system. 
More details on their sizing and characteristics can be found 
in [McKenna2016]_.   

The parameters for the capacitance and resistences are taken from CREST_, 
and updated parameters for the German case will be released 
in future versions. 
    

    
.. note::
   A more appropriate model for the German building stock will be 
   provided in future releases.  
        
 .. 6R2C building thermal model
    
.. Alternatively can be selected the model 6R2C, that starting from 
.. the model 5R1C of EN ISO 13790 integrates the resistance (1R) 
.. and the capacity (1C) of the radiator system (see :numref:`ISO13790-thermal-model`).  
    
.. For this model, parameters are currently available for a single apartment
.. configuration, as presented in [Vivian2017]_.

Domestic hot water demand
------------------------

Currently demod simulates domestic hot water demand 
following the approach developed in [McKenna2016]_.


Heating controls
------------------------

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

    
Heating systems
------------------------

Currently demod implements a set of heating systems, following the 
approach developed in [McKenna2016]_.
    
    




 
 .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://www.lboro.ac.uk/research/crest/demand-model/ 