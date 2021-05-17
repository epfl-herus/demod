==========
Simulators
==========

The following simulators are ready to use for simulations .



Identity Cards
-------------------

An ID cards system simplifies the usage of simulators.

Each simulator is given different types of parameters:

* **Params**: the parameters of input for the :py:meth:`__init__`.
* **Data**: the loading methods required from the Dataloader given in the :py:attr:`~demod.utils.cards_doc.Params.data` param.
* **Step input**: the arguments of the :py:meth:`step` method of the simulator.
* **Output**: the get methods available.
* **Step size**: the time between to steps.


With these identity cards, the usage of the simulators and the datasets
is made simpler.
The possible values for each parameters are available
at :ref:`Cards Documentation <cards-documentation>`.



Activity simulators
-------------------

Activity simulators generate patterns of the residents occupations during the
days.


.. module:: demod.simulators.crest_simulators

.. autoclass:: Crest4StatesModel


.. module:: demod.simulators.sparse_simulators
    :noindex:

.. autoclass:: SparseTransitStatesSimulator

.. autoclass:: SubgroupsActivitySimulator






Weather simulators
---------------------

It is possible to either use real weather data or to simulate the weather.

.. module:: demod.simulators.weather_simulators
    :noindex:

.. autoclass:: CrestIrradianceSimulator


.. autoclass:: CrestClimateSimulator


.. autoclass:: RealClimate

.. autoclass:: RealInterpolatedClimate

Ligthing simulators
---------------------

.. module:: demod.simulators.lighting_simulators

.. autoclass:: FisherLighitingSimulator

.. autoclass:: CrestLightingSimulator

.. automethod:: CrestLightingSimulator.sample_bulbs_configuration


Appliances simulators
---------------------

Appliances simulator simulate the electricity consumption of different house
appliances.

.. module:: demod.simulators.appliance_simulators
    :noindex:


.. autoclass:: SubgroupApplianceSimulator



Heating  simulators
--------------------

.. module:: demod.simulators.heating_simulators
    :noindex:

.. autoclass:: FiveModulesHeatingSimulator




Heating Components
~~~~~~~~~~~~~~~~~~~

Heating  simulators can be splitted into different heating components.


.. autoclass:: BuildingThermalDynamics
.. autoclass:: Thermostats
.. autoclass:: HeatDemand
.. autoclass:: SystemControls
.. autoclass:: HeatingSystem



Maybe add external cylinder in the future ???

Variable Thermostats
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VariableThermostatTemperatureSimulator

CREST heating
~~~~~~~~~~~~~


.. autoclass:: CrestControls



Load simulator
------------------
This simulator combines all previous simulators in a single simulator.
This is very convenient for simple simulations.


.. module:: demod.simulators.load_simulators
.. autoclass:: LoadSimulator
    :members:



