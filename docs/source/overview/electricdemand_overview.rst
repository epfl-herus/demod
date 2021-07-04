================================
Appliance usage and lighting
================================

Appliance usage and lighting modules typically consider household
occupancy/activity profiles and technical characteristics and statistical data
for the different appliances to simulate daily electric load profile
with a 1 min time resolution.

For appliance usage and load simulation, the following modules are available:

- :ref:`overview_occupancy_based_appliance_usage`

while for lighting demand:

- :ref:`overview_fisher_lighting`
- :ref:`overview_CREST_lighting`



Appliance usage
-----------------

The type of appliance influences the method used to estimate
the usage and thus the load profile.
There are three different ways of categorizing appliances.

Based on their load profile:

- *Constant load*: an average load value is assigned to the appliance
  when it is used (e.g., )
- *Time varying load*: appliances that have their power consumption
  varying during an operating cycle.
  For example, the cycle of a washing machine consists of several stages
  of heating, washing, draining, spinning, rinsing,
  which have different power demands.

Based on their usage duration:

- *Fixed duration*: the appliance is always used for the same
  number of timesteps (e.g., kettle and dish washer)
- *Stochastic duration*: the duration is drawn from a random
  distribution. Example: the TV duration in CREST is drawn from
  an exponential distribution, and the hot water durations
  from a set of empirical discrete distributions.

Based on usage patterns:

- *Level usage*: appliances that switch-on and off independently from
  their usage, such as fridge and freezer.
- *Activity dependent*: appliances directly related to an activity.
  Therefore, their usage occurs when at least one resident is undertaking
  the corresponding activity. For example the TV or the oven.

.. note::

  Please keep in mind that these categories depend
  both on the technical characteristics of the device
  and the modeling assumptions and simplifications.
  For example, the dishwasher can be considered either *constant load*
  or *time varing load* depending on the accuracy of the profile used.



.. _overview_occupancy_based_appliance_usage:

Occupancy-based appliance usage simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.appliance_simulators.OccupancyApplianceSimulator`.

:Description: The appliance usage and load modules draws inspiration
  from CREST_ model, developed in excel VBA and presented in
  [Richardson2010]_ and [McKenna2016]_.

  **Initialization.** At the beginning of a run, the model populates
  each dwelling with a set of appliances. This process can be based on
  generic or sociodemographic-specific statistical ownership data.

  **Appliance usage and load.**
  Appliance usage and load profiles are estimated as follows:

  1. first, the activity probability density function (pdf) is multiplied by a calibration scalar,
     whose value is assigned such that the simulated annual consumption
     of a device matches a target value;
  2. second, the turn-on event occurs if the probability exceeds a
     random draw;
  3. when a turn-on event occurs, the duration of the event is
     also estimated;
  4. finally, the turn-off event occurs at the end of the scheduled
     duration or when the active occupancy becomes zero if the device
     depends on the activity.

  .. note::
      - For *level usage* appliances, whicht do not depend on active occupancy
        (e.g.,  fridge and freezer) step 1 assumes activity pdf
        to be equal to one.

      - For appliances with a *fixed duration* step 3 is ignored.

:Availability: This simulator is available for UK and German households.
  However, few modifications are implemented according to data availability.

  **Initialization.** For UK case, the model stochastically estimates
  the appliance set for each household using statistical ownership data
  from multiple sources [Richardson2010]_.

  For the simulation of the German case, demod introduces the following
  changes compared to CREST_:

  - Households are initialized with a set of appliances that is dependent
    on socio-demographic data, using the dataset from [Destatis2017]_.
  - The full set of available appliances is updated to reflect obsolescence
    (e.g., answer machine, cassette / CD player) and changes
    (e.g., tables, game console) in technology.

  **Appliance usage and load.**
  Here is the list of activities that are relevant to the use
  of specific appliances: watching TV; cooking; laundry;
  washing(self) / dressing; ironing; housecleaning; *electronics*;
  *dish washing*. The two latter activities in italics are only
  available for the German case:

  - The *dish washing* activity is present in the German-time-use_
    and corresponds to the use of the dishwasher or the sink.
  - *Electronics* has been added and accounts for the use of computers,
    laptops, tablets, printers and gaming consoles.

:Compatibility: These modules are compatible with occupancy simulators:
  :ref:`overview_4_States` and :ref:`overview_transit_occupancy`.



Lighting
------------

In demod, two modules are available for lighting simulation.
Both depend on two parameters:

- The *number of active occupants*, which counts how many people are
  likely to need light.
- *External radiation* or natural lighting, as human perception of the
  natural light level within a building is a key factor determining use
  of artificial lighting.


.. _overview_fisher_lighting:

Fisher's lighting simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.lighting_simulators.FisherLightingSimulator`.

:Description: This model is presented in [Fisher2015]_, an it computes
  lighting power demand at time *t* as:

  :math:`P_{el,l}(t)=n_{active}(t) \cdot P_{el,l,pp} \cdot
  \frac{I_{g,max}-I_g(t)}{I_{g,max}-I_{g,min}}`

  where :math:`n_{active}(t)` is the number of active occupants
  at a given time, :math:`P_{el,l,pp}` is a constant for accounting
  for light usage per person and :math:`I_{g}(t); I_{g,max}; I_{g,min}`
  are respectively the current irradiation,
  and the boundaries between which the light usage rate increases linearly
  between 0 and 1 as the external irradiation decreases.

:Availability: This simulator is available for UK and German households.

:Compatibility: These modules are compatible with occupancy simulators:
  :ref:`overview_4_States` and :ref:`overview_transit_occupancy`.


.. _overview_CREST_lighting:

CREST lighting simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.lighting_simulators.CrestLightingSimulator`.

:Description:
  This approach is based on the work by Richardson et al. [Richardson2009]_.
  It computes light switch on/off events, considering *irradiation*
  and *effective occupancy*, which takes into account occupants'
  sharing of lights within the same room.
  It also takes into account any lights that are left on during the day
  and the diversity of households.

:Availability: This simulator is available for UK and German households.

  In order to better fit this module to the German case,
  two main modifications are made to the approach of Richardson et al.:

  - The number of lights in a household is initialized following
    the approach of [Frondel2019]_. Here, the number of installed bulbs
    is generated using a normal distribution :math:`N(25.11,15.92)`.
  - Moreover, the type of light bulbs is defined through a discrete
    distribution: LED 65%, CFL 25%, and incandescent 10%.

:Compatibility: These modules are compatible with occupancy simulators:
  :ref:`overview_4_States` and :ref:`overview_transit_occupancy`.

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINKs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _German-time-use: https://www.forschungsdatenzentrum.de/de/haushalte/zve

.. _CREST: https://repository.lboro.ac.uk/articles/dataset/CREST_Demand_Model_v2_0/2001129