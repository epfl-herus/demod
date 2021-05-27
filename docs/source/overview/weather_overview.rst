======================
Weather simulators
======================

In demod, you can use real historical weather data or 
generate synthetic profiles. Here the list of the available modules:

- :ref:`overview_CREST_irradiance_simulator`
- :ref:`overview_CREST_climate_simulator`
- :ref:`overview_RealClimate_simulator`
- :ref:`overview_RealInterpolatedClimate_simulator`

.. _overview_CREST_irradiance_simulator:

CREST irradiance simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.weather_simulators.CrestIrradianceSimulator`.

:Description: This module simulates daily irradiance with 1 min time resolution,
    following the approach developed in CREST_ . 
    On one hand, it estimates the irradiance clear sky,
    taking into account geographical position, solar angle and time of year. 
    On the other hand, the clearness of the sky is estimated using 
    a Markov-chain tecnique, based on historical clearness values.
    The product of the two allows to estimate the 
    effective global radiation on horizontal surface. 

:Availability: As the global clearness TPM is constructed 
    based on global irradiance data, 
    measured at Loughborough University in 2007, this module can be used 
    for the UK context. 
    However, if you have equivalent data for other regions, 
    you can easily extend its application.


.. _overview_CREST_climate_simulator:

CREST climate simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.weather_simulators.CrestClimateSimulator`.

:Description: This module simulates the climate 
    (i.e., irradiance and external air temperature) with 1 min time resolution, 
    following the approach developed in CREST_ . 
    It is based on :py:class:`.CrestIrradianceSimulator`, and an
    `autoregressive–moving-average (ARMA) model <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model>`_
    for the external air temperature.

    In order to maintain the appropriate correlation between radiation 
    and external air temperature, the calculation is done in two steps:

    1. *Average day temperature*. The average daily temperature is generated
       based on the long-term average daily temperature for
       the day of year selected by the user, combined with a stochastic 
       deviation around the average to add an appropriate amount of 
       randomness.
    2. *Stochastic daily temperature profile*. First a daily maximum and minimum
       temperature are assigned based on the expected cumulative irradiance 
       for the day. Then, the temperature profile is simulated by 
       considering the incident irradiance during daylight hours and 
       the clearness index during night hours. 
       As discussed in [McKenna2016]_, this approach guarantees 
       a certain degree of correlation between temperature and irradiance, 
       while keeping the average temperature for the day unchanged.

:Availability: As the model employs historical data 
    (i.e., avarage external air temperature) from the UK,
    it can be used for this context. 
    However, if you have equivalent data for other regions, 
    you can easily extend its application.


.. _overview_RealClimate_simulator:

Real climate simulator
~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.weather_simulators.RealClimate`.

:Description: This module allows you to use historical climate data 
    (i.e., irradiance and temperature) in your simulation. 
    Once the dataset containing the historical climate data has been imported, 
    this module iterates the simulation 
    by selecting the days and hours of interest.  
    
:Availability: In demod you can use data for Germany with a temporal 
    resolution of 1 hour derived 
    from `Renewables ninja <https://www.renewables.ninja/>`_ 
    and available `here <https://data.open-power-system-data.org/weather_data/2020-09-16>`_. 
    This dataset also contains data for other European countries that 
    can therefore be used for new simulations. 




.. _overview_RealInterpolatedClimate_simulator:

Real interpolated climate simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:API: For details about the implementation of
  this simulator you can visit
  :py:class:`~demod.simulators.weather_simulators.RealInterpolatedClimate`.

:Description: Since sometimes the available historical data do not have 
    the desired temporal resolution, 
    it is possible to use this module to obtain more granular profiles.  
    The operation of this module is similar to :ref:`overview_RealClimate_simulator`,
    but in this case the simulation is iterated 
    over the interpolated climate profiles. 
 
:Availability: In demod you can use data for Germany with a temporal 
    resolution of 1 hour derived 
    from `Renewables ninja <https://www.renewables.ninja/>`_ 
    and available `here <https://data.open-power-system-data.org/weather_data/2020-09-16>`_. 
    This dataset also contains data for other European countries that 
    can therefore be used for new simulations. 
