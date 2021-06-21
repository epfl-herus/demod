"""Some simulators for the climate."""


import datetime
from demod.datasets.base_loader import ClimateLoader
from typing import Dict, List, Union
from demod.datasets.OpenPowerSystems.loader import OpenPowerSystemClimate
import math
from time import time
from demod.utils.distribution_functions import check_valid_cdf, rescale_cdf
from demod.datasets.CREST.loader import Crest
from ..utils.data_types import DataInput
from ..utils.monte_carlo import monte_carlo_from_1d_cdf
from .base_simulators import (
    Callbacks,
    InitilizationTime,
    SimLogger,
    Simulator,
    TimeAwareSimulator,
)

import numpy as np


class ClimateSimulator(TimeAwareSimulator):
    """Absract class for climate simulators.
    """

    def __init__(self, **kwargs) -> None:
        """Create a climate simulator.

        It simulates the irradiance from the sun and the oustide temperature.
        """
        super().__init__(n_households=1, **kwargs)

    def get_outside_temperature(self) -> float:
        """Return the temperature of outside."""
        raise NotImplementedError()

    def get_irradiance(self) -> float:
        """Return the temperature of outside."""
        raise NotImplementedError()


class CrestIrradianceSimulator(ClimateSimulator):
    """Irradiance simulator from CREST.

    Simulates the irradiance, based on a markov
    transition probabilies model for the irradiance during the day.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.initial_clearness`
    Data
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_clearness_tpms`
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_geographic_data`
    Step input
        None
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_irradiance`
    Step size
        1 Minute

    """

    data: DataInput

    def __init__(
        self,
        data: DataInput = "CREST",
        initial_clearness: float = 0.99,
        start_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0),
        **kwargs
    ) -> None:
        """Initialize the Crest climate simulator.

        Args:
            data: the data to be used
            initial_clearness: the clearness at the start of the simulation
                It will not be used during the initialization
            start_datetime: the start of the simuulation
        """
        if data == "CREST":
            data = Crest()

        geo_dict = data.load_geographic_data()

        self.longitude = geo_dict["longitude"]
        self.latitude = geo_dict["latitude"]
        self.meridian = geo_dict["meridian"]

        # summer time is the french version of daylight saving time, sry
        self._use_summer_time = geo_dict["use_daylight_saving_time"]

        clearness_tpm, clearness_values, step_size = data.load_clearness_tpms()
        self.clearness_cdf = np.cumsum(clearness_tpm, axis=1)
        self.clearness_values = clearness_values

        check_valid_cdf(self.clearness_cdf)

        if 'step_size' in kwargs:

            if step_size != kwargs['step_size']:
                raise ValueError(
                    "'step_size' = {} was specified in {}'"
                    ". It uses the step_size = {} from {} "
                    " which are not the same.".format(
                        kwargs['step_size'],
                        self,
                        step_size,
                        data.load_clearness_tpms
                    ))

            kwargs = kwargs.copy()
            kwargs.pop('step_size')

        super().__init__(
            step_size=step_size,
            start_datetime=start_datetime,
            **kwargs
        )

        self.initialize_starting_state(initial_clearness)

    def initialize_starting_state(self, initial_clearness) -> None:
        """Initialize the starting state of the clearness.

        The clearness that it given will the one at midnight on the
        start day, so not at the exact start of the simulation

        Args:
            initial_clearness: clearness at midnight the day of
                start_datetime
        """
        # Finds where the initial clearness is in the values
        indices = np.where(self.clearness_values == initial_clearness)[0]
        self._clearness_state = int(
            # Happens when 'clear' is the same as 1.
            indices if len(indices) == 1 else indices[0]
        )

        self.on_after_next_day()

        super().initialize_starting_state(
            initialization_time=datetime.time(0, 0, 0)
        )
        # Assuume midnight sun is down
        self.irradiance = 0

    def _next_clearness(self):
        # update the state through a Markov Chain Monte Carlo algorithm
        self._clearness_state = monte_carlo_from_1d_cdf(
            self.clearness_cdf[self._clearness_state]
        )
        return self.clearness_values[
            self._clearness_state
        ]  # return the value of the clearness

    def get_day_of_year(self):
        return self.current_time.timetuple().tm_yday

    def get_current_clearness(self):
        return self.clearness_values[self._clearness_state]

    def is_summer_time(self):
        day_of_year = self.get_day_of_year()
        # Day of the year that summer time starts = 87
        # Day of the year that summer time ends = 304
        return day_of_year >= 87 and day_of_year < 304

    def _calculate_global_irradiance(self):

        dblDayOfYear = self.get_day_of_year()

        # Calculate B
        dblB = 360 * (dblDayOfYear - 81) / 364

        # Calculate equation of time
        dblEquationOfTime = (
            (9.87 * math.sin(2 * dblB * math.pi / 180))
            - (7.53 * math.cos(dblB * math.pi / 180))
            - (1.5 * math.sin(dblB * math.pi / 180))
        )

        # Calculate time correction factor
        self.dblTimeCorrectionFactor = (
            4 * (self.longitude - self.meridian)
        ) + dblEquationOfTime

    def get_irradiance(self) -> float:
        """Get the current simulated irradiance."""
        return self.irradiance

    def on_after_next_day(self) -> None:
        """Callback to update the irradiance for the next day.

        It is based on the earth position and angle
        around the sun.
        """
        # This must be calculated as the earth rotates
        self._calculate_global_irradiance()

    @Callbacks.after_next_day
    def step(self):

        dblDayOfYear = self.get_day_of_year()
        local_hour = self.current_time.hour
        # Update local standard time hour but only if Daylight saving applies
        # in that country
        if self._use_summer_time and self.is_summer_time():
            local_hour -= 1

        # Calculate hours before solar noon
        dblHoursBeforeSolarNoon = 12.0 - (
            local_hour
            + (self.current_time.minute / 60.0)
            + (self.dblTimeCorrectionFactor / 60.0)
        )

        # Calculate extraterrestrial radiation
        dblG_et = 1367.0 * (
            1.0 + (0.034 * math.cos(2.0 * math.pi * dblDayOfYear / 365.25))
        )

        # Calculate optical depth
        dblOpticalDepth = 0.174 + (
            0.035 * math.sin(2.0 * math.pi * (dblDayOfYear - 100) / 365)
        )

        # Calculate hour angle
        dblHourAngle = 15.0 * dblHoursBeforeSolarNoon

        # Calculate declination
        dblDeclination = 23.45 * math.sin(
            2.0 * math.pi * (284 + dblDayOfYear) / 365.25
        )

        # Calculate solar altitude angle
        dblSolarAltitudeAngle = (
            180.0
            / math.pi
            * math.asin(
                (
                    math.cos(self.latitude * math.pi / 180.0)
                    * math.cos(dblDeclination * math.pi / 180.0)
                    * math.cos(dblHourAngle * math.pi / 180.0)
                )
                + (
                    math.sin(self.latitude * math.pi / 180.0)
                    * math.sin(dblDeclination * math.pi / 180.0)
                )
            )
        )

        # Calculate clearsky beam radiation at surface (plane tracking the sun)
        if dblSolarAltitudeAngle > 0:
            self._irradiance_clearsky = dblG_et * math.exp(
                (0 - dblOpticalDepth)
                / math.sin(dblSolarAltitudeAngle * math.pi / 180)
            )
        else:
            self._irradiance_clearsky = 0

        # Get clearness index for this time step
        self._clearness = self._next_clearness()

        # Calculate global radiation on surface horizontal
        self.irradiance = (
            self._irradiance_clearsky
            * self._clearness
            * math.sin(dblSolarAltitudeAngle * math.pi / 180)
        )

        super().step()


class CrestClimateSimulator(ClimateSimulator):
    """Climate simulator from CREST.

    Simulates the climate, based on
    :py:class:`.CrestIrradianceSimulator`, and an
    `ARMA model <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model>`_
    for the temperature.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.initial_clearness`
    Data
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_clearness_tpms`
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_geographic_data`
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_temperatures_arma`
    Step input
        None
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_irradiance`
        :py:meth:`~demod.utils.cards_doc.Sim.get_outside_temperature`
    Step size
        1 Minute

    """

    def __init__(
        self,
        data: DataInput = "CREST",
        start_datetime: datetime.datetime = datetime.datetime(
            2014, 1, 1, 0, 0, 0
        ),
        initial_clearness: float = 0.99,
        **kwargs
    ) -> None:
        """Initialize a climate simulator.

        Args:
            data: Data to be used. Defaults to "CREST".
            start_datetime: The start of the simulaiton.
                Defaults to datetime.datetime( 2014, 1, 1, 0, 0, 0 ).
            initial_clearness: Clearness at the start of the simulation.
                Can be from 0 to 1. Defaults to 0.99.
        """


        super().__init__(start_datetime=start_datetime, **kwargs)
        # remove some kwargs for irradiance
        kwargs.pop("logger", None)
        self.irradiance_sim = CrestIrradianceSimulator(
            data,
            start_datetime=start_datetime,
            initial_clearness=initial_clearness,
            **kwargs)

        if data == "CREST":
            data = Crest()

        # Set the cloud cooling rate
        self.cloud_cooling_rate = 0.1 / 60.0

        arma_dic = data.load_temperatures_arma()
        self.arma_dic = arma_dic

        self.initialize_starting_state()

    def initialize_starting_state(self, *args, **kwargs) -> None:

        # Set values of the 1st parameters for AR, MA ,ARMA
        self.r = np.random.randn() * self.arma_dic["SD"]
        self.ar = 0.0
        self.ma = 0.0
        self.arma = 0.0

        self.on_before_next_day()

        self.irradiance = next(self.irradiance_iter)
        self.temperature = next(self.temperature_iter)

        super().initialize_starting_state(
            *args, initialization_time=datetime.time(0, 0, 0), **kwargs
        )

    def on_before_next_day(self) -> None:
        """Update the simulator for the next day.

        The temperature can only be calculated once we know the irradiance
        of the day.
        Therefore, this first computes the irradiance and then the temperature.
        Then it stores the two as iterators to be called by the step function.
        """

        # updates some parameters
        self._update_ARMA()
        self._compute_extraterrestrial_irradiance()

        # simulate the irradiance
        daily_irradiance = []
        daily_clearness = []
        for _ in range(24 * 60):
            daily_irradiance.append(self.irradiance_sim.get_irradiance())
            daily_clearness.append(self.irradiance_sim.get_current_clearness())
            self.irradiance_sim.step()

        # compute the daily temprature based on irradiance
        daily_temperature = self._step_day(
            np.array(daily_irradiance).reshape(-1),
            np.array(daily_clearness).reshape(-1),
        )

        # creates iterators to step access the temp and irradiance
        self.irradiance_iter = iter(daily_irradiance)
        self.temperature_iter = iter(daily_temperature)


    def get_irradiance(self) -> float:
        return self.irradiance

    def get_outside_temperature(self) -> float:
        return self.temperature

    @Callbacks.before_next_day
    def step(self) -> None:
        self.irradiance = next(self.irradiance_iter)
        self.temperature = next(self.temperature_iter)
        return super().step()

    def _update_ARMA(self):
        # sample a random number from a normal
        new_r = np.random.randn() * self.arma_dic["SD"]
        # ... AR part AR(t)= AR*AR(t-1)+E(t)
        new_ar = self.ar * self.arma_dic["AR"] + new_r
        # ... MA part: MA= MA*E(t-1)+E(t)
        new_ma = self.r * self.arma_dic["MA"] + new_r
        # ... ARMA part: ARMA= AR*AR(t-1)+MA*E(t-1)+E(t)
        new_arma = (
            self.ar * self.arma_dic["AR"]
            + self.r * self.arma_dic["MA"]
            + new_r
        )

        self.r = new_r
        self.ar = new_ar
        self.ma = new_ma
        self.arma = new_arma

        # Calculate Td_model_Ar and Td_model_ARMA
        td = self.arma_dic["T_mean"] + self.arma_dic["T_std"] * math.sin(
            360.0 * self.get_day_of_year() / 365.0 * math.pi / 180.0
            + self.arma_dic["T_shift"] / 365.0 * 2.0 * math.pi
        )
        self.Td_ar = td + self.ar
        self.Td_arma = td + self.arma

    def _compute_extraterrestrial_irradiance(self):
        # determine the extraterrestrial irradiance
        self.cSolarConstant = 1367 * (
            1
            + (0.034 * math.cos(2 * math.pi * self.get_day_of_year() / 365.25))
        )

    def get_day_of_year(self):
        return self.current_time.timetuple().tm_yday

    def _step_day(self, irradiances, clearness_indexes) -> np.ndarray:

        # calculate the cumulative global horizontal irradiance
        cumulative_radiation_2 = np.cumsum(irradiances)

        cumulative_radiation_1 = np.zeros_like(cumulative_radiation_2)
        cumulative_radiation_3 = np.zeros_like(cumulative_radiation_2)
        cumulative_radiation_4 = np.zeros_like(cumulative_radiation_2)
        # if it is daylight, calculate the cumulative extraterrestrial
        # irradiance and calculate a ratio of irradiances
        mask_daylight = irradiances > 0

        # Determine number of minutes outside of daylight
        # Determine the temperature difference between sunset and the
        # minimum temperature for the day

        cumulative_radiation_1[mask_daylight] = self.cSolarConstant
        cumulative_radiation_1 = np.cumsum(cumulative_radiation_1)
        cumulative_radiation_1[~mask_daylight] = 0

        cumulative_radiation_3[mask_daylight] = (
            cumulative_radiation_2[mask_daylight]
            / cumulative_radiation_1[mask_daylight]
        )
        cumulative_radiation_3[~mask_daylight] = np.nan

        # Find the maximum of the ratio and store the minute when this occurs
        kx_max = np.nanmax(cumulative_radiation_3)
        kx_max_i = np.nanargmax(cumulative_radiation_3)

        # store the cumulative irradiance value
        total_irradiation = cumulative_radiation_2[-1]
        # Convert units of irradiance
        total_irradiation = (total_irradiation / 1000.0) * (60.0 / 3600.0)

        # Set a temperature standard deviation
        dTd_SD = 1

        # Given the cumulative irradiance, determine the total deviation of
        # temperatures around the daily average
        dTd = 20 * math.log(total_irradiation + 2.5, 10) - 7

        dTd = dTd + np.random.randn() * dTd_SD

        # Set the max and min temperature for the day
        Td_min = self.Td_arma - 0.5 * dTd
        Td_max = self.Td_arma + 0.5 * dTd

        # Set the slopes of the temperature profiles before and after the
        # maximum temperature
        slope_before = (Td_max - Td_min) / kx_max
        slope_after = slope_before * 1.7

        # For each minute up to the maximum temperature
        # If it's daylight, set the external air temperature
        mask_daylight = irradiances[: kx_max_i + 1] > 0
        cumulative_radiation_4[: kx_max_i + 1] = np.where(
            mask_daylight,
            Td_min + slope_before * cumulative_radiation_3[: kx_max_i + 1],
            cumulative_radiation_4[: kx_max_i + 1],
        )

        # For each minute after the maximum temperature
        # If it's daylight, set the external air temperature
        mask_daylight = irradiances[(kx_max_i + 1):] > 0
        cumulative_radiation_4[(kx_max_i + 1):] = np.where(
            mask_daylight,
            Td_max
            - (
                slope_after
                * (kx_max - cumulative_radiation_3[(kx_max_i + 1):])
            ),
            cumulative_radiation_4[(kx_max_i + 1):],
        )

        # Determine number of minutes outside of daylight
        di = np.sum(irradiances == 0)
        # Determine the temperature difference between sunset and the
        # minimum temperature for the day
        indices = np.where(irradiances > 0)[0]
        sunset_i = indices[-1]
        sunrise_i = indices[0]
        dT = (
            cumulative_radiation_4[sunset_i]
            - cumulative_radiation_4[sunrise_i]
        )  # = Td_sunset - Td_min

        # Determine how much cooling is therefore required
        CoolingRate = dT / di
        CloudCoolingRate = 0.025

        # Calculate the overnight mean clearness index
        vOvernightMeanClearness = np.mean(clearness_indexes[irradiances == 0])

        # Calculate the overnight temperatures
        # after day
        cumulative_radiation_4[sunset_i:] = cumulative_radiation_4[
            sunset_i - 1
        ]
        cumulative_radiation_4[sunset_i:] -= np.cumsum(
            CoolingRate
            - CloudCoolingRate
            * (vOvernightMeanClearness - clearness_indexes[sunset_i:])
        )

        # before day
        cumulative_radiation_4[:sunrise_i] = cumulative_radiation_4[-1]
        cumulative_radiation_4[:sunrise_i] -= np.cumsum(
            CoolingRate
            - CloudCoolingRate
            * (vOvernightMeanClearness - clearness_indexes[:sunrise_i])
        )

        # output temperature data
        return cumulative_radiation_4


class RealClimate(ClimateSimulator):
    """Simulator that outputs real climate data.

    Iterates over the historical climate data.

    Note:
        In a future update this could create getters and handle
        all the data names loaded by
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`
    Step input
        None
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_irradiance`
        :py:meth:`~demod.utils.cards_doc.Sim.get_outside_temperature`
    Step size
        :py:attr:`~demod.utils.cards_doc.Params.data`.
        :py:attr:`~demod.datasets.base_loader.ClimateLoader.step_size`

    """

    def __init__(
        self, data: ClimateLoader = 'Germany',
        start_datetime: datetime.datetime = datetime.datetime(
            1980, 1, 1, 0, 0, 0
        ),
        **kwargs
    ) -> None:
        """Create a real climate simulator.

        Args:
            data: Datset to be used. Defaults to 'Germany'.
            start_datetime: The start of the simulation.
                Defaults to datetime(1980, 1, 1, 0, 0, 0).
        """
        if data == 'Germany':
            data = OpenPowerSystemClimate('germany')

        # should get the tzinfromation as well
        climate_data = data.load_historical_climate_data(start_datetime)


        if 'step_size' in kwargs:
            raise ValueError((
                "'step_size' cannot be specified in {}"
                ". It uses the step size from {}."
                "You can use {} if you want to use a specific step_size."
            ).format(
                self,
                data.load_historical_climate_data,
                RealInterpolatedClimate
            ))

        super().__init__(
            start_datetime=start_datetime,
            step_size=data.step_size, **kwargs
            )

        self.irradiance_iter = iter(
            climate_data.get('irradiance')
            if climate_data.get('radiation_global') is None
            else climate_data.get('radiation_global')
        )

        self.temperature_iter = iter(climate_data['outside_temperature'])

        intialization_time = climate_data['datetime'][0].astype(
            datetime.datetime
        )
        intialization_time = intialization_time.replace(
            tzinfo=start_datetime.tzinfo
        )
        # Checks that the data does not start after the start datetime
        if intialization_time > start_datetime:
            raise ValueError(
                "Requested start_datetime is : {}, but dataset {} for"
                " country '{}', starts "
                "only at {}".format(
                    start_datetime,
                    data,
                    data.country,
                    intialization_time
                )
            )

        self.initialize_starting_state(intialization_time)

    def initialize_starting_state(
        self,  initialization_time: InitilizationTime
    ) -> None:
        """Initialize the weather with the first datapoints.

        Args:
            initialization_time: The first datetime, from the datapoints.
        """
        self.irradiance = next(self.irradiance_iter)
        self.temperature = next(self.temperature_iter)
        super().initialize_starting_state(
            initialization_time=initialization_time
        )

    def step(self) -> None:
        """Update the new datapoints for irradiance and temperature.
        """
        self.irradiance = next(self.irradiance_iter)
        self.temperature = next(self.temperature_iter)
        super().step()

    def get_irradiance(self) -> float:
        return self.irradiance

    def get_outside_temperature(self) -> float:
        return self.temperature


class RealInterpolatedClimate(ClimateSimulator):
    """Simulator that interpolated values from real climate data.

    `scipy <https://www.scipy.org/>`_ will be required to
    use this simulator.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.step_size`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.interpolation_kind`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.ClimateLoader.load_historical_climate_data`
    Step input
        None
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_irradiance`
        :py:meth:`~demod.utils.cards_doc.Sim.get_outside_temperature`
    Step size
        Any. Set in Params.

    """
    interpolators: Dict[str, object]
    def __init__(
        self,
        data: DataInput = 'Germany',
        start_datetime: datetime.datetime = (
            datetime.datetime(1980, 1, 1, 0, 0, 0)
        ),
        interpolation_kind: Union[str, int] = 'linear',
        **kwargs
    ) -> None:
        try:
            from scipy.interpolate import interp1d
        except ImportError as imp_err:
            raise RuntimeError(
                'Could not import from scipy.'
                'You can install it using "pip install scipy".') from imp_err

        if data == 'Germany':
            data = OpenPowerSystemClimate('germany')

        self.data = data

        climate_dict = data.load_historical_climate_data(start_datetime)

        super().__init__(start_datetime=start_datetime, **kwargs)

        self.datetime_values_utc = np.array(
            climate_dict.pop('datetime'), dtype='datetime64[m]')
        self.initial_time = self.datetime_values_utc[0]
        # Check where is the end of the data
        self.last_time = self.datetime_values_utc[-1].astype(datetime.datetime)
        self.last_time = self.last_time.replace(
            tzinfo=None if start_datetime.tzinfo is None
            else datetime.timezone.utc
        )
        # must use real values for interp1d
        minutes = (self.datetime_values_utc - self.initial_time).astype(float)

        self.interpolators = {
            key: interp1d(
                minutes, values,
                kind=interpolation_kind, assume_sorted=True
            )
            for key, values in climate_dict.items()
        }
        # creats getters correponding to the received data
        for key in climate_dict.keys():
            setattr(
                self,
                'get_' + key,
                self._make_interpolated_getter(key)
            )

        intialization_time = self.datetime_values_utc[0].astype(
            datetime.datetime
        )
        intialization_time = intialization_time.replace(
            tzinfo=start_datetime.tzinfo
        )

        # Checks that the data does not start after the start datetime
        if intialization_time > start_datetime:
            raise ValueError(
                "Requested start_datetime in utc is : {},"
                " but dataset {} for"
                " country '{}', starts "
                "only at {}".format(
                    start_datetime,
                    data,
                    data.country,
                    intialization_time
                )
            )

        self.initialize_starting_state(intialization_time=intialization_time)

    def step(self) -> None:
        super().step()
        # check we did not reach the end of the data
        if self.current_time > self.last_time:
            raise ValueError(
                "Time {} is the end of the recorded data for {}.".format(
                    self.current_time, self.data
                )
            )

    def _make_interpolated_getter(self, key: str):
        _key = key

        def interpolated_getter():
            np_current_time_utc = np.array(
                datetime.datetime.utcfromtimestamp(
                    self.current_time.timestamp()
                ) if self.current_time.tzinfo is not None
                else self.current_time, dtype='datetime64[m]'
            )
            # gets the relative position
            minute = (
                np_current_time_utc
                - self.initial_time
            ).astype(float)
            return self.interpolators[_key](minute)

        return interpolated_getter
