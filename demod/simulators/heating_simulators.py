"""Heating simulation simulating the heat load of households."""
from __future__ import annotations
from ..utils.sim_types import (
    HeatOutputs,
    HeatingControls,
    HeatingDemand,
    Temperatures,
    ThermostatsStates,
)
from typing import Dict, Tuple, Union
from demod.datasets.base_loader import HeatingLoader
from demod.utils.data_types import DataInput
import warnings
import numpy as np
import datetime
from .base_simulators import SimLogger, Simulator, cached_getter
from .util import assign_external_array, assign_external_dict
from ..utils.monte_carlo import (
    monte_carlo_from_1d_pdf,
    monte_carlo_from_1d_cdf,
)
from ..utils.distribution_functions import check_valid_cdf
from ..datasets.Germany.loader import GermanDataHerus


class VariableThermostatTemperatureSimulator(Simulator):
    """Simulator for variable room thermostats.

    Based on the Living lab study by
    [Sovacool2020]_ that shows 6 different kinds of
    heating patterns.

    * *Cool Conservers*, Often adjust temperature to try and cut bills
    * *Steady and Savvy*, Rarely adjust their heating as they are fine with 18-20c
    * *Hot and Cold Fluctuators*, Often adjust temperature to get comfortable
    * *On-Demand Sizzlers*, Some like it hotter or want to spend more than others in their home
    * *On-off Switchers*, Turn it on and off to try and make sure home is only warm when someone is in
    * *Toasty Cruisers*,  Love feeling cosy and prefer not to put clothes on if they are cold


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.initial_occupancy`
        :py:attr:`~demod.utils.cards_doc.Params.initial_act_occ`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        None.
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.occupancy`
        :py:attr:`~demod.utils.cards_doc.Inputs.active_occupancy`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermostat_temperatures`
    Step size
        10 Minutes.
    """

    LOW_T = 16.0

    def load_patterns(self):
        pattern_labels = [
            "Cool Conservers",  # Often adjust temperature to try and cut bills
            "Steady and Savvy",  # Rarely adjust their heating as they are fine with 18-20c
            "Hot and Cold Fluctuators",  # Often adjust temperature to get comfortable
            "On-Demand Sizzlers",  # Some like it hotter or want to spend more than others in their home
            "On-off Switchers",  # Turn it on and off to try and make sure home is only warm when someone is in
            "Toasty Cruisers",  #  Love feeling cosy and prefer not to put clothes on if they are cold
        ]
        fixed_pattern = [False, True, False, False, False, True]
        mean_T = [17, 18.5, 20, 23, 20, 21]
        std_T = [1.5, 1, 2, 2, 0.5, 1]
        prob_changing_comfort = [0.1, 0.05, 0.1, 0.1, 0, 0.05]

        return (
            pattern_labels,
            fixed_pattern,
            mean_T,
            std_T,
            prob_changing_comfort,
        )

    def initialize_fixed_patterns(self):
        """Generates start and stop times for the different patterns
        This is completely randomly attributed for each household.
        A beter option would be to look in some data.
        """
        fixed_pattern_dict = {}
        fixed_pattern_dict["wake up hour"] = 6 + monte_carlo_from_1d_pdf(
            [0.25] * 4, n_samples=self.n_households
        )
        fixed_pattern_dict["wake up minutes"] = 10 * monte_carlo_from_1d_pdf(
            [0.5] + [0.1] * 5, n_samples=self.n_households
        )
        # patterns where the switch on during the day occurs or not
        fixed_pattern_dict["change during day"] = np.array(
            monte_carlo_from_1d_pdf([0.5] * 2, n_samples=self.n_households),
            dtype=bool,
        )
        # stay for at least 30 min after waking up, then leave and change heating
        fixed_pattern_dict[
            "minutes before leaving"
        ] = 30 + 10 * monte_carlo_from_1d_pdf(
            [0.05] * 20, n_samples=self.n_households
        )
        fixed_pattern_dict["leaving hour"] = (
            fixed_pattern_dict["wake up hour"]
            + (
                fixed_pattern_dict["wake up minutes"]
                + fixed_pattern_dict["minutes before leaving"]
            )
            // 60
        ) % 24
        fixed_pattern_dict["leaving minutes"] = (
            fixed_pattern_dict["wake up minutes"]
            + fixed_pattern_dict["minutes before leaving"]
        ) % 60
        # come back time
        fixed_pattern_dict[
            "minutes before returning"
        ] = 240 + 10 * monte_carlo_from_1d_pdf(
            [0.05] * 20, n_samples=self.n_households
        )
        fixed_pattern_dict["returning hour"] = (
            fixed_pattern_dict["leaving hour"]
            + (
                fixed_pattern_dict["leaving minutes"]
                + fixed_pattern_dict["minutes before returning"]
            )
            // 60
        ) % 24
        fixed_pattern_dict["returning minutes"] = (
            fixed_pattern_dict["leaving minutes"]
            + fixed_pattern_dict["minutes before returning"]
        ) % 60

        # where people go to bed
        fixed_pattern_dict["bed time hour"] = (
            20
            + monte_carlo_from_1d_pdf([0.2] * 5, n_samples=self.n_households)
        ) % 24
        fixed_pattern_dict["bed time minutes"] = 10 * monte_carlo_from_1d_pdf(
            [0.5] + [0.1] * 5, n_samples=self.n_households
        )

        self.fixed_pattern_dict = fixed_pattern_dict

    def attribute_patterns(self, pattern_attribution_method="living lab"):
        (
            pattern_labels,
            fixed_pattern,
            mean_T,
            std_T,
            prob_changing_comfort,
        ) = self.load_patterns()
        self.pattern_labels = pattern_labels

        if pattern_attribution_method == "random":
            self.pattern_type = np.random.randint(
                0, len(pattern_labels), size=self.n_households
            )
        elif (
            pattern_attribution_method == "living lab"
        ):  # paper Sovacool et al. 2020
            # attribute along the distribution form the paper
            self.pattern_type = monte_carlo_from_1d_pdf(
                [0.06, 0.24, 0.17, 0.12, 0.28, 0.13], self.n_households
            )

        elif np.isin(pattern_attribution_method, pattern_labels):
            # attributes the same pattern to all the residents
            self.pattern_type = np.ones(
                self.n_households, dtype=int
            ) * pattern_labels.index(pattern_attribution_method)
        else:
            raise ValueError("unkown argument pattern_attribution_method")

        self.patterns_mean_T = np.array(mean_T)
        self.max_deviation_T = np.array(std_T)
        self.prob_changing_comfort = prob_changing_comfort

    def __init__(
        self,
        n_households,
        initial_occupancy,
        initial_act_occ,
        start_datetime,
        *args,
        **kwargs
    ):
        """Initialize a Simulator with variable temperature

        Args:
            n_households (int): The number of households to be simiulated
            initial_occupancy (ndarray): the occupancy at the start
            initial_act_occ (ndarray): the active occupancy at the start of the simulation
            time (datetime.datetime): a datetime object for the current time
            pattern_attribution_method (str): kwarg that determine the method for the attribution of the
                temperature patterns
        """
        super().__init__(n_households, *args, **kwargs)
        self.attribute_patterns(*args, **kwargs)
        self.last_occupancy = initial_occupancy
        self.last_act_occ = initial_act_occ
        self.time = start_datetime
        self.initialize_starting_state(0)

    def initialize_starting_state(self, start_time_step):
        # basic initialization, assumes everyone is sleeping
        self.thermostats_T = np.array(self.patterns_mean_T[self.pattern_type])
        self.initialize_fixed_patterns()
        return super().initialize_starting_state(
            start_time_step=start_time_step
        )

    def step(
        self,
        occupancy,
        active_occupancy,
        increment=datetime.timedelta(minutes=10),
    ):
        self.time += increment
        self.update_thermostat_temperatures(occupancy, active_occupancy)
        super().step()

    def update_fluctuators(self, occupancy, active_occupancy, label):
        """Often adjust temperature to get comfortable.
        same implementation for cool conservers, and on demand sizzler

        Args:
            occupancy (ndarray(int)): The occupancy in the households
            active_occupancy (ndarray(int)): The active occupancy in the household
        """
        i = self.pattern_labels.index(label)
        mask = self.pattern_type == i

        new_T = self.thermostats_T[mask]
        # check in which case the households are
        mask_leaving = (occupancy[mask] == 0) & (self.last_occupancy[mask] > 0)
        mask_arrival = (occupancy[mask] > 0) & (self.last_occupancy[mask] == 0)
        mask_go_bed = (
            (active_occupancy[mask] == 0)
            & (self.last_act_occ[mask] > 0)
            & ~mask_arrival
            & ~mask_leaving
        )
        mask_wake_up = (
            (active_occupancy[mask] > 0)
            & (self.last_act_occ[mask] == 0)
            & ~mask_arrival
            & ~mask_leaving
        )

        # if leaving => set to minimal temperature
        new_T[mask_leaving] = self.LOW_T
        # if arriving => set to a high temperature
        new_T[mask_arrival] = self.patterns_mean_T[i] + self.max_deviation_T[i]
        # if going to bed => set to a medium temperature
        new_T[mask_go_bed] = self.patterns_mean_T[i]
        # if waking up => set to a high temperature
        new_T[mask_wake_up] = self.patterns_mean_T[i] + self.max_deviation_T[i]

        # change for comfort when active occuppancy and with the random
        # sample of comfort changing
        mask_comfort = (active_occupancy[mask] > 0) & np.array(
            np.random.binomial(
                1, self.prob_changing_comfort[i], size=np.sum(mask)
            ),
            dtype=bool,
        )

        # sample a random change of temperature
        rand_T_increase = (
            np.random.randint(
                0, 2 * self.max_deviation_T[i] + 1, size=np.sum(mask_comfort)
            )
            / 2.0
        )

        # assign the comfort changed temperature
        # print(rand_T_increase, self.thermostats_T, mask, mask_comfort)
        new_T[mask_comfort] = self.patterns_mean_T[i] + rand_T_increase

        self.thermostats_T[mask] = new_T

    def update_onoff_switchers(self, occupancy):
        """Turn it on and off to try and make sure home is only warm when someone is in
        Args:
            occupancy (ndarray(int)): The occupancy in the households

        Note:
            The temperature is always constant when heating, and is 0 if not heating (no occupants)
        """
        i = self.pattern_labels.index("On-off Switchers")
        mask = self.pattern_type == i
        new_T = self.thermostats_T[mask]

        mask_occupancy = occupancy[mask] > 0
        # when there is an occupant

        new_T[mask_occupancy] = self.patterns_mean_T[i]
        # when no occupant, temperature is low
        new_T[~mask_occupancy] = self.LOW_T

        self.thermostats_T[mask] = new_T

    def update_constant_patterns(self, label):

        # check which part of the day we are
        hour = self.time.hour
        minute = self.time.minute

        i = self.pattern_labels.index(label)
        mask = self.pattern_type == i
        new_T = self.thermostats_T[mask]

        # gets the mask where the constant transitions occur
        mask_wake_up = (
            self.fixed_pattern_dict["wake up hour"][mask] == hour
        ) & (self.fixed_pattern_dict["wake up minutes"][mask] == minute)
        # patterns where the switch on during the day occurs or not
        mask_leaving = np.logical_and.reduce(
            (
                self.fixed_pattern_dict["change during day"][mask],
                self.fixed_pattern_dict["leaving hour"][mask] == hour,
                self.fixed_pattern_dict["leaving minutes"][mask] == minute,
            )
        )

        mask_arrival = (
            self.fixed_pattern_dict["returning hour"][mask] == hour
        ) & (self.fixed_pattern_dict["returning minutes"][mask] == minute)
        mask_go_bed = (
            self.fixed_pattern_dict["bed time hour"][mask] == hour
        ) & (self.fixed_pattern_dict["bed time minutes"][mask] == minute)

        # if leaving => set a a low temperature, constant will not assume no one is there
        new_T[mask_leaving] = self.patterns_mean_T[i] - self.max_deviation_T[i]
        # if arriving => set to a high temperature
        new_T[mask_arrival] = self.patterns_mean_T[i] + self.max_deviation_T[i]
        # if going to bed => set to a medium temperature
        new_T[mask_go_bed] = self.patterns_mean_T[i]
        # if waking up => set to a high temperature
        new_T[mask_wake_up] = self.patterns_mean_T[i] + self.max_deviation_T[i]

        self.thermostats_T[mask] = new_T

    def update_thermostat_temperatures(self, occupancy, active_occupancy):
        # update the six kinds of heating profiles

        self.update_fluctuators(occupancy, active_occupancy, "Cool Conservers")
        self.update_fluctuators(
            occupancy, active_occupancy, "Hot and Cold Fluctuators"
        )
        self.update_fluctuators(
            occupancy, active_occupancy, "On-Demand Sizzlers"
        )
        self.update_onoff_switchers(occupancy)
        self.update_constant_patterns("Steady and Savvy")
        self.update_constant_patterns("Toasty Cruisers")

        self.last_occupancy = np.array(occupancy)
        self.last_act_occ = np.array(active_occupancy)

    def get_thermostat_temperatures(self):
        """Getter for the temperatures

        Returns:
            ndarray: the thermostat temperatures for each households
        """
        return np.array(self.thermostats_T)


class AbstractHeatingSimulator(Simulator):
    def get_energy_consumption(self):
        raise NotImplementedError(
            "{} must implement get_energy_consumption().".format(
                type(self).__name__
            )
        )


class BuildingThermalDynamics(Simulator):
    """Simulator for the buildling thermal dynamics.

    It simulates the energy flows between different house components.
    The room, the heating system, the walls, the outside.
    Based on the current temperatures, at each step it updates the temperature
    based on energy flows.
    To compute the energy flows, it takes information from the
    climate and from the heat gains inside the house at each step.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.heating_system`
        :py:attr:`~demod.utils.cards_doc.Params.initial_outside_temperature`
        :py:attr:`~demod.utils.cards_doc.Params.step_size`
        :py:attr:`~demod.utils.cards_doc.Params.target_temperatures`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_buildings_dict`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.outside_temperature`
        :py:attr:`~demod.utils.cards_doc.Inputs.irradiance`
        :py:attr:`~demod.utils.cards_doc.Inputs.dhw_heating_demand`
        :py:attr:`~demod.utils.cards_doc.Inputs.occupancy_thermal_gains`
        :py:attr:`~demod.utils.cards_doc.Inputs.lighting_thermal_gains`
        :py:attr:`~demod.utils.cards_doc.Inputs.appliances_thermal_gains`
        :py:attr:`~demod.utils.cards_doc.Inputs.heat_outputs`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_temperatures`
    Step size
        Any.
    """

    step_size: datetime.timedelta

    def __init__(
        self,
        n_households: int,
        heating_system: HeatingSystem,
        initial_outside_temperature: float,
        step_size: datetime.timedelta = datetime.timedelta(minutes=1),
        target_temperatures: Tuple[np.ndarray, Temperatures] = None,
        data: DataInput = GermanDataHerus(),
        **kwargs
    ) -> None:
        """Create a Simulator for the buildling thermal dynamics.

        It updates the temperature inside the different parts of the building
        at different times of the day.

        Args:
            n_households: number of households to simulate
            heating_system: the heating system of the building.
            initial_outside_temperature: Outside temperature at initial time.
            data: The data to use. Defaults to 'Germany'.
        """
        super().__init__(n_households, **kwargs)
        self.outside_temperature = initial_outside_temperature

        self.step_size = step_size

        self.buildings = self.read_buildings(data=data)
        self.assign_building_to_households()

        self.initialize_temperatures(
            initial_outside_temperature, target_temperatures
        )
        self._read_heating_system(heating_system)

        self.initialize_starting_state()

    def _read_heating_system(self, heating_system: HeatingSystem):
        self.H_cyl_loss = heating_system.h_loss
        self.C_cyl = heating_system.C_cyl

    def assign_building_to_households(self):
        cdf = np.cumsum(self.buildings["equipped_prob"])
        try:
            check_valid_cdf(cdf)
        except ValueError as e:
            raise Exception(
                "Incorrect probabilities distribuiton for the heating systems equippememnt probabilities"
            )
        self.building_types = monte_carlo_from_1d_cdf(
            cdf, n_samples=self.n_households
        )

        self.emitters_nominal_temperature = self.buildings[
            "emitters_target_temperature"
        ][self.building_types]
        self.C_em = self.buildings["emitters_capacitance"][self.building_types]
        self.H_em = self.buildings["emitters_transfer_coef"][
            self.building_types
        ]

        self.C_b = self.buildings["ext_capacitance"][self.building_types]
        self.C_i = self.buildings["int_capacitance"][self.building_types]
        # Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance (W/K)
        self.H_bi = self.buildings["build_int_transfer_coef"][
            self.building_types
        ]
        # Thermal transfer coefficient between outside air and external building thermal capacitance (W/K)
        self.H_ob = self.buildings["out_build_transfer_coef"][
            self.building_types
        ]
        # Thermal transfer coefficient representing ventilation losses between outside air and internal building thermal capacitance (W/K)
        self.H_v = self.buildings["ventilation_transfer_coef"][
            self.building_types
        ]
        self.building_area = self.buildings["floor_area"][self.building_types]
        self.irradiance_multiplier = self.buildings["irradiance_multiplier"][
            self.building_types
        ]

    def read_buildings(self, data: HeatingLoader):

        return data.load_buildings_dict()

    def initialize_temperatures(
        self, outside_temperature, target_temperatures
    ):
        """Initialize the house temperatures based on the target_temperatures.

        TODO: improve, for input of type Temperatures.
        TODO: improve with different algorithm.

        Args:
            outside_temperature: The temperature of outside.
            target_temperatures: The target temperatures.
        """
        self.temperatures = {}
        if target_temperatures is None:
            warnings.warn(
                "No target temperature specified in {class_name}. Will sample from N(20,2)".format(
                    class_name=type(self).__name__
                )
            )
            # generates the target from a random distribuution
            target_temperatures = np.random.randn(self.n_households) * 2 + 20.0

        # Assign initial building temperatures
        # In hot climates or warm weather the initial building temperatures are likely to
        # be increased external temperature
        # assume the house is already around good temperature uniform between deadbands
        self.temperatures[
            "interior"
        ] = target_temperatures + np.random.uniform(
            -1, 1, self.n_households
        )  # TODO: use deadband variable
        # assume that the building is between the inside and the outside temperature
        self.temperatures["building"] = (
            outside_temperature + target_temperatures
        ) / 2.0

        # emitters are on or off depending if the target temperature is reached on not
        self.temperatures["emitter"] = np.where(
            self.temperatures["interior"] >= target_temperatures,
            self.temperatures["interior"],
            self.emitters_nominal_temperature,
        )

        # no cooling in our model
        # self.temperatures['cooling'] = np.array(self.temperatures['interior'])

        # Set the initial temperature of the hot water tank
        self.temperatures["cylinder"] = 50.0 + 2 * np.random.uniform(
            size=self.n_households
        )

        # Set cold_water inlet temperature
        self.temperatures["cold_water"] = 10.0

    def step(
        self,
        outside_temperature: float,
        irradiance: float,
        dhw_heating_demand: np.ndarray,
        occupancy_thermal_gains: np.ndarray,
        lighting_thermal_gains: np.ndarray,
        appliances_thermal_gains: np.ndarray,
        heat_outputs: HeatOutputs,
    ):
        """Update the building temperatures based on heat flows.

        Args:
            outside_temperature: current outside temperature
            irradiance: Current irradiance
            dhw_heating_demand: The demand for heating the water for dhw
            occupancy_thermal_gains: Thermal gains from occupancy
            lighting_thermal_gains: Thermal gains from lighting
            appliances_thermal_gains: Thermal gains from appliances
            heat_outputs: The heat produce by a heating system and put
                inside the system.
        """

        # Get or calculate the thermal gains

        # ... from primary heating system to space
        phi_hSpace = heat_outputs["space_heating"]
        # ... from cooling system to space
        # no cooling system implemented

        # ... from primary heating system to hot water
        phi_hWater = heat_outputs["dhw"]

        # ... from passive solar gains
        phi_s = irradiance * self.irradiance_multiplier
        # ... from occupants, lighting and appliances
        phi_c = (
            occupancy_thermal_gains
            + lighting_thermal_gains
            + appliances_thermal_gains
        )

        # ... from solar thermal collector (if any)
        # dblPhi_collector = aSolarThermal(intRunNumber).GetPhi_s(currentTimeStep)

        # Get the variable hot water demand heat transfer coefficient
        H_dhw = dhw_heating_demand

        # Calculate change in building external thermal node temperatures
        d_T_b = (self.step_size.total_seconds() / self.C_b) * (
            -(self.H_ob + self.H_bi) * self.temperatures["building"]
            + self.H_bi * self.temperatures["interior"]
            + self.H_ob * outside_temperature
        )

        # Calculate change in building internal thermal node temperatures
        d_T_i = (self.step_size.total_seconds() / self.C_i) * (
            self.H_bi * self.temperatures["building"]
            - (self.H_v + self.H_bi + self.H_em + self.H_cyl_loss)
            * self.temperatures["interior"]
            + self.H_v * outside_temperature
            + self.H_em * self.temperatures["emitter"]
            + self.H_cyl_loss * self.temperatures["cylinder"]
            + phi_s
            + phi_c
        )

        # Calculate change in heat emitter temperatures (heating radiators only)
        d_T_em = (self.step_size.total_seconds() / self.C_em) * (
            self.H_em * self.temperatures["interior"]
            - self.H_em * self.temperatures["emitter"]
            + phi_hSpace
        )

        # Calculate change in temperature of hot water cylinder
        d_T_cyl = (self.step_size.total_seconds() / self.C_cyl) * (
            self.H_cyl_loss * self.temperatures["interior"]
            - (self.H_cyl_loss + H_dhw) * self.temperatures["cylinder"]
            + H_dhw * self.temperatures["cold_water"]
            + phi_hWater  # + phi_collector
        )

        # Update building thermal node temperatures for this time step
        self.temperatures["building"] += d_T_b
        self.temperatures["interior"] += d_T_i
        self.temperatures["emitter"] += d_T_em
        self.temperatures["cylinder"] += d_T_cyl

        return super().step()

    def get_temperatures(self):
        """Getter for the current temperatures.

        Returns a dictionary containing entries for each parts of the
        building
        temperatures:

            * temperatures['building']
            * temperatures['interior']
            * temperatures['emitter']
            * temperatures['cylinder']

        Each entry is a numpy array containing data for each of the households.

        Returns:
            dict: temperatures of different house components
        """
        return self.temperatures


class Thermostats(Simulator):
    """Simulator for the thermostats of the house.

    Simulates the state of different thermostats (can be ON or OFF
    = True or False).
    Thermostat control the temperature of different component.
    They are switched to on once the temperature of a component is
    below its target_temperature minus a dead_band.
    Components:

    * cylinder
    * emitters
    * interior (room temperature)

    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.initial_temperatures`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_thermostat_dict`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.temperatures`
    Optional step input
        :py:attr:`~demod.utils.cards_doc.Params.target_temperatures`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermostat_states`
        :py:meth:`~demod.utils.cards_doc.Sim.get_target_temperatures`
    Step size
        Any.

    Note:
        A future implementation could make it any_component friendly.
        It could be useful to simulate multiple rooms for example.
    """

    target_temperature: Dict[str, np.ndarray]

    def __init__(
        self,
        n_households: int,
        initial_temperatures: Temperatures = None,
        data: DataInput = GermanDataHerus(),
        **kwargs
    ):
        super().__init__(n_households, **kwargs)

        self.thermostat = self.read_thermostats(data=data)

        self.initialize_starting_state(initial_temperatures, **kwargs)

    def initialize_starting_state(
        self, initial_temperatures: Temperatures = None, **kwargs
    ):
        """Initialize the starting state of the thermostats.

        If initial_temperatures are provided the thermostats are set based on
        wether the temperature is higher than the target initial_temperatures.

        If not provided they will be off.

        Args:
            initial_temperatures: Temperatures at the start. Defaults to None.
        """

        self.initialize_target_temperatures()

        if initial_temperatures is None:
            # assign the temperatures to be the same
            temperatures = {}
            temperatures["cylinder"] = self.target_temperatures["dhw"]
            temperatures["interior"] = self.target_temperatures["space_heating"]
            temperatures["emitter"] = self.target_temperatures["emitter"]
        else:
            temperatures = initial_temperatures

        thermostat_states = {}
        # Determine initial thermostat states
        thermostat_states["hot_water"] = (
            temperatures["cylinder"] < self.target_temperatures["dhw"]
        )
        thermostat_states["space_heating"] = (
            temperatures["interior"] < self.target_temperatures["space_heating"]
        )
        thermostat_states["emitter"] = (
            temperatures["emitter"] < self.target_temperatures["emitter"]
        )

        self.thermostat_states = thermostat_states

        # TODO : check below combi boiler impelmemntation
        # for non combi boiler, the heating of the water is performed to keep cylinder at high t
        # self.thermostat_states['hot_water'][~self.is_combi_boiler] = self.target_temperatures['dhw'][~self.is_combi_boiler]

        return super().initialize_starting_state(**kwargs)

    def initialize_target_temperatures(self):
        """Initialiaze the target temperatures for each households."""
        target_temperature = {}
        # determine space heating thermostat set point
        r = monte_carlo_from_1d_cdf(
            self.thermostat["home_temperatures_cdf"],
            n_samples=self.n_households,
        )
        target_temperature["space_heating"] = self.thermostat[
            "home_temperatures_values"
        ][r]

        # determine hot water thermostat set point
        r = monte_carlo_from_1d_cdf(
            self.thermostat["water_temperatures_cdf"],
            n_samples=self.n_households,
        )
        target_temperature["dhw"] = self.thermostat[
            "water_temperatures_values"
        ][r]

        target_temperature["emitter"] = self.thermostat[
            "emitter_setpoints"
        ] * np.ones(self.n_households)

        self.target_temperatures = target_temperature

    def read_thermostats(self, data: HeatingLoader):
        return data.load_thermostat_dict()

    def set_new_target_temperature(
        self, target_temperatures: Tuple[np.ndarray, Temperatures]
    ):
        """Set new target temperatures.

        Args:
            target_temperatures: If array is used, it will be the target
                for space heating.
        """
        if isinstance(target_temperatures, dict):
            for key, item in target_temperatures.items():
                self.target_temperatures[key] = item
        elif isinstance(target_temperatures, np.ndarray):
            # assume the room temperature was given as target_temperature
            self.target_temperatures["space_heating"] = np.array(
                target_temperatures
            )
        else:
            raise TypeError("target_temperature must be dict or ndarray")

    def step(
        self,
        temperatures: Temperatures,
        target_temperatures: Tuple[np.ndarray, Temperatures] = None
    ) -> None:
        """Update the thermostat_states with the new temperatures.

        Can also update the target_temperatures based on optional
        :py:obj:`target_temperature`.

        Args:
            temperatures: The new temperatures dic.
            target_temperatures: New target temperatures, Optional.
                Defaults to None.
        """

        if target_temperatures is not None:
            self.set_new_target_temperature(target_temperatures)

        # Calculate the thermostat states for current time step, based partly on states in previous time step

        # following logic
        # thermostat is on if either :
        # is on and temperature has not reached T + deadband (continues to heat)
        # is off and temperature is below T - deadband (require switchon heating)

        # Hot water thermostat
        self.thermostat_states["hot_water"] = np.where(
            np.logical_or(
                self.thermostat_states["hot_water"]
                & (
                    temperatures["cylinder"]
                    < (
                        self.target_temperatures["dhw"]
                        + self.thermostat["deadband"]["hot_water"]
                    )
                ),
                ~self.thermostat_states["hot_water"]
                & (
                    temperatures["cylinder"]
                    <= (
                        self.target_temperatures["dhw"]
                        - self.thermostat["deadband"]["hot_water"]
                    )
                ),
            ),
            True,
            False,
        )
        # Room heating thermostat
        self.thermostat_states["space_heating"] = np.where(
            np.logical_or(
                self.thermostat_states["space_heating"]
                & (
                    temperatures["interior"]
                    < (
                        self.target_temperatures["space_heating"]
                        + self.thermostat["deadband"]["space_heating"]
                    )
                ),
                ~self.thermostat_states["space_heating"]
                & (
                    temperatures["interior"]
                    <= (
                        self.target_temperatures["space_heating"]
                        - self.thermostat["deadband"]["space_heating"]
                    )
                ),
            ),
            True,
            False,
        )
        # Emitters thermostat
        self.thermostat_states["emitter"] = np.where(
            np.logical_or(
                self.thermostat_states["emitter"]
                & (
                    temperatures["emitter"]
                    < (
                        self.target_temperatures["emitter"]
                        + self.thermostat["deadband"]["emitter"]
                    )
                ),
                ~self.thermostat_states["emitter"]
                & (
                    temperatures["emitter"]
                    <= (
                        self.target_temperatures["emitter"]
                        - self.thermostat["deadband"]["emitter"]
                    )
                ),
            ),
            True,
            False,
        )
        return super().step()

    def get_thermostat_states(self) -> ThermostatsStates:
        return self.thermostat_states

    def get_target_temperatures(self) -> Temperatures:
        return self.target_temperatures


class HeatDemand(Simulator):
    """Simulator for the heat demand of the households.

    It computes the heat demand for both:
    domestic hot water and space heating.

    There exist different algorithm for computing the heat demand.

    The hot water demand is calculate based on
    :py:attr:`~demod.utils.cards_doc.Inputs.dhw_heating_demand`
    and also accounts for losses
    in the hot water cylinder. It also aims at keeping the
    cylinder hot.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.heating_system`
        :py:attr:`~demod.utils.cards_doc.Params.building`
        :py:attr:`~demod.utils.cards_doc.Params.step_size`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.heatdemand_algo`
    Data
        None
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.temperatures`
        :py:attr:`~demod.utils.cards_doc.Params.target_temperatures`
        :py:attr:`~demod.utils.cards_doc.Inputs.dhw_heating_demand`
        :py:attr:`~demod.utils.cards_doc.Inputs.outside_temperature`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_heat_demand`
    Step size
        Any.
    """

    def __init__(
        self,
        n_households: int,
        heating_system: HeatingSystem,
        building: BuildingThermalDynamics,
        step_size: datetime.timedelta = datetime.timedelta(minutes=1),
        heatdemand_algo: str = "heat_max_emmiters",
        deadbands: dict[str:float] = {
            'space_heating': 1,
            'hot_water': 5,
            'emitter': 5
        },
        **kwargs
    ) -> None:
        """Initialize the heat demand calculator.

        Args:
            n_households: number of households simulated
            heating_system: The related heating system.
            building: The related building.
            algo: The algorithm to use. Defaults to 'heat_max_emmiters'.
        """
        super().__init__(n_households, **kwargs)

        self.target_heat_demand = {}

        self.step_size = step_size

        self.deadbands = deadbands

        self._read_building(building)
        self._read_heating_system(heating_system)

        self.target_space_heat_algo = heatdemand_algo
        self.initialize_starting_state(**kwargs)

    def initialize_starting_state(self, start_time_step=0, **kwargs):
        # sets the initial demand to 0
        self.target_heat_demand["dhw"] = np.zeros(self.n_households)
        self.target_heat_demand["space_heating"] = np.zeros(self.n_households)

        return super().initialize_starting_state(
            start_time_step=start_time_step, **kwargs
        )

    def _read_heating_system(self, heating_system):
        self.C_cyl = heating_system.C_cyl
        self.h_loss = heating_system.h_loss

    def _read_building(self, building):
        self.C_em = building.C_em
        self.H_em = building.H_em
        self.C_i = building.C_i
        self.H_v = building.H_v

        self.H_ibo = (1 / building.H_bi + 1 / building.H_ob) ** (-1)

    def _compute_target_heat_output_hot_water(
        self, temperatures, dhw_heating_demand, target_temperature
    ):
        # Get the hot water thermostat set point for this dwelling
        hot_water_target_temperature = (
            target_temperature['dhw']
            + self.deadbands['hot_water']
        )

        # Calculate target heat input required from heating system to deliver appropriate
        # temperature of hot water
        cylinder_part = (
            self.C_cyl
            / self.step_size.total_seconds()
            * (hot_water_target_temperature - temperatures["cylinder"])
        )
        hot_water_part = dhw_heating_demand * (
            temperatures["cylinder"] - temperatures["cold_water"]
        )
        loss_part = self.h_loss * (
            temperatures["cylinder"] - temperatures["interior"]
        )
        return cylinder_part + hot_water_part + loss_part

    def _compute_target_heat_output_space_heating(
        self, temperatures, target_temperature, outside_temperature=None
    ):

        # calculate the target heat delivery to heat emitters to achieve the set point
        if self.target_space_heat_algo == "heat_max_emmiters":
            #  Temperature deadband for emitters
            target_emitter_temperature = (
                target_temperature["emitter"] + self.deadbands['emitter']
            )
            dblPhi_hSpaceTarget = (
                self.C_em
                / self.step_size.total_seconds()
                * (target_emitter_temperature - temperatures["emitter"])
                + self.H_em
                * (temperatures["emitter"] - temperatures["interior"])
            )

        elif self.target_space_heat_algo == "room_estimation":
            # first check if the required parameter for this algo was given
            if outside_temperature is None:
                new_algo = "heat_max_emmiters"
                warnings.warn(
                    'outside_temperature must be define to use {old_algo} \
                    in {class_name}. \n Algo changed to "{new_algo}".'.format(
                        old_algo=self.target_space_heat_algo,
                        class_name=type(self).__name__,
                        new_algo=new_algo,
                    ),
                    UserWarning,
                )  # TODO: check if that is the best warning type to use
                self.target_space_heat_algo = new_algo
                return self._compute_target_heat_output_space_heating(
                    temperatures,
                    target_temperature,
                )
            # new implementation suggest that we target direclty at
            # heating the room and we don't focus at heating only the emmiters
            dblPhi_hSpaceTarget = (
                self.C_i
                / self.step_size.total_seconds()
                * (
                    target_temperature["space_heating"]
                    + self.deadbands['space_heating']
                    - temperatures["interior"]
                )
                - self.H_em
                * (temperatures["emitter"] - temperatures["interior"])
                + self.H_v * (temperatures["interior"] - outside_temperature)
                + self.H_ibo * (temperatures["interior"] - outside_temperature)
            )

        else:
            raise ValueError(
                "Invalid algo for target space heating, set a correct one in __init__"
            )

        return dblPhi_hSpaceTarget

    def step(
        self,
        temperatures: Temperatures,
        target_temperatures: Tuple[np.ndarray, Temperatures],
        dhw_heating_demand: np.ndarray,
        outside_temperature: float = None,
    ):
        """Update the demand for space heating and domestic hot water.

        Args:
            temperatures: The current temperatures.
            target_temperatures: The target tempertures from the thermostats.
            dhw_heating_demand: The heating demand for dhw.
            outside_temperature: The temperature of outside
                required only if
                :py:attr:`~demod.utils.cards_doc.Params.heatdemand_algo`
                is 'room_estimation'. Defaults to None.
        """

        self.target_heat_demand[
            "dhw"
        ] = self._compute_target_heat_output_hot_water(
            temperatures=temperatures,
            dhw_heating_demand=dhw_heating_demand,
            target_temperature=target_temperatures,
        )
        self.target_heat_demand[
            "space_heating"
        ] = self._compute_target_heat_output_space_heating(
            outside_temperature=outside_temperature,
            temperatures=temperatures,
            target_temperature=target_temperatures,
        )

        return super().step()

    @cached_getter
    def get_heat_demand(self) -> HeatingDemand:
        return self.target_heat_demand


class SystemControls(Simulator):
    """Simulator for the controls of the heating system.

    It checks which controls should be sent to the
    :py:class:`.HeatingSystem`, based on the heat demand and
    on the thermostats.

    It can handle combi boilers, which means that the boiler does not
    stay on to keep the cylinder at a high temperature.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`is_combi_boiler`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        None
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.dhw_heating_demand`
        :py:attr:`~demod.utils.cards_doc.Inputs.thermostat_states`
        :py:attr:`~demod.utils.cards_doc.Inputs.has_external_cylinder`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_controls`
    Step size
        Any.

    Attributes:
        is_combi_boiler:
            Specifies whether the heating system is a combi boiler or
            not.
            Combi boiler don't maintain the hot water cylinder to a high
            temperature and produce dhw only on demand.
    """

    is_combi_boiler: np.ndarray

    def __init__(
        self, n_households: int, is_combi_boiler: np.ndarray = None, **kwargs
    ) -> None:
        """Creates a simulator for the controls of the heating system.

        Args:
            n_households: The number of simuulated households.
            is_combi_boiler: Whether the boiler are combi. Defaults to None.
        """
        super().__init__(n_households, **kwargs)
        self.initialize_starting_state(**kwargs)

        # assigns the combi boilers
        self.is_combi_boiler = (
            np.array(is_combi_boiler)
            if is_combi_boiler is not None
            else np.zeros(n_households, dtype=bool)
        )

    def initialize_starting_state(self, **kwargs) -> None:
        """Initialize the controls to OFF."""
        # TODO: check if we want a better initalization
        self.heating_controls = {}
        self.heating_controls["heater_on"] = np.zeros(
            self.n_households, dtype=bool
        )
        self.heating_controls["heat_water_on"] = np.zeros(
            self.n_households, dtype=bool
        )
        self.heating_controls["space_heating_on"] = np.zeros(
            self.n_households, dtype=bool
        )
        return super().initialize_starting_state(**kwargs)

    def step(
        self,
        dhw_heating_demand: np.ndarray,
        thermostat_states: ThermostatsStates,
        has_external_cylinder: np.ndarray = None,
    ) -> None:
        """Update the heating system controls.

        Checks the themrmostats and water demand to decide
        whether the heating system must be on or off and where
        the heat output should flow (dhw or space_heating)

        Args:
            dhw_heating_demand: The demand for hot water
            thermostat_states:
                The states of the thermostats, mapped in a dict
                of boolean arrays.
            has_external_cylinder:
                array of bool depending on the household has an external
                simulated cylinder
        """
        if has_external_cylinder is None:
            # No external cylinder
            has_external_cylinder = np.full_like(self.is_combi_boiler, False)
        # Determine whether hot water heating is required
        heat_water_onoff = np.where(
            self.is_combi_boiler | has_external_cylinder,
            # If it's a combi system or that there is an external cylinder
            # system, then hot water control signal is determined
            # by hot water demand
            dhw_heating_demand > 0,
            # otherwise for regular or system boilers it depends
            # only on the thermostat states
            thermostat_states["hot_water"],
        )

        # space heating requires space thermostat and emitters to be on
        space_heating_onoff = (
            thermostat_states["space_heating"] * thermostat_states["emitter"]
        )

        # Determine with the heating system should be switched on if either the hot water is needed
        # or if the space heating is needed
        heater_onoff = heat_water_onoff | space_heating_onoff

        # assigns the new controls
        self.heating_controls["heater_on"] = heater_onoff
        self.heating_controls["heat_water_on"] = heat_water_onoff
        self.heating_controls["space_heating_on"] = space_heating_onoff
        return super().step()

    def get_controls(self) -> HeatingControls:
        return self.heating_controls


class CrestControls(SystemControls):
    """NotImplemented yet
    """
    pass

class HeatingSystem(Simulator):
    """Simulator for the heating system.

    It simulates the consumption of the heating system to provide
    requested heat data.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.initial_controls`
        :py:attr:`~demod.utils.cards_doc.Params.initial_heat_demand`
    Data
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_heating_system_dict`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.heating_controls`
        :py:attr:`~demod.utils.cards_doc.Inputs.heat_demand`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_fuel_consumptions`
        :py:meth:`~demod.utils.cards_doc.Sim.get_power_demand`
        :py:meth:`~demod.utils.cards_doc.Sim.get_energy_consumption`
        :py:meth:`~demod.utils.cards_doc.Sim.get_heat_outputs`
    Step size
        Any.
    """

    def __init__(
        self,
        n_households: int,
        data: DataInput = GermanDataHerus(),
        initial_controls: HeatingControls = None,
        initial_heat_demand: HeatingDemand = None,
        **kwargs
    ) -> None:
        """Initialize a heating system simulator.

        Args:
            n_households: Number of households to simulate.
            data: The dataset to use. Defaults to GermanDataHerus().
            initial_controls: The controls at the start of the
                simulation. Defaults to None.
            initial_heating_demand: The heating demand at the start of
                the simuation.
        """
        super().__init__(n_households, **kwargs)

        self.heating_systems = self.read_heating_system(data=data)
        self.assign_heating_systems_to_households()

        self.initialize_starting_state(initial_controls, initial_heat_demand)

    def initialize_starting_state(
        self,
        initial_controls: HeatingControls,
        initial_heat_demand: HeatingDemand,
    ) -> None:
        """Initialize the starting state of the heating system.

        The initialization depend on which paremeters are given as
        inputs.

        Args:
            initial_controls: value at the start of the simulation.
            initial_heat_demand: value at the start of the simulation.
        """
        initial_controls = initial_controls or {
            "heater_on": np.zeros(self.n_households, dtype=bool),
            "heat_water_on": np.zeros(self.n_households, dtype=bool),
            "space_heating_on": np.zeros(self.n_households, dtype=bool),
        }
        self.controls = initial_controls
        self.heat_outputs = {}
        self.heat_outputs["dhw"] = np.zeros(self.n_households)
        self.heat_outputs["space_heating"] = np.zeros(self.n_households)
        self.heat_outputs["total"] = np.zeros(self.n_households)

        if initial_heat_demand:
            # If we know the demand at the start of the simulation,
            # run one step to compute the initial outputs
            return super().initialize_starting_state(
                1,
                heating_controls=initial_controls,
                target_heat_demand=initial_heat_demand,
            )

        return super().initialize_starting_state(start_time_step=0)

    def read_heating_system(self, data: HeatingLoader) -> dict:
        """Read the heating system values from the dataset.

        Uses :py:meth:`load_heating_system_dict`.

        Args:
            data: The dataset that should be read.

        Returns:
            heating_system_dict: The heating system dictionary.
        """

        return data.load_heating_system_dict()

    def assign_heating_systems_to_households(self):
        cdf = np.cumsum(self.heating_systems["equipped_prob"])
        try:
            check_valid_cdf(cdf)
        except ValueError as e:
            raise Exception(
                "Incorrect probabilities distribuiton for the heating systems "
                " equipment probabilities: {} ".format(
                    self.heating_systems["equipped_prob"]
                )
            )
        self.heating_types = monte_carlo_from_1d_cdf(
            cdf, n_samples=self.n_households
        )

        self.dblP_pump = self.heating_systems["pump_power"][self.heating_types]
        self.dblP_standby = self.heating_systems["standby_power"][
            self.heating_types
        ]

        self.dblFuelFlowRate = self.heating_systems["fuel_flow_rate"][
            self.heating_types
        ]
        self.flow_rate_to_W = self.heating_systems["flow_rate_to_W"][
            self.heating_types
        ]
        self.fuel_types = self.heating_systems["fuel_type"][self.heating_types]

        # Heat output of unit
        self.max_Phi_h_space = self.heating_systems["heat_output_sh"][
            self.heating_types
        ]
        self.max_Phi_h_dhw = self.heating_systems["heat_output_dhw"][
            self.heating_types
        ]

        self.h_loss = self.heating_systems["cyl_loss"][self.heating_types]
        SPECIFIC_HEAT_CAPACITY_WATER = 4200.0
        self.C_cyl = (
            self.heating_systems["cyl_volume"][self.heating_types]
            * SPECIFIC_HEAT_CAPACITY_WATER
        )

        self.is_combi_boiler = self.heating_systems["is_combi"][
            self.heating_types
        ]

    @cached_getter
    def get_heat_outputs(self) -> HeatOutputs:
        return self.heat_outputs

    @cached_getter
    def get_pumps_power_consumptions(self) -> np.ndarray:
        """Get the power consumed by the heating system pumps.

        Returns:
            power_consumptions: consumpton for each household.
        """
        # the heating system pump will operate if space heating is on
        P_h = np.where(
            self.controls["space_heating_on"],
            self.dblP_pump,
            self.dblP_standby
        )
        return P_h

    @cached_getter
    def get_fuel_consumptions(self) -> Dict[str, np.ndarray]:
        out_dict = {}
        # check where the heating is occuring & ensure no future division by 0
        heating_occurs = self.controls["heater_on"] & (
            (self.max_Phi_h_space > 0) | (self.max_Phi_h_dhw > 0)
        )
        heat_outputs = self.get_heat_outputs()
        phi_space = heat_outputs["space_heating"]
        phi_dhw = heat_outputs["dhw"]

        for h_type in np.unique(self.fuel_types):
            if h_type == "Mainsgas":

                out_dict[h_type] = np.where(
                    (self.fuel_types == h_type) & heating_occurs,
                    self.dblFuelFlowRate
                    * 3600.  # m^3/ h -> m^3/ sec
                    * (
                        phi_space / self.max_Phi_h_space
                        + phi_dhw / self.max_Phi_h_dhw
                    ),
                    0,
                )
            elif h_type == "Electricity":

                out_dict[h_type] = np.where(
                    (self.fuel_types == h_type) & heating_occurs,
                    self.dblFuelFlowRate
                    * 1000.  # kW -> W
                    * (
                        phi_space / self.max_Phi_h_space
                        + phi_dhw / self.max_Phi_h_dhw
                    ),
                    0,
                )
            else:
                assert isinstance(h_type, str), "fuel types must be strings"
                raise ValueError("Type of fuel not recognized : " + h_type)
        return out_dict

    @cached_getter
    def get_energy_consumption(self) -> np.ndarray:
        """Retuurn the energy consumption.

        Only implements Mainsgas and Electricity at the moment.

        Returns:
            [type]: [description]
        """
        # in different fuels (gaz, wood, electricity)
        fuel_cons = self.get_fuel_consumptions()
        elec = fuel_cons["Electricity"] if "Electricity" in fuel_cons else 0
        gas = fuel_cons["Mainsgas"] if "Mainsgas" in fuel_cons else 0
        # convert gas flow to Watts
        gas = gas / self.dblFuelFlowRate * self.flow_rate_to_W
        return self.get_pumps_power_consumptions() + elec + gas

    @cached_getter
    def get_power_demand(self):
        """Getter for the power demand of the heating system.
        Returns the electricity used by the heating system,
        plus the electricity of running the pumps.

        Returns:
            ndarray: the power demand
        """
        fuel_cons = self.get_fuel_consumptions()
        elec = fuel_cons["Electricity"] if "Electricity" in fuel_cons else 0
        return self.get_pumps_power_consumptions() + elec

    def step(
        self,
        heating_controls: HeatingControls,
        target_heat_demand: HeatingDemand,
    ) -> None:
        """Perform a heating system step.

        It will compute the new heat output of the heating system using the
        target heat demand and the controls of the heating system.
        It also follows the constraints of the heating system, in terms of
        maximum heat outputs.
        If the heat demand exceeds the maximum capacity,
        domestic hot water is then prioritised.

        Args:
            heating_controls: Dictronnary containing the control arrays.
            target_heat_demand: Dictionnary containing the heat demand arrays.
        """
        #  Get the control signals from the heating controller

        # First case, if heat water is on
        # assign heat to hot water, bound by max and min values of the heating system
        requested_heat_output_dhw = np.clip(
            target_heat_demand["dhw"], 0.0, self.max_Phi_h_dhw
        )
        self.heat_outputs["dhw"] = np.where(
            heating_controls["heater_on"] & heating_controls["heat_water_on"],
            requested_heat_output_dhw,
            0.0,
        )
        # Compute remaining heat output
        remaining_heat_output = np.clip(
            self.max_Phi_h_space - requested_heat_output_dhw,
            0.0,
            self.max_Phi_h_space,
        )
        # then assign remaining required capacity to space heating
        requested_heat_output_sh = np.clip(
            target_heat_demand["space_heating"], 0.0, remaining_heat_output
        )
        self.heat_outputs["space_heating"] = np.where(
            (
                heating_controls["heater_on"]
                & heating_controls["heat_water_on"]
                & heating_controls["space_heating_on"]
            ),
            requested_heat_output_sh,
            0.0,
        )

        # Otherwise, if heat water is off, heat is required only for space heating
        # assign heat to space heating, bound by max and min values
        requested_heat_output_sh = np.clip(
            target_heat_demand["space_heating"], 0.0, self.max_Phi_h_space
        )
        self.heat_outputs["space_heating"] = np.where(
            (  # Space heating on but not water heating
                heating_controls["heater_on"]
                & (~heating_controls["heat_water_on"])
                & heating_controls["space_heating_on"]
            ),
            requested_heat_output_sh,
            self.heat_outputs["space_heating"],
        )

        # assign total heat output
        self.heat_outputs["total"] = (
            self.heat_outputs["space_heating"] + self.heat_outputs["dhw"]
        )

        # stores the controls for the getters
        self.controls = heating_controls

        return super().step()


class FiveModulesHeatingSimulator(AbstractHeatingSimulator):
    """Simulator for the heat load using 5 components.

    It simulates the energy consumption of an household required for
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

    * An external heating system
        it can pass its heat outputs through
        :py:meth:`set_external_heating_system`
        and in the step method:
        :py:attr:`~demod.utils.cards_doc.Inputs.external_heat_outputs`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_dhw_outputs`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_sh_outputs`
        If external heat outputs are given, the heat load simulator will
        return 0 power demand for those households.

    * Hot water storage cylinder
        if the heating system also has an external hot water storage cylinder,
        it can communicate using
        :py:attr:`~demod.utils.cards_doc.Inputs.external_cylinder_temperature`


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.initial_outside_temperature`
        :py:attr:`~demod.utils.cards_doc.Params.heatdemand_algo`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.step_size`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_heating_system_dict`
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_thermostat_dict`
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_buildings_dict`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.outside_temperature`
        :py:attr:`~demod.utils.cards_doc.Inputs.irradiance`
        :py:attr:`~demod.utils.cards_doc.Inputs.dhw_heating_demand`
        :py:attr:`~demod.utils.cards_doc.Inputs.occupancy_thermal_gains`
        :py:attr:`~demod.utils.cards_doc.Inputs.lighting_thermal_gains`
        :py:attr:`~demod.utils.cards_doc.Inputs.appliances_thermal_gains`
    Optional step input
        :py:attr:`~demod.utils.cards_doc.Inputs.external_target_temperature`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_heat_outputs`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_dhw_outputs`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_sh_outputs`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_cylinder_temperature`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_room_temperature`
        :py:meth:`~demod.utils.cards_doc.Sim.get_total_heat_demand`
        :py:meth:`~demod.utils.cards_doc.Sim.get_dhw_demand`
        :py:meth:`~demod.utils.cards_doc.Sim.get_sh_demand`
        :py:meth:`~demod.utils.cards_doc.Sim.get_power_demand`
        :py:meth:`~demod.utils.cards_doc.Sim.get_controls`
    Step size
        Any.
    """

    def __init__(
        self,
        n_households: int,
        initial_outside_temperature: float,
        heatdemand_algo: str = "heat_max_emmiters",
        data: DataInput = GermanDataHerus(),
        step_size: datetime.timedelta = datetime.timedelta(minutes=1),
        **kwargs
    ) -> None:
        super().__init__(n_households, **kwargs)

        # Initialize parameters to interact with external heating components
        self.use_external_heating_system = np.full(n_households, False)
        self.use_external_cylinder = np.full(n_households, False)
        self.cyl_dhw_demand = np.zeros(n_households, dtype=float)

        # intialize the 5 sub simulators
        self.heatingsystem = HeatingSystem(n_households, data=data)
        self.thermostats_sim = Thermostats(n_households, data=data)
        self.building_sim = BuildingThermalDynamics(
            n_households,
            self.heatingsystem,
            initial_outside_temperature,
            data=data,
            step_size=step_size,
            target_temperatures=self.thermostats_sim.get_target_temperatures()[
                "space_heating"
            ],
        )
        self.controls_sim = SystemControls(
            n_households, is_combi_boiler=self.heatingsystem.is_combi_boiler
        )
        self.heatdemand_sim = HeatDemand(
            n_households,
            self.heatingsystem,
            self.building_sim,
            heatdemand_algo=heatdemand_algo,
            deadbands=self.thermostats_sim.thermostat['deadband']
        )

        self.initialize_starting_state()

    def initialize_starting_state(self):

        # first step for ensuring that all the inputs outputs are set ?
        super().initialize_starting_state()

    def set_external_heating_system(self, new_heating_system: HeatingSystem):
        """Assign an external heating system.

        Will read some parameters of the new heating systems that are
        required for the heating simulation, such as wheter the heating
        systems have combi boilers, or the attributes aboutes heat
        transfers.

        Args:
            new_heating_system: The new :py:class:`.HeatingSystem` to
                consider.
        """

        # assigns heating system variables to other components
        self.controls_sim.is_combi_boiler = new_heating_system.is_combi_boiler
        self.heatdemand_sim._read_heating_system(new_heating_system)
        self.building_sim._read_heating_system(new_heating_system)

        self.use_external_heating_system = np.full(self.n_households, True)

    def _split_heat_outputs_dhw_sh(
        self, heat_output, heat_controls, heat_demand
    ):
        """Splits the heat output into dhw and sh outputs, given the heat demands and heat controls

        Args:
            heat_output (ndarray): The total heating output
            heat_controls (dict): The heating controls
            heat_demand (dict): The heating demand

        Returns:
            dict: Heat outputs splitted by dhw and sh
        """
        heat_outputs_dict = {}
        # first assign heat to dhw

        requested_heat_output_dhw = np.clip(heat_demand["dhw"], 0, heat_output)
        heat_outputs_dict["dhw"] = np.where(
            heat_controls["heater_on"] & heat_controls["heat_water_on"],
            requested_heat_output_dhw,
            0,
        )
        # then assign remaining required capacity to space heating
        remaining_heat_output = heat_output - requested_heat_output_dhw
        requested_heat_output_sh = np.clip(
            heat_demand["space_heating"], 0, remaining_heat_output
        )
        heat_outputs_dict["space_heating"] = np.where(
            heat_controls["heater_on"]
            & heat_controls["heat_water_on"]
            & heat_controls["space_heating_on"],
            requested_heat_output_sh,
            0,
        )

        # Otherwise, if heat water is off, heat is required only for space heating
        # assign heat to space heating, bound by max and min values
        requested_heat_output_sh = np.clip(
            heat_demand["space_heating"], 0, heat_output
        )
        heat_outputs_dict["space_heating"] = np.where(
            heat_controls["heater_on"]
            & (~heat_controls["heat_water_on"])
            & heat_controls["space_heating_on"],
            requested_heat_output_sh,
            heat_outputs_dict["space_heating"],
        )

        # the remaining heat not used should go to the cylinder ()
        remaining_heat_output = (
            heat_output
            - heat_outputs_dict["space_heating"]
            - heat_outputs_dict["dhw"]
        )
        heat_outputs_dict["dhw"] += remaining_heat_output

        return heat_outputs_dict

    def _set_external_heating_systems(self, external_heat_outputs_dict):
        """Check where a heat input was provided by external sources.

        nan values are present where no external heating system exists.

        Args:
            external_heat_outputs_dict: heat output dict

        Returns:
            ndarray(bool): Where there is an external heating system.
        """
        mask = np.full(self.n_households, False)
        if "dhw" in external_heat_outputs_dict:
            mask = np.where(
                np.isnan(external_heat_outputs_dict["dhw"]), mask, True
            )
        if "space_heating" in external_heat_outputs_dict:
            mask = np.where(
                np.isnan(external_heat_outputs_dict["space_heating"]),
                mask,
                True,
            )
        return mask

    def _assign_external_heat_outputs(
        self, external_heat_outputs, external_dhw_outputs, external_sh_outputs
    ):
        """Assigns the different external outputs.

        In the external arrays, nan values indicate that no value should
        be assigned
        external_dhw_outputs and external_sh_outputs will override the
        external_heat_outputs.

        Args:
            external_heat_outputs : array of the external output values
            external_dhw_outputs : array of the external output values
            external_sh_outputs : array of the external output values

        Returns:
            heat_output_dictionary
        """
        if isinstance(external_heat_outputs, dict):
            out_dict = external_heat_outputs
        else:
            out_dict = {}
        # convert the external heat_outputs as ndarray to the dict heat output format
        if isinstance(external_heat_outputs, np.ndarray):
            # convert to dict
            out_dict = self._split_heat_outputs_dhw_sh(
                external_heat_outputs,
                self.controls_sim.get_controls(),
                self.heatdemand_sim.get_heat_demand(),
            )

        if external_dhw_outputs is not None:
            # Creates the array to store the dhw
            out_dict["dhw"] = assign_external_array(
                np.full(external_dhw_outputs.shape, np.nan),
                external_dhw_outputs,
            )

        if external_sh_outputs is not None:
            # Creates the array to store the dhw
            out_dict["space_heating"] = assign_external_array(
                np.full(external_sh_outputs.shape, np.nan), external_sh_outputs
            )

        return out_dict

    def step(
        self,
        outside_temperature: float,
        irradiance: float,
        dhw_heating_demand: np.ndarray,
        occupancy_thermal_gains: np.ndarray,
        lighting_thermal_gains: np.ndarray,
        appliances_thermal_gains: np.ndarray,
        external_target_temperature: Union[Temperatures, np.ndarray] = None,
        external_heat_outputs: Union[
            np.ndarray,
            HeatOutputs,
        ] = None,
        external_dhw_outputs: np.ndarray = None,
        external_sh_outputs: np.ndarray = None,
        external_cylinder_temperature: np.ndarray = None,
    ) -> None:
        """Perform a heating system step.

        External array are optionals.
        For all the external arrays, nan values indicate that no value
        should be assigned.
        The internal simulated values are used instead for those households
        where nan values are provide.

        Args:
            outside_temperature: The outside temperature
            irradiance: Irradiance
            dhw_heating_demand: The heat demand for Domestic Hot Water
            occupancy_thermal_gains: The thermal gains of occupancy.
            lighting_thermal_gains: The thermal gains of lighting
            appliances_thermal_gains: The thermal gains of applainces
            external_target_temperature:
                Dictionary containing the different target_temperature,
                or a ndarray that will be interpreted as the
                inside temperature. Defaults to None.
            external_heat_outputs:
                The external heat outputs.
                If dictionarry, will read the different heat output (dhw, sh).
                If array, it will split the output depending on the
                requested heat, but will give priority to fullfill dhw.
                Defaults to None.
            external_dhw_outputs: array of the external output for dhw.
                will override the value from external_heat_outputs.
            external_sh_outputs: array of the external output for dhw.
                will override the value from external_heat_outputs.
            external_cylinder_temperature: array that specify at which
                temperature the external cylinder is.
                This allows to compute the heat transfers for the
                :py:class:`BuildingThermalDynamics`
        """

        # first get the inputs variables from the internal or external simulators
        temperatures = self.building_sim.get_temperatures()

        external_heat_outputs_dict = self._assign_external_heat_outputs(
            external_heat_outputs, external_dhw_outputs, external_sh_outputs
        )

        # records where heat comes from external heating systems
        self.use_external_heating_system = self._set_external_heating_systems(
            external_heat_outputs_dict
        )

        # Assign the temperature of the external cylinder
        temperatures["cylinder"] = assign_external_array(
            temperatures["cylinder"], external_cylinder_temperature
        )
        # also assigns external cylinder in sim
        self.building_sim.temperatures["cylinder"] = temperatures["cylinder"]


        # update the simulators
        self.thermostats_sim.step(
            temperatures=temperatures,
            target_temperatures=external_target_temperature
        )
        target_temperature = self.thermostats_sim.get_target_temperatures()

        self.controls_sim.step(
            dhw_heating_demand=dhw_heating_demand,
            thermostat_states=self.thermostats_sim.get_thermostat_states(),
            has_external_cylinder=self.use_external_cylinder,
        )
        self.heatdemand_sim.step(
            temperatures=temperatures,
            target_temperatures=target_temperature,
            dhw_heating_demand=dhw_heating_demand,
            outside_temperature=outside_temperature,
        )
        self.heatingsystem.step(
            heating_controls=self.controls_sim.get_controls(),
            target_heat_demand=self.heatdemand_sim.get_heat_demand(),
        )

        # Computes the heat outputs for updating the building Temperatures
        heat_outputs = assign_external_dict(
            self.heatingsystem.get_heat_outputs(), external_heat_outputs_dict
        )

        self.building_sim.step(
            outside_temperature=outside_temperature,
            irradiance=irradiance,
            dhw_heating_demand=dhw_heating_demand,
            occupancy_thermal_gains=occupancy_thermal_gains,
            lighting_thermal_gains=lighting_thermal_gains,
            appliances_thermal_gains=appliances_thermal_gains,
            heat_outputs=heat_outputs,
        )

        # dhw demand is changed depending on the external cylinder
        if external_cylinder_temperature is not None:
            self.use_external_cylinder = ~np.isnan(
                external_cylinder_temperature
            )
            self.cyl_dhw_demand = self._compute_dhw_external_cyl(
                temperatures, dhw_heating_demand, target_temperature["dhw"]
            )

        super().step()

    def _compute_dhw_external_cyl(
        self, temperatures, dhw_heating_demand, hot_water_target_temperature
    ):
        """Compute the dhw demand when there is an external cylinder."""
        hot_water_part = dhw_heating_demand * (
            hot_water_target_temperature - temperatures["cold_water"]
        )
        loss_part = self.heatdemand_sim.h_loss * (
            temperatures["cylinder"] - temperatures["interior"]
        )
        return hot_water_part + loss_part

    @cached_getter
    def get_room_temperature(self):
        return self.building_sim.temperatures["interior"]

    @cached_getter
    def get_total_heat_demand(self):
        """Return the total heat demand."""
        controls = self.controls_sim.get_controls()
        return controls["heater_on"] * (
            self.get_dhw_heat_demand() + self.get_sh_heat_demand()
        )

    @cached_getter
    def get_dhw_heat_demand(self):
        """Get the heat demand for domestic hot water.

        This demand correspond to the heat demand made to the
        :py:class:`.HeatingSystem`, by the
        :py:class:`SystemControls` and the :py:class:`HeatDemand`.
        Includes also the losses from the cylinder to the space and
        maintaining the cylinder to a high temperature.
        If an external cylinder is used, only computes the water
        necessary for dhw, and not the water required by the cylinder.
        """
        heat_demand_dict = self.heatdemand_sim.get_heat_demand()
        controls = self.controls_sim.get_controls()
        # changes the heat demand for hot water if external cylinder
        return np.where(
            self.use_external_cylinder,
            self.cyl_dhw_demand * controls["heat_water_on"],
            heat_demand_dict["dhw"] * controls["heat_water_on"],
        )

    @cached_getter
    def get_sh_heat_demand(self):
        heat_demand_dict = self.heatdemand_sim.get_heat_demand()
        controls = self.controls_sim.get_controls()
        return heat_demand_dict["space_heating"] * controls["space_heating_on"]

    @cached_getter
    def get_power_demand(self):
        return np.where(
            self.use_external_heating_system,
            0.0,  # an external system was set
            self.heatingsystem.get_power_demand(),
        )

    def get_energy_consumption(self):
        return self.get_power_demand()

    def get_thermostat_states(self):
        return self.thermostats_sim.get_thermostat_states()

    def get_target_temperatures(self):
        return self.thermostats_sim.get_target_temperatures()

    def get_temperatures(self):
        return self.building_sim.get_temperatures()

    def get_heat_outputs(self):
        return self.heatingsystem.get_heat_outputs()

    def get_controls(self):
        return self.controls_sim.get_controls()
