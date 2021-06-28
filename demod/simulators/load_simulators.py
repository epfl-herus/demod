"""Simulators for the full households loads."""

from demod.utils.data_types import DataInput
from demod.utils.sim_types import HeatOutputs, Subgroups, Temperatures
from typing import Any, List, Union
from demod.simulators.util import sample_population
from demod.datasets.base_loader import DatasetLoader
from demod.datasets.Germany.loader import GermanDataHerus
import warnings
import numpy as np
import pandas as pd
import datetime

from .base_simulators import Simulator, TimeAwareSimulator, cached_getter
from .weather_simulators import ClimateSimulator, CrestClimateSimulator, RealInterpolatedClimate
from .sparse_simulators import (
    SparseTransitStatesSimulator,
    SubgroupsActivitySimulator,
)
from .appliance_simulators import AppliancesSimulator, OccupancyApplianceSimulator
from .lighting_simulators import CrestLightingSimulator
from .heating_simulators import AbstractHeatingSimulator, FiveModulesHeatingSimulator
from .crest_simulators import Crest4StatesModel


def _instantiate_activity_simulator(
        data: DataInput,
        subgroups: Subgroups,
        counts: List[int],
        start_datetime: datetime.datetime,
    ) -> Any:

    errors = []
    try:
        return SubgroupsActivitySimulator(
                subgroups,
                counts,
                subsimulator=SparseTransitStatesSimulator,
                start_datetime=start_datetime,
                data=data,
            )
    except NotImplementedError as err:
        errors.append(err)

    try:
        return Crest4StatesModel(
            sum(counts), data=data, start_datetime=start_datetime
            )
    except NotImplementedError as err:
        errors.append(err)

    raise NotImplementedError((
        'LoadSimulator could not create activity simulator for this'
        'dataset : {}. \n Following errors: {}').format(
            data, errors
        )
    )


def _instantiate_climate(
        data: DataInput,
        start_datetime: datetime.datetime,
        step_size: datetime.timedelta,
    ) -> Any:

    errors = []
    try:
        return RealInterpolatedClimate(
            start_datetime=start_datetime,
            step_size=step_size,
            data=data,
        )

    except NotImplementedError as err:
        errors.append(err)

    try:
        return CrestClimateSimulator(
            data, start_datetime, step_size=step_size,
        )
    except NotImplementedError as err:
        errors.append(err)

    raise NotImplementedError((
        'LoadSimulator could not create climate simulator for this'
        'dataset : {}. \n Following errors: {}').format(
            data, errors
        )
    )


class LoadSimulator(TimeAwareSimulator):
    """Simulates the full load of households.

    This simulator accepts different simulator as inputs.

    The simulator can accept an external weather simulator that simulates
    temperature and irradiation.

    It can also accept external heating systems handling the heat
    production.

    The data inputs of this simulator can vary depending wether the
    heating simulator or the weather simulator is included.
    Also, the simulator might use a different activity simulator
    based on the data input. (see the code for more information)


    Contains:
        - The activity of the residents
        - The weather (Temperature and Irradiance)
        - Lighting
        - Appliances (electric + domestic hot water)
        - Heating (building thermal dynamics + heating system)

    Attributes:
        include_heating: Wether to include the heating of the households
        include_climate: Wether to simulate internally the climate (requires
            external inputs if not.)

    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.heatdemand_algo`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.initial_outside_temperature`
        :py:attr:`include_heating`
        :py:attr:`include_climate`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.utils.cards_doc.Loader.load_sparse_tpm`
        :py:meth:`~demod.utils.cards_doc.Loader.load_tpm`
        :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
        :py:meth:`~demod.datasets.tou_loader.LoaderTOU.load_activity_probability_profiles`
        :py:meth:`~demod.datasets.base_loader.ApplianceLoader.load_appliance_ownership_dict`
        :py:meth:`~demod.datasets.base_loader.ApplianceLoader.load_appliance_dict`
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_crest_lighting`
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_bulbs_config`
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_heating_system_dict`
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_thermostat_dict`
        :py:meth:`~demod.datasets.base_loader.HeatingLoader.load_buildings_dict`
    Step input
        None
    Optional step input
        :py:attr:`~demod.utils.cards_doc.Inputs.external_outside_temperature`
        :py:attr:`~demod.utils.cards_doc.Inputs.external_irradiance`
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
        1 Minute
    """

    climate: ClimateSimulator
    activity_simulator: TimeAwareSimulator
    appliance_simulator: AppliancesSimulator
    light_sim: Simulator
    heating: AbstractHeatingSimulator
    include_climate: bool
    include_heating: bool

    def _get_model_params_dict(self, model_params):
        # check that the input was a dictionary, else try to parse a file
        if model_params is None:
            model_params_dict = {}
        elif type(model_params) is dict:
            model_params_dict = model_params
        elif type(model_params) is str:
            model_params_dict = self._convert_input_str_to_params(model_params)
        else:
            TypeError(
                "model_params must be str of the file path or dict "
                "containing model parameters."
            )

        return model_params_dict

    def __init__(
        self,
        n_households,
        start_datetime=datetime.datetime(2014, 1, 1, 4, 0, 0),
        include_heating=True,
        include_climate=True,
        initial_outside_temperature=5.0,
        data=GermanDataHerus(),
        **kwargs
    ):



        super().__init__(
            n_households=n_households,
            start_datetime=start_datetime,
            **kwargs
        )


        # sample the population
        subgroups, pdf, _ = data.load_population_subgroups()
        counts = sample_population(
            self.n_households, pdf
        )

        # initialize all simulators in the correct order
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.activity_simulator = _instantiate_activity_simulator(
                data,
                subgroups,
                counts,
                start_datetime,
            )

        active_occupancy = self.activity_simulator.get_active_occupancy()

        self.appliance_simulator = OccupancyApplianceSimulator(
            subgroups,
            counts,
            initial_active_occupancy=active_occupancy,
            start_datetime=start_datetime,
            data=data
        )


        if include_climate:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.climate = _instantiate_climate(
                    data,
                    start_datetime,
                    self.step_size,
                )
            initial_outside_temperature = (
                self.climate.get_outside_temperature()
            )
        else:
            self.climate = None
        self.include_climate = include_climate

        self.light_sim = CrestLightingSimulator(
            self.n_households,
            data=data,
            bulbs_sampling_algo='randn'
            )

        if include_heating:
            self.heating = FiveModulesHeatingSimulator(
                self.n_households,
                initial_outside_temperature,
                data=data,
            )
        self.include_heating = include_heating

        super().initialize_starting_state(initialization_time=start_datetime)

    def step(
        self,
        external_outside_temperature: float = None,
        external_irradiance: float = None,
        external_target_temperature: Union[Temperatures, np.ndarray] = None,
        external_heat_outputs: Union[
            np.ndarray,
            HeatOutputs,
        ] = None,
        external_dhw_outputs: np.ndarray = None,
        external_sh_outputs: np.ndarray = None,
        external_cylinder_temperature: np.ndarray = None,
    ) -> None:
        """Step function for a load simulator.

        Can accepts external inputs for some of the subsimulators.
        """
        # one minutes steps

        # first updates the climate
        if self.include_climate:
            self.climate.step()

        if (
            (not self.include_climate)
            and (
                (external_irradiance is None)
                or (external_outside_temperature is None)
            )
        ):
            raise ValueError(
                "Inputs must be given for external_irradiance and "
                "temperature if Loadsimulator does not include a "
                "climate simulator."
            )
        # get external climate and irradiance if given
        outside_temperature = (
            external_outside_temperature
            if external_outside_temperature is not None
            else self.climate.get_outside_temperature()
        )
        irradiance = (
            external_irradiance
            if external_irradiance is not None
            else self.climate.get_irradiance()
        )

        # then perform the updates on the simulators

        if (self.current_time_step % 10) == 0:
            self.activity_simulator.step()

        active_occupancy = self.activity_simulator.get_active_occupancy()

        self.appliance_simulator.step(active_occupancy)
        self.light_sim.step(active_occupancy, irradiance)

        if self.include_heating:

            self.heating.step(
                outside_temperature=outside_temperature,
                irradiance=irradiance,
                dhw_heating_demand=(
                    self.appliance_simulator.get_dhw_heating_demand()
                ),
                occupancy_thermal_gains=(
                    self.activity_simulator.get_thermal_gains()
                ),
                lighting_thermal_gains=(
                    self.light_sim.get_thermal_gains()
                ),
                appliances_thermal_gains=(
                    self.appliance_simulator.get_thermal_gains()
                ),
                external_target_temperature=external_target_temperature,
                external_heat_outputs=external_heat_outputs,
                external_dhw_outputs=external_dhw_outputs,
                external_sh_outputs=external_sh_outputs,
                external_cylinder_temperature=external_cylinder_temperature,
            )
        super().step()


    @cached_getter
    def get_power_demand(self):
        # sum over all the appliances
        _heating = (
            self.heating.get_power_demand()
            if self.include_heating
            else np.zeros(self.n_households)
        )
        return (
            self.appliance_simulator.get_power_demand()
            + self.light_sim.get_power_demand()
            + _heating
        )

    @cached_getter
    def get_total_heat_demand(self):
        return self.heating.get_total_heat_demand()

    @cached_getter
    def get_dhw_heat_demand(self):
        return self.heating.get_dhw_heat_demand()

    @cached_getter
    def get_sh_heat_demand(self):
        return self.heating.get_sh_heat_demand()

    def get_temperatures(self):
        return self.heating.get_temperatures()
