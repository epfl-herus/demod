"""Inventory of the documentation of all methods or attributes for 'cards'.

It does not contain all the methods as some are already documented
in the code.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime, time, timedelta

from numpy import ndarray

from ..datasets.base_loader import DatasetLoader
from ..simulators.base_simulators import SimLogger, Simulator
from ..simulators.heating_simulators import (
    BuildingThermalDynamics,
    HeatingSystem,
)


from .sim_types import (
    ActivityLabels,
    AppliancesDict,
    HeatOutputs,
    HeatingControls,
    HeatingDemand,
    StateLabels,
    Subgroup,
    Subgroups,
    Temperatures,
    PDF,
    TPM,
    TPMs,
    ThermostatsStates,
)
from .data_types import DataInput


class Sim:
    """Documentation of the getters."""

    ###################################
    # Common to different simulators #
    ###################################
    def get_thermal_gains(self) -> ndarray:
        """Return the current :py:attr:`~.Inputs.thermal_gains`."""
        pass

    #######################
    # Activity simulators #
    #######################
    def get_occupancy(self) -> ndarray:
        """Return the current :py:attr:`~.Inputs.occupancy`."""
        pass

    def get_active_occupancy(self) -> ndarray:
        """Return the current :py:attr:`~.Inputs.active_occupancy`."""
        pass

    def get_n_doing_activity(self, activity: str) -> ndarray:
        """Get the number of residents doing the desired activity.

        The simulator must have activity_labels.

        Args:
            activity:
                the name of the requested activity as a string

        Raises:
            ValueError: If the given value of input is not valid
            Exception: If the simulator object has not been initialized
                with the :py:attr:`activity_labels` attributes
            TypeError: If the input state has an undesired type

        Returns:
            The number of occupants performing the activity as ndarray
            of integers.
        """

    ######################
    # Weather simulators #
    ######################
    def get_irradiance(self) -> float:
        """Return current :py:attr:`~.Inputs.irradiance`."""
        pass

    def get_outside_temperature(self) -> float:
        """Return current :py:attr:`~.Inputs.outside_temperature`."""
        pass

    ######################
    # Heating simulators #
    ######################
    def get_temperatures(self) -> Temperatures:
        """Return the current :py:attr:`~.Inputs.temperatures`."""
        pass

    def get_thermostat_states(self) -> ThermostatsStates:
        """Return the current :py:attr:`~.Inputs.thermostat_states`."""
        pass

    def get_target_temperatures(self) -> Temperatures:
        """Return the current :py:attr:`~.Params.target_temperatures`."""
        pass

    def get_heat_demand(self) -> HeatingDemand:
        """Return the current :py:attr:`~.Inputs.heat_demand`."""
        pass

    def get_controls(self) -> HeatingControls:
        """Return the current :py:attr:`~.Inputs.heating_controls`."""
        pass

    def get_heat_outputs(self) -> HeatOutputs:
        """Return the current :py:attr:`~.Inputs.heat_outputs`."""
        pass

    def get_fuel_consumptions(self) -> Dict[str, ndarray]:
        """Return the cuurrent fuel consumptions.

        Returns:
            fuel_dictionary: Key correspond to the kind of fuel

        Unit: Depends on the type of fuels , Watts [W] or Flow [m^3/s]

        """
    #########################
    # Appliances simulators #
    #########################
    def get_energy_consumption(self) -> ndarray:
        """Return current energy consumption.

        Returns the consumption of each simulated households.
        Energy consumption includes electricity, fuels, heat, ...

        :Unit: Watts [W]
        """
        pass

    def get_power_demand(self) -> ndarray:
        """Return current electricity consumption.

        Returns the consumption of each simulated households.

        :Unit: Watts [W]
        """
        pass

    def get_dhw_heating_demand(self) -> ndarray:
        """Return the current :py:attr:`~.Inputs.dhw_heating_demand`.

        Returns the total heat required by the appliances of hot water.
        The unit is W/K, which means that the heat depend on the
        temperature difference between the cold water and the target
        temperature for the hot water.
        """


class Loader:
    """Inventory of the data loader methods and attributes.

    Attributes can be passed to data loader object when created.
    They can specify a loading method

    Attributes:
        population_type: A string that represent the method used to
            differentiate the households of a population, based on different
            subgroups.
            Possible values:

                * 'resident_number'
                    The population subgroups depend on the number of
                    residents in the households.
                * 'crest'
                    Same as resident number
                * 'household_types'
                    Population is split in subgroups that differentiate
                    families (one-two parents), single, couples
                    based on household_type and n_residents

        activity_type: The type of activity that should be loaded from
            a Time-Of-Use dataset.
            Known values

                * 'Sparse9States'
                    Add link to description :
                * '4_States'
                    Add link to description :

        allow_pickle: Whether to allow using pickle to read the parsed
            data. Make sure that the files are secured when allow_pickle
            is True, as it is not protected against attacks.
            Set False by default.

        refresh_time: The time of the day at which the data should be
            reloaded. (Used mainly for TOU-based datasets
            and weather simulators.)
    """

    activity_type: str
    allow_pickle: bool
    refresh_time: time
    population_type: str

    def load_tpm(self, subgroup: Subgroup) -> Tuple[TPMs, StateLabels, PDF]:
        """Load a set of :py:attr:`~.Params.tpms`.

        Args:
            subgroup: The subgroup of the desired tpm.

        Returns:
            the transition probability matrix,
            the labels and the initial pdf
        """
        pass

    def load_sparse_tpm(
        self, subgroup: Subgroup
    ) -> Tuple[TPMs, StateLabels, ActivityLabels, PDF]:
        """Load a set of :py:class:`~demod.utils.sparse.Sparse_TPM`.

        Also returns states labels and activity labels.
        State labels map the n_states of the tpm to an integer value
        representing how many residents are performing an activity.
        Activity labels maps those activity to activity names.

        Args:
            subgroup: The subgroup of the desired tpm.

        Returns:
            the sparse tpm,
            the states labels, the activity labels, and the initial pdf
        """
        pass

    def load_population_subgroups(
        self, population_type: str
    ) -> Tuple[Subgroups, List[float], int]:
        """Load the subgroups and their numbers of a population.

        The population refers to the households population.
        Returns the list of subgroups, the proportion of each
        subgroups in the population and the total number of
        households for this population.
        Different splitting can be specified using the the
        :py:attr:`population_type` argument.

        Returns:
            subgroups_list, subgroup_prob, total_population
        """
        pass

    def load_clearness_tpms(self) -> Tuple[TPM, StateLabels, timedelta]:
        """Return TPM for the clearness of the sky, with the labels.

        The tpm containains the probability that the sky clearness
        changes at each step.

        Returns:
            1. The TPM of clearness
            2. Labels containing the clearness value of each TPM states
            3. The step size of the tpm, resolution of the transitions.
        """
        pass

    def load_temperatures_arma(self) -> Dict[str, float]:
        """Load the parameters for a temperature arma model.

        Returns:
            arma_dict: contains the parameters of the arma model.
        """


class Inputs:
    r"""Inventory of the possible simulators inputs.

    Shows the name of the input and links it to a related getter.

    Attributes:
        occupancy: The number of residents that are at home.

            :Unit: people, :py:obj:`int`
            :Related Getter: :py:meth:`~.Sim.get_occupancy`

        active_occupancy: The number of residents that are active and at
            home.

            :Unit: people, :py:obj:`int`
            :Related Getter: :py:meth:`~.Sim.get_active_occupancy`

        activities_dict: The dictionary containing the activities of the
            residents.

            :Unit: people, :py:obj:`int`
            :Related Getter: :py:meth:`~.Sim.get_activities`

        thermal_gains: The heat produced by the subjects of the
            simulator.

            :Unit: heat, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_thermal_gains`

        occupancy_thermal_gains: The heat produced by the occupants.

            :Unit: heat, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_thermal_gains`

        lighting_thermal_gains: The heat produced by the lighting.

            :Unit: heat, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_thermal_gains`

        appliances_thermal_gains: The heat produced by the appliances.

            :Unit: heat, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_thermal_gains`

        irradiance: Current irradiance.

            :Unit: intensity, :math:`[W/m^2]`
            :Related Getter: :py:meth:`~.Sim.get_irradiance`

        outside_temperature: Current temperature outdoor.

            :Unit: temperature, :math:`[^{\circ} C]`
            :Related Getter: :py:meth:`~.Sim.get_outside_temperature`


        temperatures: A dictionary containing the temperature for
            different parts of the house:

                * temperatures['building']
                    The walls of the building.
                * temperatures['interior']
                    The inside of the house, indoor.
                * temperatures['emitter']
                    The radiators.
                * temperatures['cylinder']
                    The hot water tank.

            :Unit: temperature, :math:`[^{\circ} C]`
            :Related Getter: :py:meth:`~.Sim.get_temperatures`

        power_consumption: Current consumption of electricity.

            :Unit: power, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_power_consumption`

        dhw_heating_demand: The heat required for heating the water for
            domestic hot water.

            :Unit: capacity, :math:`[W/K]`
            :Related Getter: :py:meth:`~.Sim.get_dhw_heating_demand`

        heat_outputs: The heat produced by a
            :py:attr:`~.Params.heating_system`. Usually has keys

            * 'space_heating', the heat output for space heating
            * 'dhw', the heat output for domestic hot water
            * 'total', the sum of all the outputs

            :Unit: power, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_heat_outputs`

        thermostat_states: Dictionary containing the values for
            different thermostats.
            The states can be ON or OFF = True or False.

            :Unit: binary, :py:obj:`bool`
            :Related Getter: :py:meth:`~.Sim.get_thermostat_states`

        heat_demand: Dictionary with the demand of heat for space heating
            and domestic hot water. Keys are

            * 'space_heating', the heat demand for space heating
            * 'dhw', the heat demand for domestic hot water

            :Unit: heat, :math:`[W]`
            :Related Getter: :py:meth:`~.Sim.get_heat_demand`

        heating_controls: Dictionary with the controls that are sent
            to the heating system. Keys are

            * 'heater_on', if the heating system should be activated
            * 'heat_water_on', if the heating system should heat water for dhw
            * 'space_heating_on', if the heating system should heat space

            :Unit: binary, :py:obj:`bool`
            :Related Getter: :py:meth:`~.Sim.get_controls`

        external_outside_temperature: Input from an external component for
            the outside temperature.

            :Unit: temperature, :math:`[^{\circ} C]`
            :Related Getter: :py:meth:`~.Sim.get_outside_temperature`

        external_irradiance: Input from an external component for
            the irradiance.

            :Unit: irradiance, :math:`[W/m^2]`
            :Related Getter: :py:meth:`~.Sim.get_irradiance`

        external_target_temperature: Input from an external component for
            the room temperatures.

            :Unit: temperature, :math:`[^{\circ} C]`
            :Related Getter: :py:meth:`~.Sim.get_target_temperatures`

        external_heat_outputs: Input from an external component for
            :py:attr:`heat_outputs`

        external_dhw_outputs: Input from an external component for
            the domestic hot water part of :py:attr:`heat_outputs`

        external_sh_outputs: Input from an external component for
            the space heating part of :py:attr:`heat_outputs`

        external_cylinder_temperature: Input from an external cylinder.

            :Unit: temperature, :math:`[^{\circ} C]`
            :Related Getter: None

        has_external_cylinder: array of True and False depending on the
            household has an external simulated cylinder

            :Unit: binary, :py:obj:`bool`
            :Related Getter: None





    """

    active_occupancy: ndarray

    irradiance: float

    temperatures: Temperatures

    thermal_gains: ndarray
    occupancy_thermal_gains: ndarray
    lighting_thermal_gains: ndarray
    appliances_thermal_gains: ndarray

    power_consumption: ndarray
    dhw_heating_demand: ndarray
    heat_outputs: HeatOutputs
    thermostat_states: ThermostatsStates
    heat_demand: HeatingDemand
    heating_controls: HeatingControls

    external_target_temperature: Tuple[ndarray, Temperatures]
    external_heat_outputs: Tuple[ndarray, HeatOutputs]
    external_dhw_outputs: ndarray
    external_sh_outputs: ndarray
    external_cylinder_temperature: ndarray
    has_external_cylinder: ndarray


class Params:
    """Inventory of the possible simulators parameters.

    Attributes:
        n_households: The number of households to simulate.

        n_households_list: Used for
            :py:class:`~demod.simulators.base_simulators.MultiSimulator`
            . The number of households to simulate for each subsimulator.

        logger: Stores the variables at every simulation steps.

        data: The :py:class:`~demod.datasets.base_loader.DatasetLoader` that
            should be used.

        subsimulator: Used for
            :py:class:`~demod.simulators.base_simulators.MultiSimulator`
            . The class of simulator to be used as subsimulator.

        time_aware: A boolean specifying if the simulator should keep
            track of the datetime during the simulation. Usually used
            for simulators that can update their attributes depending on
            the datetime.

        start_datetime: A `datetime.datetime object
            <https://docs.python.org/3/library/datetime.html#datetime-objects>`_
            specifiying the moment of
            the start of the simulation.

        step_size: A `datetime.timedelta object
            <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_
            specifying the duration of a simulator step.

        subgroup: A python dictionary containing information on a
            certain types
            of people or households from the dataset.
            a subgroup has different key-value pairs that specifiy
            socio-techno-economico-types of the population.
            The possible keys used in demod's subgroups are defined
            below.

            .. note::
                Combination of different keys is based on intersection, which
                means that only households/persons that respect all
                the conditions implied by the keys will be included.

            :Possible keys:

                * subgroup['n_residents']
                    int, The number of residents in the household
                * subgroup['household_type']
                    int, The number of residents in the household
                    From :py:class:`~demod.datasets.GermanTOU.loader.GTOU`

                    * 1 = One person household
                    * 2 = Couple without kid
                    * 3 = Single Parent with at least one kid under 18
                      and the other under 27
                    * 4 = Couple with at least one kid under 18
                      and the other under 27
                    * 5 = Others

                * subgroup['life_situation']
                    int, The life situation of the persons.
                    From :py:class:`~demod.datasets.GermanTOU.loader.GTOU`

                    * 1 = Self-employed, freelancer, farmer, family worker
                    * 2 = Employee, worker, civil servant, judge,
                      temporary / professional soldier, voluntary social /
                      ecological / cultural year, voluntary military service,
                      federal voluntary service
                    * 3 = Trainee (also intern, volunteer)
                    * 4 = Partial Work Time (work or retirement)
                    * 5 = On parental leave (with an employment contract
                      that has not been terminated)
                    * 6 = Student, Pupil
                    * 7 = Unemployed
                    * 8 = Retired or early retirement
                    * 9 = Permanently disabled
                    * 10 = Housewife / househusband/ houseparent
                    * 11 = Not gainfully employed for other reasons

                    .. note::
                        This works better with single persons households
                        in general.

                * subgroup['household_position']
                    int, The position of the person in the household or family.
                    From :py:class:`~demod.datasets.GermanTOU.loader.GTOU`

                    * 1 = Main revenue maker
                    * 2 = Partner or Spouse
                    * 3 = Kid

                    .. note::
                        This works better with single persons households
                        or person subgroup.


                * subgroup['age']
                    int or tuple, The age of the participants, if tuple
                    returns all inside interval.
                * subgroup['hh_mean_age']
                    tuple, The mean age of all memeber of the household.
                    Includes all households with mean age inside the
                    interval given as tuple.
                * subgroup['gender']
                    int, 1 = man, 2 = woman, 3 = other
                * subgroup['weekday']
                    int or list, The day of the week (1 to 7). If list,
                    include all days in list. (1 = Monday, 7 = Sunday).
                * subgroup['quarter']
                    int or list, The quarter of the year (1 to 4).


        subgroups: A list of :py:attr:`subgroup`.

        subgroups_list: Same as :py:attr:`subgroups` but more explicit
            name.

        tpm: A transition probability matrix. It is a matrix that
            stores the probability of changing from one state to an
            other.

        tpms: An iterable of :py:attr:`tpm` where tpms succeeding each
            other over time. Usually iterable over one day.

        population_sampling_algo: A string representing the algorithm to
            use for sampling the population.
            Possibile values detailed in
            :py:func:`~demod.simulators.util.sample_population`

        appliances_dict: A dictionary that defines some appliances
            properties. The values mapped by the keys are all array_like
            of size = appliances_dict['number'].

            * appliances_dict['name']
                *str*, The name of the appliances
            * appliances_dict['type']
                *str*, The type of the appliance. Possible values in
                :py:attr:`appliance_type`
            * appliances_dict['related_activity']
                *str*, The activity that correspond to this appliance
                usage. Possible values in
                :py:attr:`appliance_activity`
            * appliances_dict['uses_water']
                *bool*, True if the appliance consumes water.
            * appliances_dict['mean_elec_consumption']
                *float*, Average electrical consumption of the appliances
                in :math:`[W]`
            * appliances_dict['standby_consumption']
                *float*, Electrical consumption of the appliances during
                the stand-by (off) mode, in :math:`[W]`
            * appliances_dict['mean_water_consumption']
                *float*, Average water consumption of the appliances
                in :math:`[l/min]`
            * appliances_dict['inactive_switch_off']
                *bool*, Whether the appliance should switch off if the
                occupants stop the activity, leave or sleep.
            * appliances_dict['equipped_prob']
                *float*, Probability that a household is equipped with
                this appliance.
            * appliances_dict['mean_duration']
                *float*, The average duration of a use-cycle of this
                appliance in :math:`[min]`
            * appliances_dict['after_cycle_delay']
                *float*, The duration after a use-cycle, for which the
                appliance cannot be used in :math:`[min]`
            * appliances_dict['switch_on_prob_crest']
                *float*, Probability that the appliance is switched on
                following the CREST appliance model.
            * appliances_dict['heat_gains_ratio']
                *float*, The proportion of power that is converted into
                heat gains.
            * appliances_dict['mean_power_factor']
                *float*, See CREST to know what it is.
            * appliances_dict['poisson_sampling_lambda_liters']
                *float*, For water appliances, the sampling of the liters
                consumed is made using a poisson distribution, see
                :py:func:`numpy.random.poisson`

            * appliances_dict['number']
                *int*, Special value which is not an array.
                The number of appliances present in the appliances_dict.

        appliance_type: A string that represent the type of appliance.
            If the name is followed by '_2', it signifies that the
            appliance is a secondary appliance of that type.
            Nested levels indicate subtypes.
            Nested name should always finish with '_basename'.
            Types are very important for datset compatibilities.
            The current naming is purely arbitrary, but it could be
            changed in the future for compatibilty with an official
            naming convention from other sources.

            Possible values:

            * 'freezer'

                * 'chest_freezer'
                * 'fridge_freezer'
                * 'upright_freezer

            * 'fridge'

            * 'phone'

                * 'fixed_phone'
                * 'mobile_phone'
                * 'smart_phone'
                * 'answermachine_phone'

            * 'tablet'
            * 'speaker'

                * 'cd_speaker'
                * 'radio_speaker'
                * 'hifi_speaker'

            * 'tv'

                * 'crt_tv'
                * 'tft_tv'
                * 'lcd_tv'
                * 'led_tv'
                * 'dlp_tv'
                * 'plasma_tv'
                * 'oled_tv'

            * 'box'

                * 'tv_box'
                * 'internet_box'
                * 'dual_box'
                Internet + TV
                * 'wifi_box'

            * 'console'

                * 'gaming_console'
                * 'blueray_console'
                * 'dvd_console'

            * 'computer'

                * 'laptop_computer'
                * 'fixed_computer' (Desktop)
                * 'gaming_computer'

            * 'monitor'

                * 'crt_monitor'
                * 'tft_monitor'
                * 'lcd_monitor'
                * 'led_monitor'
                * 'dlp_monitor'
                * 'touchscreen_monitor'
                * 'plasma_monitor'
                * 'oled_monitor'

            * 'projector'

            * 'printer'

                * 'fax_printer'

            * 'lamp'

                * 'christmas_lamp'

            * 'clock'

                * 'alarm_clock'

            * 'hob'

                * 'electric_hob'
                * 'gaz_hob'

            * 'oven'
            * 'microwave'
            * 'kettle'
            * 'waterboiler'
            * 'coffemachine'

                * 'beans_coffemachine'
                * 'capsule_coffemachine'

            * 'toaster'
            * 'fitnessmachine'
            * 'dishwasher'
            * 'washingmachine'
            * 'dryer'

                * 'tumble_dryer'
                * 'washer_dryer'
                    Washing Machine that do both washing and drying.

            * 'iron'
            * 'heater'

                * 'water_heater'

                    * 'des_water_heater'
                        Domestic electric storage
                        water heater

                    * 'einst_water_heater'
                        Instant electric hot water

            * 'cleaner'

                * 'vacuum_cleaner'

            * 'shower'

                * 'electric_shower'

            * 'basin'
            * 'sink'
            * 'bath'

        appliance_activity: A string that represents the activity
            related to the the appliance usage.
            Possible values:

            * 'dishwashing'
                Washing the dishes, running the dishwasher.
            * 'self_washing'
                General activity for washing itself (showering, bathing,
                drying, ...)
            * 'cleaning'
                Cleaning the house (brooming, vaccum, ...)
            * 'level'
                For appliances that do not have a usage depending on the
                activity, but switch on and off depending on the pump
                function (fridges and freezers)
            * 'active_occupancy'
                Only depends on the active presence of occupants.
            * 'constant'
                Always consumes, no matter the activity
            * 'laundry'
                Related to cloths cleaning. (Washing machine, dryer)
            * 'ironing'
                Related to passing the iron.
            * 'watching_tv'
                Related to watching the tv.
            * 'cooking'
                General purpose for cooking activities.
            * 'electronics'
                General purpose for using electronics. (Computer, consoles)

        equipped_sampling_algo: A string representing the algorithm to
            use for sampling the appliances of a house.
            Possible values detailed in
            :py:meth:`~demod.simulators.appliance_simulators.AppliancesSimulator.sample_available_appliances`

        real_profiles_algo: A string representing the algorithm to
            use for sampling the real load profiles of appliances.
            Possible values detailed in
            :py:meth:`~demod.simulators.appliance_simulators.AppliancesSimulator.sample_real_load_profiles`

        bulbs_sampling_algo: The algorithm to sample the lighting bulbs
                installed in each house. See
                :py:meth:`~demod.simulators.lighting_simulators.CrestLightingSimulator.sample_bulbs_configuration`.

        initialization_method: A name of a method for initialization.
            Possible methods:

            * 'off'
                Everything is off a the start of the simulation
            * 'on'
                Everything is on at the start of the simulation

        heating_system: The heating system simulator used in the simulation.

        building: The building simulator used in the simulation.

        target_temperatures: The target
            :py:attr:`~demod.utils.cards_doc.Inputs.temperatures`
            from the thermostats.

        heatdemand_algo: The algorithm to compute the heat demand for
            space heating.

            * 'room_estimation'
                Try to approximate the exact
                amount of heat required to heat the room to the target
                temperature
            * 'heat_max_emmiters'
                Computes the heat required only for
                heating the emmiters to their maximum temperature
                (target_temperature + deadband)



        initial_active_occupancy: The active occupancy at the start of
            the simulation. Used for the initialization of a simulator.

        initial_activities_dict: The activities at the start of
            the simulation. Used for the initialization of a simulator.

        initial_clearness: The clearness at the start of the simulation.

        initial_irradiance: The irradiance at the start of the simulation.

        initial_outside_temperature:
            The
            :py:attr:`~demod.utils.cards_doc.Inputs.outside_temperature`
            at the start of the simulation.

        initial_temperatures:
            The :py:attr:`~demod.utils.cards_doc.Inputs.temperatures`
            at the start of the simulation.

        initial_controls:
            The :py:attr:`~demod.utils.cards_doc.Inputs.heating_controls`
            at the start of the simulation.

        initial_heat_demand:
            The :py:attr:`~demod.utils.cards_doc.Inputs.heat_demand`
            at the start of the simulation.

        interpolation_kind: Specifies the kind of interpolation used by
            the
            `scipy interpolation function
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_
            .


    """

    n_households: int
    n_households_list: List[int]
    logger: SimLogger
    data: DataInput
    subsimulator: Simulator
    time_aware: bool
    start_datetime: datetime
    step_size: timedelta
    subgroup: Subgroup
    subgroups: Subgroups
    subgroups_list: Subgroups
    tpm: TPM
    tpms: TPMs
    population_sampling_algo: str
    appliances_dict: AppliancesDict
    appliance_type: str
    appliance_activity: str
    equipped_sampling_algo: str
    initialization_method: str

    heating_system: HeatingSystem
    building: BuildingThermalDynamics
    target_temperatures: Tuple[ndarray, Temperatures]
    heatdemand_algo: str

    initial_active_occupancy: ndarray
    initial_clearness: float
    initial_irradiance: float
    initial_outside_temperature: float
    initial_temperatures: Temperatures
    initial_controls: HeatingControls

    interpolation_kind: Union[str, int]
