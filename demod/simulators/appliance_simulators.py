import datetime
from demod.utils.data_types import DataInput
from demod.utils.sim_types import Subgroups
from demod.utils.appliances import get_ownership_from_dict

from typing import List
from demod.datasets.base_loader import DatasetLoader
from demod.datasets.Germany.loader import GermanDataHerus
import warnings
import numpy as np
import pandas as pd
import os
import itertools
import inspect

from .base_simulators import Callbacks, TimeAwareSimulator, cached_getter
from ..utils.monte_carlo import monte_carlo_from_cdf
from .util import OLD_DATASET_PATH, subgroup_add_time
from ..helpers import subgroup_file


class AppliancesSimulator(TimeAwareSimulator):
    """Class for the appliances

    Args:
        Simulator ([type]): [description]
    """
    data: DatasetLoader

    def __init__(
        self,
        n_households,
        appliances_dict,
        *args,
        equipped_sampling_algo="basic",
        **kwargs
    ):
        """[summary]"""
        super().__init__(n_households, *args, **kwargs)
        self.appliances = appliances_dict

        # give appliances to the households
        self.available_appliances = self.sample_available_appliances(
            equipped_sampling_algo=equipped_sampling_algo
        )

    def initialize_starting_state(self, *args, **kwargs):
        """Initialize the starting states of the appliances.
        Calls as well the parents methods for initialization, passing
        args and kwargs.

        Returns:
            [type]: [description]
        """
        # stores the number of times before the appliance stops being used
        self.n_times_left = np.zeros_like(
            self.available_appliances, dtype=float
        )
        # stores the number of time till the appliances can be used again after refresh period
        delays = [
            np.random.randint(0, delay, size=self.n_households)
            if delay > 0
            else np.zeros(self.n_households, dtype=int)
            for delay in self.appliances["after_cycle_delay"]
        ]
        self.n_times_till_refresh = np.asarray(delays, dtype=int).T
        initialization_time = kwargs.pop("initialization_time", None)
        return super().initialize_starting_state(
            *args, initialization_time=initialization_time, **kwargs
        )

    def sample_available_appliances(
        self, equipped_sampling_algo: str = "basic"
    ) -> np.ndarray:
        """Initialize which appliances are available in the households.

        Args:
            equipped_sampling_algo: The method used for sampling.
                Available:

                    * 'basic':
                        uses a monte carlo sampling from
                        appliances['equipped dwellings probs']
                    * 'subgroup':
                        like basic sampling but uses
                        :py:meth:`self.data.load_subgroup_ownership_pdf`
                    * 'correlated':
                        check which appliances are usually coming
                        together. not implemented yet
                    * 'set_defined':
                        matches the requested set given as
                        :py:attr:`appliance_set`. not implemented yet
                    * 'all':
                        give all appliances to every households

                Defaults to 'basic'.

        Raises:
            ValueError: For unknown :py:attr:`equipped_sampling_algo`

        Return:
            available_array, an boolean array of size size =
            (n_household, n_appliances) where True means the appliance
            is available and False that it is not.
        """
        if equipped_sampling_algo == "basic":
            available_probs = self.appliances["equipped_prob"][None, :]
            rand = np.random.uniform(
                size=(self.n_households, self.appliances["number"])
            )
            return rand < available_probs

        elif equipped_sampling_algo == "subgroup":
            return self._sample_from_subgroup()

        elif equipped_sampling_algo == "correlated":
            # improve : would be better when reading a data set to think about the correlations between the appliances possession
            raise NotImplementedError()
        elif equipped_sampling_algo == "all":
            # Return array of True
            return np.ones(
                (self.n_households, self.appliances['number']),
                dtype=bool
            )
        else:
            raise ValueError(
                "Unknown equipped_sampling_algo : "
                + equipped_sampling_algo
            )

    def _sample_from_subgroup(self):
        raise NotImplementedError()

    def get_mask_available_appliances(self):
        # available if in the house and not already used and not in refresh time
        return np.logical_and.reduce(
            (
                self.available_appliances,
                self.n_times_left <= 0,
                self.n_times_till_refresh <= 0,
            )
        )

    def switch_on(self, indexes_household, indexes_appliance):
        """Function that switches on appliances at the given indices.
        Different models can be implemented with different time variations for the different
        appliances.

        Args:
            indexes_household ([type]): the index of the household.
            indexes_appliance ([type]): the index of the appliances, matching with the households

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    def switch_off_inactive(self, active_occupancy):

        raise NotImplementedError()

    def step(self, active_occupancy, switchon_probs):
        """Step function for the appliances.

        It
        1. controls the appliances usage,
        2. randomly samples switcho-events,
        3. switch-off appliances that must stop.

        Args:
            active_occupancy (ndarray(n_hh)): The active occupancy
            switchon_probs (ndarray(n_hh, n_app)): Probabilities of switch on for each household and appliances
        """

        super().step()

        switchon_probs = np.array(switchon_probs)

        # update the variables iteration
        self.n_times_left[self.n_times_left > 0] -= 1
        self.n_times_till_refresh[self.n_times_till_refresh > 0] -= 1

        # make switch on impossible if appliance not available (probs of non available are 0)
        switchon_probs *= self.get_mask_available_appliances()

        # sample with Monte Carlo which appliances must be switched on
        indexes_household, indexes_appliance = np.where(
            np.random.uniform(size=switchon_probs.shape) < switchon_probs
        )

        self.switch_on(indexes_household, indexes_appliance)

        self.switch_off_inactive(active_occupancy)

    def get_current_usage(self) -> np.ndarray:
        """Return the current used appliances in each households.

        Returns:
            usage_array, an boolean array of size size =
            (n_household, n_appliances)
            True if the appliance is currently used by the household
            else false
        """
        return self.n_times_left > 0

    def get_current_power_consumptions(self):
        """Return the power consumed by each appliances in each household.

        Returns:
            usage_array, an float array of size size =
            (n_household, n_appliances) with value being the power
            in Watts.
        """

        raise NotImplementedError()

    def get_power_demand(self):

        return np.sum(self.get_current_power_consumptions(), axis=-1)

    def get_thermal_gains(self):
        return self.get_power_demand()

    @cached_getter
    def get_energy_consumption(self):
        return np.sum(self.get_current_power_consumptions(), axis=-1)


class SubgroupApplianceSimulator(AppliancesSimulator):
    """Simulator for appliances differentiating subgroups.

    This simulators is an improvement of the original Crest Appliance
    simulation.
    Its improvements are mainly the compatibility for any subgroup
    activity profiles, the ownership of the appliance that can be
    subgroup dependant, and also the customization of the appliances
    in the dataset.

    The data loading methods determine which appliances should be
    loaded and their characteristics.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.subgroups_list`
        :py:attr:`~demod.utils.cards_doc.Params.n_households_list`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.equipped_sampling_algo`
        :py:attr:`~demod.utils.cards_doc.Params.initial_active_occupancy`
    Data
        :py:meth:`~demod.datasets.tou_loader.LoaderTOU.load_activity_probability_profiles`
        :py:meth:`~demod.datasets.base_loader.ApplianceLoader.load_appliance_ownership_dict`
        :py:meth:`~demod.datasets.base_loader.ApplianceLoader.load_appliance_dict`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.active_occupancy`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_power_demand`
        :py:meth:`~.AppliancesSimulator.get_current_usage`
        :py:meth:`~.AppliancesSimulator.get_current_power_consumptions`
        :py:meth:`~.AppliancesSimulator.get_current_water_consumptions`
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`
        :py:meth:`~demod.utils.cards_doc.Sim.get_dhw_heating_demand`
    Step size
        1 Minute.
    """

    # The activity labels
    activities_labels: List[str]
    # activities_labels indexes of the appliances
    activities_inverse: np.ndarray

    def __init__(
        self,
        subgroup_list: Subgroups,
        n_households_list: List[int],
        data: DataInput = GermanDataHerus(),
        step_size: datetime.timedelta = datetime.timedelta(minutes=1),
        initial_active_occupancy: np.ndarray = None,
        equipped_sampling_algo: str = 'subgroup',
        *args,
        **kwargs
    ):
        """Create a simulator for the appliances."""


        self.data = data

        appliances_dict = self.data.load_appliance_dict()

        self.subgroup_list = subgroup_list

        # keep in memory from which subgroup the hh are
        n_households_list = np.array(n_households_list, dtype=int)
        self.hh_types = np.concatenate(
            [i * np.ones(n, dtype=int) for i, n in enumerate(n_households_list)]
        ).reshape(-1)

        # Total number of households
        n_hh = np.sum(n_households_list)

        # checks the step size
        if step_size != datetime.timedelta(minutes=1):
            raise ValueError(
                "Step size must be 1 Minute for Appliance Simulator."
            )

        # initialize the appliances similarly for all the housholds
        super().__init__(
            n_hh, appliances_dict, *args,
            step_size=step_size,
            equipped_sampling_algo=equipped_sampling_algo,
            **kwargs
        )

        self._create_activities_labels()


        self.initialize_starting_state(
            initial_active_occupancy, *args, **kwargs
        )


    def initialize_starting_state(
        self, initial_active_occupancy, *args, **kwargs
    ):
        kwargs.pop("initialization_time", None)
        # initialization time, appliances can only start at the specified starting times
        initialization_time = self.data.refresh_time

        # step input to use during initialization, send ones if no initial value given
        if initial_active_occupancy is None:
            initial_active_occupancy = np.ones(self.n_households)

        # sets the activity pdf, as required for initialization
        self.update_subgroups_with_time()
        self.set_activity_pdf()

        return super().initialize_starting_state(
            initialization_time=initialization_time,
            active_occupancy=initial_active_occupancy,
        )

    def _create_activities_labels(self):
        # Finds the related activity index of appliances
        activities_labels, inverse = np.unique(
            self.appliances["related_activity"], return_inverse=True
        )
        self.activities_labels = activities_labels
        self.activities_inverse = inverse

    def _sample_from_subgroup(self):
        # load the ownership probs for appliances and each subgroup
        dicts_probs = [
            self.data.load_appliance_ownership_dict(subgroup)
            for subgroup in self.subgroup_list
            ]
        # Ignore the warning raised by get_ownership_from_dict
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            probs = np.asarray(
                [get_ownership_from_dict(
                    self.appliances, dic
                ) for dic in dicts_probs]
            )
        # Gets probs corresponding to each household
        available_probs = probs[self.hh_types]
        if probs.shape[-1] != self.appliances["number"]:
            raise ValueError(
                "Appliance ownership does not match the number of appliances."
            )
        rand = np.random.uniform(
            size=(self.n_households, self.appliances["number"])
        )
        return rand < available_probs

    def on_before_next_day_4am(self):
        if self.data.refresh_time == datetime.time(4, 0, 0):
            # Use increment of step size, as we update before the next day
            self.update_subgroups_with_time(optional_increment=self.step_size)
            self.set_activity_pdf()

    def on_before_next_day(self):
        if self.data.refresh_time == datetime.time(0, 0, 0):
            # Use increment of step size, as we update before the next day
            self.update_subgroups_with_time(optional_increment=self.step_size)
            self.set_activity_pdf()

    def set_activity_pdf(self):
        """Sets the activity probability profiles for this simulator.
        It loads different profiles for the different subgroups.
        """

        # laods the activity profiles
        act_profiles = [
            self.data.load_activity_probability_profiles(subgroup)
            for subgroup in self.subgroup_list
        ]
        # DIM0: subgroup, Dict[DIM1:n_times, DIM2:active_occuupancy

        # Attributes the activity profiles for each activity
        try:
            probs = np.asarray([
                [
                    # Ensures the activity are sorted as in self.activities_labels
                    act_profiles_dict[act_lab]
                    for act_lab in self.activities_labels
                    ]
                for act_profiles_dict in act_profiles
                ]
            )
        except KeyError as key_err:
            # If an activity of some appliances is not in the activity dataset
            problematic_appliances = self.appliances['name'][
                self.activities_inverse == list(self.activities_labels).index(key_err.args[0])
            ]
            raise ValueError((
                "Missing activity '{}' from activity dataset loader: {}, "
                "which is "
                "required for appliances '{}' from dataset loader: {}."
            ).format(
                key_err.args[0],
                self.data.load_activity_probability_profiles,
                problematic_appliances,
                self.data.load_appliance_dict,
            )) from key_err

        # DIM0: subgroups, DIM1: activity, DIM2: time, DIM3: active_occupancy
        # swap the subgroup and the time to make an iterator of the time
        self.activities_pdf_iterator = itertools.cycle(
            np.moveaxis(probs, 2, 0)
        )

    def update_subgroups_with_time(
        self, optional_increment=datetime.timedelta(seconds=0), *args, **kwargs
    ):
        """Updates the subgroups using the current time.

        Args:
            optional_increment (datetime.timedelta, optional):
                An value to increment the time, usefull if you want to update using another value than the current time. Defaults to datetime.timedelta(seconds=0).
        """
        for subgroup in self.subgroup_list:
            subgroup_add_time(
                subgroup,
                self.current_time + optional_increment,
                *args,
                **kwargs
            )

    def switch_on(self, indexes_household, indexes_appliance):
        """Switch on model for CREST. uses a simple mean value for the duration

        Args:
            indexes_household ([type]): [description]
            indexes_appliance ([type]): [description]
        """
        # first switch on
        # add a simple value that is the mean
        cycle_duration = self.appliances["mean_duration"][indexes_appliance]
        self.n_times_left[
            indexes_household, indexes_appliance
        ] = cycle_duration
        # adds the time till refresh is possible
        self.n_times_till_refresh[indexes_household, indexes_appliance] = (
            cycle_duration
            + self.appliances["after_cycle_delay"][indexes_appliance]
        )

        # Special switchon for the water fixtures
        mask_water_fixture = self.appliances["uses_water"][indexes_appliance]
        self._switch_on_water_fixtures(
            indexes_household[mask_water_fixture],
            indexes_appliance[mask_water_fixture],
        )

    def _switch_on_water_fixtures(self, indexes_household, indexes_appliance):

        # get the volumes that will be consumed
        lam = self.appliances["poisson_sampling_lambda_liters"][
            indexes_appliance
        ]
        volumes = np.random.poisson(lam)
        # compute the duration from the mean usage
        durations = (
            volumes
            / self.appliances["mean_water_consumption"][indexes_appliance]
        )
        # water fixtures duration
        self.n_times_left[indexes_household, indexes_appliance] = durations
        # adds the time till refresh is possible
        self.n_times_till_refresh[indexes_household, indexes_appliance] = (
            durations + self.appliances["after_cycle_delay"][indexes_appliance]
        )

    def switch_off_inactive(self, active_occupancy):
        # switch off appliances that need switch off when there is no occupancy

        # first get the mask of the ones that are switched off
        mask_app_need_switchoff = self.appliances["inactive_switch_off"]
        mask_household_need_switchoff = active_occupancy == 0

        mask_need_switchoff = (
            mask_household_need_switchoff[:, None]
            * mask_app_need_switchoff[None, :]
        )

        # finish the time remaining for the ones that were being used
        self.n_times_left[mask_need_switchoff] = 0
        # start or continue a refresh period for them

        self.n_times_till_refresh[mask_need_switchoff] = np.minimum(
            self.n_times_till_refresh[mask_household_need_switchoff][
                :, mask_app_need_switchoff
            ],
            self.appliances["after_cycle_delay"][mask_app_need_switchoff],
        ).reshape(-1)

    def get_current_power_consumptions(self):
        power_consumptions = np.zeros_like(self.n_times_left, dtype=float)
        # check the appliances that are on or off and determine the values of power for on and off
        power_consumptions += (
            (self.n_times_left == 0)
            * self.appliances["standby_consumption"]
            * self.available_appliances
        )
        power_consumptions += (self.n_times_left > 0) * self.appliances[
            "mean_elec_consumption"
        ]

        # handle the special cases (washing machines)
        # get the indicies of the specials
        index = self.appliances["type"] == "washingmachine"
        mask_available = self.available_appliances[:, index].reshape(-1)
        power_consumptions[
            mask_available, index
        ] = self._compute_washing_machine_power(
            self.n_times_left[mask_available, index], "washingmachine"
        )

        index = self.appliances["type"] == "washer_dryer"
        mask_available = self.available_appliances[:, index].reshape(-1)
        power_consumptions[
            mask_available, index
        ] = self._compute_washing_machine_power(
            self.n_times_left[mask_available, index], "washer_dryer"
        )

        return power_consumptions

    @cached_getter
    def get_current_water_consumptions(self):
        """Getter for the hot water consumption of each appliance (litres/min).

        Assumes water appliances don't have standby consumptions.

        Returns:
            np.ndarray, shape=(n_households, n_appliances)
        """
        water_consumptions = np.zeros_like(self.n_times_left, dtype=float)

        # Adds the water consumption
        water_consumptions += (self.n_times_left >= 1) * self.appliances[
            "mean_water_consumption"
        ]
        # water takes into account short duration events (less than 1 min)
        water_consumptions += (
            self.n_times_left
            * ((self.n_times_left > 0) & (self.n_times_left < 1))
            * self.appliances["mean_water_consumption"]
        )


        return water_consumptions

    @cached_getter
    def get_dhw_heating_demand(self):
        """Gets the heat demand for heating the DHW required. [W/K]

        Returns the total heat required by the appliances of hot water.
        The unit is W/K, which means that the heat depend on the
        temperature difference between the

        Returns:
            [type]: [description]
        """

        # sum the hot water demand from all the fixtures
        demands = self.get_current_water_consumptions().sum(axis=1)
        # Then calculate variable thermal resistance representing hot water demand
        # conversion from litres to m^3 and from per minute to per second
        dblV_w = demands / 1000.0 / 60.0  # m^3/sec

        # to convert from m^3 per second to kg per second
        # set the density of water
        dblRho_w = 1000.0  # kg/m^3
        dblM_w = dblRho_w * dblV_w  # kg/sec

        # convert to a thermal heat transfer coefficient in W/K
        SPECIFIC_HEAT_CAPACITY_WATER = 4200.0  # J / (kg * K)
        return SPECIFIC_HEAT_CAPACITY_WATER * dblM_w  # W/K

    def _compute_washing_machine_power(self, n_times_left, name):
        if name == "washingmachine":
            current_time = 138 - n_times_left
        elif name == "washer_dryer":
            current_time = 198 - n_times_left
        else:
            raise ValueError(name + " is not a valid name")

        power = np.zeros_like(n_times_left, dtype=float)
        # define the values of power depending on n_times left, form CREST
        power[current_time <= 8] = 73  # Start-up and fill
        power[
            np.logical_and(current_time > 8, current_time <= 29)
        ] = 2056  # Heating
        power[
            np.logical_and(current_time > 29, current_time <= 81)
        ] = 73  # Wash and drain
        power[
            np.logical_and(current_time > 81, current_time <= 92)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 92, current_time <= 94)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 94, current_time <= 105)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 105, current_time <= 107)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 107, current_time <= 118)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 118, current_time <= 120)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 120, current_time <= 131)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 131, current_time <= 133)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 133, current_time <= 138)
        ] = 568  # Fast spin
        power[
            np.logical_and(current_time > 138, current_time <= 198)
        ] = 2500  # Drying cycle

        # standby
        power[n_times_left == 0] = self.appliances["standby_consumption"][
            self.appliances["type"] == name
        ]

        return power

    def get_switchon_probs(self, active_occupancy):
        """Get the switchon probabilities for each appliance.

        Args:
            active_occupancy: array of active occupancy in each household.

        Returns:
            switchon_probs: array of size (n_households, n_appliances)
        """
        # get the activities probs for each households

        # self.current_activity_pdf
        # DIM0: subgroups, DIM1: activity, DIM2: active_occupancy
        hh_activity_probs = self.current_activity_pdf[
            self.hh_types, :, active_occupancy
        ]
        # hh_activity_probs
        # DIM0: households, DIM1: activity

        # set up the array for the switch on probabilities
        out = np.ones((self.n_households, self.appliances["number"]))

        # multiply by the switch on probabilities of each appliances
        out *= self.appliances["switch_on_prob_crest"]

        # Multiply by the probability related to activity
        out *= hh_activity_probs[:, self.activities_inverse]

        # return switchon probs
        return out

    @Callbacks.before_next_day
    @Callbacks.before_next_day_4am
    def step(self, active_occupancy):
        # check active_occupancy
        active_occupancy_ = np.array(active_occupancy, dtype=int)
        assert (
            len(active_occupancy_) == self.n_households
        ), "active occupnacy length must match the number of households"

        # keeps track of the iterations to update the activities pdf
        if (self.current_time.minute % 10) == 0:
            # update the activites pdf every 10 min
            self.current_activity_pdf = next(self.activities_pdf_iterator)
            # DIM0: subgroups, DIM1: activity, DIM2: active_occupancy

        super().step(
            active_occupancy_, self.get_switchon_probs(active_occupancy_)
        )
