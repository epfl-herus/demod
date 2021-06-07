"""This module groups various simulator for the appliances in households.

The simulators are mainly differentiated by the way they simulate
activation of the different appliances.
- Occupancy based (uses activity profiles)
- Activity based (uses directly the activity)
- Activity probabilistic (uses the activity + a probability of use)

They all inherit from an abstract simulator, that proposes a framework
to handle the appliance simulation.
"""

import datetime

from typing import Dict, List
import warnings
import numpy as np
import itertools

from .base_simulators import Callbacks, TimeAwareSimulator, cached_getter
from ..utils.subgroup_handling import add_time
from ..utils.error_messages import UNIMPLEMENTED_ALGO_IN_METHOD
from ..utils.data_types import DataInput
from ..utils.sim_types import AppliancesDict, Subgroups
from ..utils.appliances import get_ownership_from_dict
from ..datasets.base_loader import DatasetLoader
from ..datasets.Germany.loader import GermanDataHerus
from demod.utils import appliances

# Activities that imply the appliance to be used always
ALWAYS_ON_ACTIVITES = ['level', 'constant']

class AppliancesSimulator(TimeAwareSimulator):
    """Class for Simulating appliances usage.

    This class is abstract but provides an interface for simulating
    step based appliances.

    Step based appliances have a number of steps that they are
    activated before they stop.

    Some appliances can have a load pattern based on measured data.

    Attributes:
        n_times_left: An array of size = (n_households, n_appliances)
            that stores the duration of the current use.
            When n_times_left == 0, the appliance is switched OFF.
        available_appliances: An array of *boolean* of size =
            (n_households, n_appliances), where the element [i, j] is
            True if the i-est households owns the j-est appliance else
            False.
        appliances: An
            :py:class:`~demod.utils.cards_doc.Params.appliances_dict`
            containing the information for each
            of the n_appliances. Each key is a different appliance.
        load_duration: An array of size = (n_households, n_appliances)
            where values correspond to how long the corresponding load
            pattern lasts.
    """

    n_times_left: np.ndarray
    available_appliances: np.ndarray
    appliances: AppliancesDict
    load_duration: np.ndarray
    data: DatasetLoader

    def __init__(
        self,
        n_households: int,
        appliances_dict: AppliancesDict,
        equipped_sampling_algo: str = "basic",
        real_profiles_algo: str = "uniform",
        **kwargs
    ):
        """Initialize an appliance simulator with its appliances."""
        super().__init__(n_households, **kwargs)
        self.appliances = appliances_dict

        # give appliances to the households
        self.available_appliances = self.sample_available_appliances(
            equipped_sampling_algo=equipped_sampling_algo
        )

        self.sample_real_load_profiles(
            real_profiles_algo=real_profiles_algo
        )

    def initialize_starting_state(self, *args, **kwargs):
        """Initialize the starting states of the appliances.

        Calls as well the parents methods for initialization, passing
        args and kwargs.
        """
        # stores the number of times before the appliance stops being used
        self.n_times_left = np.zeros_like(
            self.available_appliances, dtype=float
        )
        self._initialize_real_loads()
        # stores the number of time till the appliances can be used
        # again after refresh period
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
            :py:attr:`available_appliances`, a boolean array of size size =
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
            # improve : would be better when reading a data set to think
            # about the correlations between the appliances possession
            # Ex. you own a dryer only if you have a wasing machine
            # a tv box only with a tv
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

    def sample_real_load_profiles(
        self, real_profiles_algo: str = 'uniform',
    ) -> np.ndarray:
        """Assign a load profile for each appliance.

        Uses :py:method`load_real_profiles` to get real profiles for the
        appliances.


        Samples the load profiles of each appliance based on the
        real_profiles_algo value.
        Note that the appliance types that do not have real load profiles
        will have constant loads.
        Splits the appliances related activity if the related activity
        is occuring the whole day (ex. 'level' activity for fridges).
        These appliances will run for the full days, repeating their profiles.

        Arguments:
            real_profiles_algo: the method used to sample the real profiles
                currently implemented
                - 'nothing', does not use the real load profiles
                - 'only_always_on', samples just the profiles of appliances
                    that are always activated (ex. fridges and freezers)
                - 'uniform', sample uniformly based on
                    the :py:attr:`~demod.utils.cards_doc.Params.appliance_type`
        """
        # Will remember which appliances use variable load profiles
        self.appliances['use_variable_loads'] = np.zeros_like(
            self.appliances['type'], dtype=bool
        )

        # Assign some variables specific to load profiles
        self.corresponding_load_id = np.zeros_like(
            self.available_appliances, dtype=int
        )
        self.load_duration = np.zeros_like(
            self.available_appliances, dtype=int
        )
        self.loads_dict = {}


        if real_profiles_algo == 'nothing':
            return

        self._sample_always_on_load_profiles()

        if real_profiles_algo == 'only_always_on':
            return

        if real_profiles_algo != 'uniform':
            raise ValueError(UNIMPLEMENTED_ALGO_IN_METHOD.format(
                real_profiles_algo,
                type(self).__name__ + '.sample_real_load_profiles'
            ) + "Only 'uniform' is implemented at the moment.")

        # Loads from the dataset
        real_loads_app_dict = self.data.load_real_profiles_dict('switchedON')
        # real_loads is a dictionarry of the load profiles:
        # {app_type: {app_name: array}}
        # For each type, different appliances
        hh_id, app_id = np.where(self.available_appliances)
        # Store the load id, duration of loads

        # Tracks which appliances will be using variable loads
        mask_variable_loads_activable = (
            np.isin(self.appliances['type'], list(real_loads_app_dict.keys()))
            & (~np.isin(
                self.appliances['related_activity'], ALWAYS_ON_ACTIVITES
            ))
        )
        self.appliances['use_variable_loads'][
            mask_variable_loads_activable
        ] = True
        # Check all types of appliances
        not_found_types = []

        for app_type in np.unique(
            self.appliances['type'][mask_variable_loads_activable]
        ):
            if app_type not in real_loads_app_dict:
                not_found_types.append(app_type)
                continue  # Go to the next type
            print('normal ', app_type)
            # Finds the appliances of this type
            mask_this_type = self.appliances['type'][app_id] == app_type
            # Sample which load is assigned to each appliance
            sampled_app_id = np.random.randint(
                0, len(real_loads_app_dict[app_type]),
                size=np.sum(mask_this_type)
            )
            # Assign the sampled ids
            self.corresponding_load_id[
                hh_id[mask_this_type], app_id[mask_this_type]
            ] = sampled_app_id

            load_durations = np.array([
                len(load) for load  # Finds the duration of each load
                in real_loads_app_dict[app_type].values()
            ], dtype=int)
            # Assign the duration of each appliance load
            self.load_duration[
                hh_id[mask_this_type], app_id[mask_this_type]
            ] = load_durations[sampled_app_id]
            # Store the loads available of each app type
            self.loads_dict[app_type] = np.zeros((
                len(real_loads_app_dict[app_type]),  # number of diff. loads
                # Take the longest load as they have different lengths
                max(load_durations)
            ))
            for i, load in enumerate(real_loads_app_dict[app_type].values()):
                self.loads_dict[app_type][i,:len(load)] = load
        if len(not_found_types):
            warnings.warn((  # When the type is not in the real loads
                "Appliance types {} are not present in real_loads_app_dict"
                " from dataset '{}'"
            ).format(not_found_types, self.data))

    def _sample_always_on_load_profiles(self):
        """The load profiles are always on."""

        # Loads from the dataset
        real_loads_app_dict = self.data.load_real_profiles_dict('full')
        # real_loads is a dictionarry of the load profiles:
        # {app_type: {app_name: array}}
        # For each type, different appliances
        hh_id, app_id = np.where(self.available_appliances)
        # Store the load id, duration of loads
        # Tracks which appliances will be using variable loads
        mask_variable_always_on_app = (
            np.isin(self.appliances['type'], list(real_loads_app_dict.keys()))
            & np.isin(self.appliances['related_activity'], ALWAYS_ON_ACTIVITES)
        )
        self.appliances['use_variable_loads'][
            mask_variable_always_on_app
        ] = True
        # Check all types of appliances
        not_found_types = []
        for app_type in np.unique(
            self.appliances['type'][mask_variable_always_on_app]
        ):
            if app_type not in real_loads_app_dict:
                not_found_types.append(app_type)
                continue  # Go to the next type
            print('always on ', app_type)
            # Finds the appliances of this type
            mask_this_type = self.appliances['type'][app_id] == app_type
            # Sample which load is assigned to each appliance
            sampled_app_id = np.random.randint(
                0, len(real_loads_app_dict[app_type]),
                size=np.sum(mask_this_type)
            )
            # Assign the sampled ids
            self.corresponding_load_id[
                hh_id[mask_this_type], app_id[mask_this_type]
            ] = sampled_app_id

            load_durations = np.array([
                len(load) for load  # Finds the duration of each load
                in real_loads_app_dict[app_type].values()
            ], dtype=int)
            # Assign the duration of each appliance load
            self.load_duration[
                hh_id[mask_this_type], app_id[mask_this_type]
            ] = load_durations[sampled_app_id]
            # Store the loads available of each app type
            self.loads_dict[app_type] = np.zeros((
                len(real_loads_app_dict[app_type]),  # number of diff. loads
                # Take the longest load as they have different lengths
                max(load_durations)
            ))
            for i, load in enumerate(real_loads_app_dict[app_type].values()):
                self.loads_dict[app_type][i, :len(load)] = load

        if len(not_found_types):
            warnings.warn((  # When the type is not in the real loads
                "Appliance types {} are not present in real_loads_app_dict"
                " from dataset '{}'"
            ).format(not_found_types, self.data))


    def _initialize_real_loads(self):
        """Init the appliances that use real loads."""
        mask_always_on = np.isin(
            self.appliances['related_activity'], ALWAYS_ON_ACTIVITES
        )[np.newaxis, :]

        mask_real_loads_on = (
            (self.load_duration > 0) & (
                self.get_current_usage() | mask_always_on
        ))
        # Sets the initial duration randomly sampled in the profiles
        self.n_times_left[mask_real_loads_on] = np.random.randint(
            0, self.load_duration[mask_real_loads_on]
        )

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

        Depending on the type of the appliances switch-on events
        are differents.
        In particular we distinguish:
            1. constant load (appliances consumes the same all the time)
            2. load pattern based (appliances has a specific load profile)
            3. varying duration (the use duration varies stochastically)


        Args:
            indexes_household: the index of the household.
            indexes_appliance: the index of the appliances, matching with the households

        Raises:
            NotImplementedError: [description]
        """
        # Finds which of the inputs are variable loads
        mask_variable_loads = np.isin(
            indexes_appliance,
            np.where(self.appliances['use_variable_loads'])[0],
        )

        self._switch_on_variable_loads(
            indexes_household[mask_variable_loads],
            indexes_appliance[mask_variable_loads]
        )

        self._switch_on_constant_loads(
            indexes_household[~mask_variable_loads],
            indexes_appliance[~mask_variable_loads]
        )


    def switch_off_inactive(self, active_occupancy):

        raise NotImplementedError()

    def step(self):
        """Step function for the appliances.

        It simply updates the appliances times left, and call the
        parent method to update steps variables.

        Implementation in child should

        1. controls the appliances usage,
        2. randomly samples switchon-events,
        3. switch-off appliances that must stop.
        """

        self._update_iteration_variables()
        super().step()

    def _update_iteration_variables(self):
        """Updates the incremental variables."""
        # update the variables iteration
        self.n_times_left[self.n_times_left > 0] -= 1
        self.n_times_till_refresh[self.n_times_till_refresh > 0] -= 1

    def get_current_usage(self) -> np.ndarray:
        """Return the current used appliances in each households.

        Returns:
            usage_array, an boolean array of size size =
            (n_household, n_appliances)
            True if the appliance is currently used by the household
            else false
        """
        return (self.n_times_left > 0) & self.available_appliances

    def get_current_power_consumptions(self):
        """Return the power consumed by each appliances in each household.

        Appliance is either in stand-by, either consuming.

        Returns:
            usage_array, an float array of size size =
            (n_household, n_appliances) with value being the power
            in Watts.
        """
        power_consumptions = np.zeros_like(self._is_used, dtype=float)

        # Standby consumption
        power_consumptions += (
            (~self.get_current_usage())
            * self.appliances["standby_consumption"]
            * self.available_appliances
        )
        # Constant consumption
        power_consumptions += (
            self.get_current_usage()
            * self.appliances["mean_elec_consumption"]
            * (~self.appliances['use_variable_loads'][np.newaxis, :])
        )
        # Variable load patterns
        power_consumptions += self._get_variable_loads_consumptions()

        return power_consumptions

    def _switch_on_variable_loads(self, indexes_household, indexes_appliance):
        """Switch on loads based on a real load profile.

        Only the indexes of appliances of variable load type should
        be given as inpute.
        """
        # Find the duration of the cycle
        cycle_duration = self.load_duration[
            indexes_household, indexes_appliance
        ]
        # Start the corresponding appliances
        self.n_times_left[
            indexes_household, indexes_appliance
        ] = cycle_duration

        self.n_times_till_refresh[indexes_household, indexes_appliance] = (
            cycle_duration
            + self.appliances["after_cycle_delay"][indexes_appliance]
        )

    def _switch_on_constant_loads(self, indexes_household, indexes_appliance):
        """Switch on load based on constant consumption."""
        # First sample for how long they will stay on
        cycle_duration = self._sample_durations(
            indexes_household, indexes_appliance
        )
        self.n_times_left[
            indexes_household, indexes_appliance
        ] = cycle_duration
        # adds the time till refresh is possible
        self.n_times_till_refresh[indexes_household, indexes_appliance] = (
            cycle_duration
            + self.appliances["after_cycle_delay"][indexes_appliance]
        )

    def _sample_durations(self, indexes_household, indexes_appliance):
        return self.appliances["mean_duration"][indexes_appliance]

    def _get_variable_loads_consumptions(self) -> np.ndarray:
        """Get the consumption of the variable loads appliances."""

        # Gets the appliances that have a variable load and are used
        mask_variable_load_used = (
            self.get_current_usage()
            & self.appliances['use_variable_loads'][np.newaxis, :]
        )

        hh_id, app_id = np.where(mask_variable_load_used)

        # Find at which step of the pattern the appliances are
        time_in_pattern = np.array(
            self.load_duration - self.n_times_left, dtype=int
        )

        # Will store the current load
        current_load = np.zeros_like(self.n_times_left, dtype=float)

        for app_type in np.unique(self.appliances['type'][app_id]):
            apps_id_this_type = np.where(
                self.appliances['type'] == app_type
            )[0]
            mask_this_type = np.isin(app_id, apps_id_this_type)

            current_loads_this_type = self.loads_dict[app_type][
                # Access the load values id of these types of appliances
                self.corresponding_load_id[
                    hh_id[mask_this_type], app_id[mask_this_type]
                ],
                # Finds the time in the load of the appliances
                time_in_pattern[hh_id[mask_this_type], app_id[mask_this_type]]
            ]
            # Set the consumption
            current_load[
                hh_id[mask_this_type], app_id[mask_this_type]
            ] = current_loads_this_type

        return current_load

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
            add_time(
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


        switchon_probs = self.get_switchon_probs(active_occupancy_)
        # make switch on impossible
        # if appliance not available (probs of non available are 0)
        switchon_probs *= self.get_mask_available_appliances()

        # sample with Monte Carlo which appliances must be switched on
        indexes_household, indexes_appliance = np.where(
            np.random.uniform(size=switchon_probs.shape) < switchon_probs
        )

        self.switch_on(indexes_household, indexes_appliance)

        self.switch_off_inactive(active_occupancy)

        super().step()  # Updates the variables


class ActivityApplianceSimulator(AppliancesSimulator):
    """Appliance simulator based on residents activities.

    Turns the appliances on and keeps them activated while residents
    are performing corresponding activites.
    The appliances are always ON while the activity is performed in
    the household.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.subgroups_list`
        :py:attr:`~demod.utils.cards_doc.Params.n_households_list`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.equipped_sampling_algo`
        :py:attr:`~demod.utils.cards_doc.Params.initial_activities_dict`
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
        From the dataset.
    """

    _is_used: np.ndarray  # Boolean array that records which app is used

    def __init__(
        self, n_households, initial_activities_dict,
        data=GermanDataHerus(),
        **kwargs
    ) -> None:
        """Create the simulator.

        Sample the appliances, but only the ones that are related to the
        activity.
        """
        appliance_dict = data.load_appliance_dict()
        self.data = data

        super().__init__(
            n_households,
            self._remove_non_compatible_appliances(appliance_dict),
            **kwargs
        )


        self.initialize_starting_state(initial_activities_dict, **kwargs)

    def _remove_non_compatible_appliances(
        self, appliance_dict: AppliancesDict
    ) -> AppliancesDict:
        """Remove appliances that cannot be simulated by this."""
        if 'probabilistic' not in appliance_dict:
            # Needs probabilitic key to remove these appliances
            return appliance_dict
        activity_appliance_dict = {}
        mask_app_for_this_sim = (
            appliance_dict['probabilistic'] == False
        )
        for key, items in appliance_dict.items():
            if key == 'number':
                # Number is not an array, account for removed act
                activity_appliance_dict[key] = int(sum(mask_app_for_this_sim))
                continue
            activity_appliance_dict[key] = items[mask_app_for_this_sim]

        return activity_appliance_dict

    def initialize_starting_state(self, initial_activities_dict, **kwargs):
        """Initialize the appliances states based on activity."""
        # Records which appliance is used
        self._is_used = np.zeros_like(
            self.available_appliances, dtype=bool
        )
        super().initialize_starting_state(self, **kwargs)
        self.previous_act_dict = initial_activities_dict.copy()

        # Records the position in the pattern of this appliance
        for act, n_performing in initial_activities_dict.items():
            # Gets indices of the appliances of act and the performing act
            app_ind = np.where(self.appliances['related_activity'] == act)[0]
            hh_ind = np.where(n_performing > 0)[0]

            self.switch_on(
                # Start all appliances corresponding to act in all hhs
                np.tile(app_ind, len(hh_ind)),
                np.repeat(hh_ind, len(app_ind))
            )

        # Turn on level appliances as they are always used
        self._is_used[:, np.isin(
            self.appliances['related_activity'], ALWAYS_ON_ACTIVITES
        )] = True


    def step(self, activities_dict: Dict[str, np.ndarray]) -> None:
        """Perform a step of simulation.

        Check the households where someone started an activity, or when
        an activity stopped.
        Updates the appliances in consequence.

        TODO: check how we want to implement secondary appliances.


        """

        for act, n_performing in activities_dict.items():
            # Get the appliances related to that activity
            mask_apps_this_act = self.appliances['related_activity'] == act
            if not np.any(mask_apps_this_act):
                # If no appliance is concerned by this activity
                continue

            # Gets the number that performed it at last step
            previous_performing = self.previous_act_dict[act]
            # Gets the ones that should change
            mask_change = previous_performing != n_performing

            # Switch off
            mask_hh_act_stops = mask_change & (n_performing == 0)
            mask_app_stop = (
                mask_apps_this_act & self.appliances['inactive_switch_off']
            )
            if np.any(mask_hh_act_stops):
                # turn off all appliances of this activity in stops hh
                self.switch_off(*np.where(
                    mask_hh_act_stops[:, np.newaxis]
                    & mask_app_stop[np.newaxis, :]
                ))

            # Switch on
            mask_hh_act_start = mask_change & (previous_performing == 0)
            if np.any(mask_hh_act_start):
                self.switch_on(*np.where(
                    mask_hh_act_start[:, np.newaxis]
                    & mask_apps_this_act[np.newaxis, :]
                ))

        self.previous_act_dict = activities_dict.copy()

        super().step()

    def _update_iteration_variables(self):
        # Load patterns should be reloaded as activity continues
        mask_used_before = self.n_times_left > 0
        super()._update_iteration_variables()
        mask_reload = (
            # Used before but not after update
            (mask_used_before & ~(self.n_times_left > 0))
            & self.appliances['use_variable_loads'][np.newaxis, :],
        )

        durations = self.load_duration[mask_reload]
        # Reload the times left
        self.n_times_left[mask_reload] += durations
        self.n_times_till_refresh[mask_reload] += durations


    def _switch_on_constant_loads(self, indexes_household, indexes_appliance):
        # Don't know for how long they will stay on
        self._is_used[indexes_household, indexes_appliance] = True

    def _switch_on_variable_loads(self, indexes_household, indexes_appliance):
        super()._switch_on_variable_loads(indexes_household, indexes_appliance)
        self._is_used[indexes_household, indexes_appliance] = True


    def switch_off(self, indexes_household, indexes_appliance):
        # Don't know for how long they will stay on
        self._is_used[indexes_household, indexes_appliance] = False


    def get_current_usage(self):
        # Usage is determined only by activity
        return self._is_used & self.available_appliances


class ProbabiliticActivityAppliancesSimulator(AppliancesSimulator):
    """Appliance simulator based on residents activities.

    Similar to :py:class:`.ActivityApplianceSimulator` but adds
    a probability that appliances are switched on or off, based on
    the :py:obj:`target_cycle_year` of the
    :py:attr:`~demod.utils.cards_doc.Params.appliances_dict`.
    Turns the appliances during the time its
    corresponding activity is performed, or during the duration
    sampled.

    .. note::
        This methods allows only for a single use per day of the probabilistic
        appliances.

    Params
        :py:attr:`~demod.utils.cards_doc.Params.subgroups_list`
        :py:attr:`~demod.utils.cards_doc.Params.n_households_list`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.equipped_sampling_algo`
        :py:attr:`~demod.utils.cards_doc.Params.initial_activities_dict`
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
        From the dataset.
    """

    def __init__(self, *args, **kwargs):
        ActivityApplianceSimulator.__init__(self, *args, **kwargs)
        # Initalize the target used per week
        self.target_weekly_cycle = self._assign_target_weekly_cycles()

    def _assign_target_weekly_cycles(self):
        # Should use subgroups instead of this to improve, as in Bottaccioli
        app_targets = self.appliances['target_cycle_year']
        return np.broadcast_to(
            app_targets, (self.n_households, len(app_targets))
        )

    def _remove_non_compatible_appliances(
        self, appliance_dict: AppliancesDict
    ) -> AppliancesDict:
        if 'probabilistic' not in appliance_dict:
            # Needs probabilitic key to remove these appliances
            raise ValueError(
                "'probabilistic' is not in appliance_dict from "
                " {}.".format(self.data)
            )
        activity_appliance_dict = {}
        mask_app_for_this_sim = appliance_dict['probabilistic']
        for key, items in appliance_dict.items():
            if key == 'number':
                # Number is not an array, account for removed act
                activity_appliance_dict[key] = int(sum(mask_app_for_this_sim))
                continue
            activity_appliance_dict[key] = items[mask_app_for_this_sim]

        return activity_appliance_dict

    def initialize_starting_state(self, initial_activities_dict):
        self.previous_act_dict = initial_activities_dict.copy()

        # Assumes probabilistic appliances are off at start
        self.n_times_left = np.zeros_like(
            self.available_appliances, dtype=float
        )

        # Initialize the number of times left that the appliances should be used
        start_weekday = self.current_time.weekday()
        self.occured_cycles = np.zeros_like(self.n_times_left, dtype=int)
        for days_passed in range(start_weekday):
            # Simulates as if the days where already passed
            start_mask = self._sample_if_occuring_today(7 - days_passed)
            self.occured_cycles += start_mask

        self.today_cycles = self._sample_if_occuring_today(7 - start_weekday)

        super().initialize_starting_state(self)

    def _sample_if_occuring_today(self, days_left: int):
        # Correction of Bottaccioli formulation, uses the already occuring
        # number of cycles
        probs = (self.target_weekly_cycle - self.occured_cycles) / days_left
        return np.random.uniform(size=self.n_times_left.shape) < probs

    def on_before_next_day(self) -> None:
        """Samples the probability that the probabilistic appliance is used."""
        weekday = self.current_time.weekday()
        self.today_cycles = self._sample_if_occuring_today(7 - weekday)


    @ Callbacks.before_next_day
    def step(self, activities_dict: Dict[str, np.ndarray]) -> None:
        """Step function for the appliances.

        It
        1. controls the appliances usage,
        2. keeps track of how long the appliances are being ran,
        3. switch-off appliances that must stop.
        4. samples switch on events of activity related appliances
        5. samples switch on events of after-activity related appliances
            ex. dishwasher
        6. samples switch on events of after-specific appliance related appliances
            ex. dryer

        Args:
            active_occupancy (ndarray(n_hh)): The active occupancy
            switchon_probs (ndarray(n_hh, n_app)): Probabilities of switch on for each household and appliances
        """
        # update the variables iteration
        mask_decrement_time = self.n_times_left > 0
        self.n_times_left[mask_decrement_time] -= 1
        self.n_times_till_refresh[self.n_times_till_refresh > 0] -= 1

        turned_off_hh, turned_off_app = np.where(
            mask_decrement_time & ~(self.n_times_left > 0)
        )
        # Start the appliances that start after another appliance
        self.switch_on(*self._get_start_after_appliance(
            turned_off_hh, turned_off_app
        ))

        for act, n_performing in activities_dict.items():
            # Get the appliances related to that activity
            mask_apps_this_act = self.appliances['related_activity'] == act
            if not np.any(mask_apps_this_act):
                # If no appliance is concerned by this activity
                continue

            # Gets the number that performed it at last step
            previous_performing = self.previous_act_dict[act]
            # Gets the ones that should change
            mask_change = previous_performing != n_performing

            # TODO check if the appliance start today
            # TODO how do we implement shiftable appliances

            # Switch off
            mask_hh_act_stops = mask_change & (n_performing == 0)
            mask_app_stop = (
                mask_apps_this_act & self.appliances['inactive_switch_off']
            )
            if np.any(mask_hh_act_stops):
                self.switch_off(*np.where(  # turn off all appliances of this act and a
                    mask_hh_act_stops[:, np.newaxis]
                    & mask_app_stop[np.newaxis, :]
                ))

            # Start the appliances that start after another activity ends
            self.switch_on(*self._get_start_after_activity(
                mask_hh_act_stops, act
            ))

            # Finds which appliances can switch on for this activity
            mask_hh_act_occurs = n_performing > 0
            can_start_hh, can_start_app = np.where(
                mask_hh_act_occurs[:, np.newaxis]
                & mask_apps_this_act[np.newaxis, :]
            )

            # Samples the appliances that should start
            switchon_probs = self._compute_switch_on_probs(
                can_start_hh, can_start_app
            )
            mask_start = (
                np.random.uniform(size=len(switchon_probs)) < switchon_probs
            )

            if np.any(mask_start):
                self.switch_on(
                    can_start_hh[mask_start],
                    can_start_app[mask_start]
                )

        self.previous_act_dict = activities_dict.copy()

        super().step()

    def _compute_switch_on_probs(self, indexes_household, indexes_appliance):
        # Don't know for how long they will stay on
        # TODO find a good solution for that
        return (
            (~self._is_used[indexes_household, indexes_appliance])
            * self.appliances['switch_on_prob_crest'][indexes_appliance]
            * self.today_cycles[indexes_household, indexes_appliance]
        )

    def _compute_switch_on_probs_for_one_time(self, indexes_household, indexes_appliance):
        """Appliances that can be switched on only after another one or when activity stops."""
        return (
            (~self._is_used[indexes_household, indexes_appliance])
            * self.today_cycles[indexes_household, indexes_appliance]
        )

    def switch_off(self, indexes_household, indexes_appliance):
        # Don't know for how long they will stay on
        self.n_times_left[indexes_household, indexes_appliance] = 0
        self.n_times_till_refresh[indexes_household, indexes_appliance] = (
            self.appliances['refresh_time'][indexes_appliance]
        )

    def _get_start_after_activity(
        self, mask_hh_act_stops, activity
    ):
        """When the activity stops."""
        inds_app_mask = np.where(
            self.appliances['previous_activity'] == activity
        )[0]
        ind_hh = np.where(mask_hh_act_stops)[0]

        # Compute probability of switch on
        switch_on_probs = self._compute_switch_on_probs_for_one_time(
            ind_hh,
            inds_app_mask
        )
        # Sample the switchon probs
        mask_switch_on = (
            np.random.uniform(size=len(switch_on_probs)) < switch_on_probs
        )
        return (
            ind_hh[mask_switch_on], inds_app_mask[mask_switch_on]
        )


    def _get_start_after_appliance(
        self, indexes_household, indexes_appliance_stopped
    ):
        """Return the indexes of hh and app that should start.

        They start after other appliances that just stopped.
        To know which start, it is based on
        self.appliances['previous_appliance_type']
        """
        # Must switchon the appliances that follow a specific switchoff
        new_app_id = np.where(  # ID where the appliance follows a previous one
            ~np.isnan(self.appliances['previous_appliance_type'])
        )
        indexes_app_start = np.array([], dtype=int)
        indexes_hh_start = np.array([], dtype=int)
        # Iterates over the appliances with a previous type
        for app_id in new_app_id:
            mask_stopped = (
                # Finds where the appliance stopped is of this previous type
                self.appliances['previous_appliance_type'][app_id]
                == self.appliances['type'][indexes_appliance_stopped]
            )
            # Adds the appliance that should start (follow a stop)
            indexes_app_start = np.append(
                indexes_app_start, app_id * np.ones(sum(mask_stopped))
            )
            indexes_hh_start = np.append(
                indexes_hh_start, indexes_household[mask_stopped]
            )
        # Compute probability of switch on
        switch_on_probs = self._compute_switch_on_probs_for_one_time(
            indexes_hh_start, indexes_app_start
        )
        # Sample the switchon probs
        mask_switch_on = (
            np.random.uniform(size=len(switch_on_probs)) < switch_on_probs
        )
        return (
            indexes_hh_start[mask_switch_on], indexes_app_start[mask_switch_on]
        )

    def switch_on_with_shift(self, indexes_household, indexes_appliance):
        """Switch-on appliances using a shift in time.

        Basically waits for the number of steps before turning on the
        appliance.
        Useful for delaying appliance use to a specific hour,
        or to model a waiting time, between two correlated appliance use
        (ex. washing machine and dryer)
        """
        raise NotImplementedError()