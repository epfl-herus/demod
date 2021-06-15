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
from ..utils.error_messages import ALGO_REQUIRES_LOADING_METHOD, UNIMPLEMENTED_ALGO_IN_METHOD, USE_OTHER_ALGOS_FOR_ALGONAME
from ..utils.data_types import DataInput
from ..utils.sim_types import ActivitiesDict, AppliancesDict, Subgroups
from ..utils.appliances import get_ownership_from_dict
from ..utils import appliances
from ..datasets.base_loader import DatasetLoader
from ..datasets.Germany.loader import GermanDataHerus

# Activities that imply the appliance to be used always
ALWAYS_ON_ACTIVITES = ['level', 'constant']


class AppliancesSimulator(TimeAwareSimulator):
    """Class for Simulating appliances usage.

    This class is abstract but provides an interface for simulating
    step based appliances.

    Appliances informations are stored in :py:obj:`self.appliance`
    which is the appliance_dict.
    The states an information of all appliances are stored as
    different numpy arrays of shape = (n_households, n_appliances).
    The attributes below show more in dept which information are
    stored.

    A state based appliance works this way:
        1. It is switched ON by the simulator at step $n$.
        2. The switch-ON event samples for how long the appliance
            will be ON, ($m$ steps)
        3. The appliance is switched-OFF at step $n+m$

    The consumption load of appliances can be either constant or
    either variable.
        * constant: a single value of consumption is assigned
        * variable: the consumption changes each step based on
            a real load profile from
            :py:method`~demod.datasets.base_loader.ApplianceLoader.load_real_profiles`
        * random: FOR FUTURE IMPLEMENTATION. The load fluctuates
            over a mean value, or using a random distribution.

    Step based appliances have a number of steps that they are
    activated before they stop. The number of steps can be
        * constant: using a mean duration
        * constant: in case of a real load profile, it is its duration
        * random: follow a distribution
        * infinite: some appliances that are always ON
        * activity related: FOR FUTURE IMPLEMENTATION. The duration is
            sample based on durations obtained for TOU statistics.

    Appliances also have a stand-by consumption for when they are OFF.

    The consumption of appliances can be electricity, DHW (domestic
    hot water), or gaz.

    Attributes:
        n_steps_left: An array of size = (n_households, n_appliances)
            that stores the duration of the current use.
            When n_steps_left == 0, the appliance is switched OFF.
        n_steps_till_refresh: An array of size =
            (n_households, n_appliances).
            Tracks how many steps are left before the appliance
            can be switched ON again.
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

    n_steps_left: np.ndarray
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
        """Initialize an appliance simulator with its appliances.

        1. Simulator is initialized.
        2. Sample the appliances available in each household
            :py:meth:`sample_available_appliances`
        3. Sample the real profiles of the appliances
            :py:meth:`sample_real_load_profiles`
        """
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

        TODO: find a way to randomly sample which appl
        """
        # stores the number of times before the appliance stops being used
        self.n_steps_left = np.zeros_like(
            self.available_appliances, dtype=float
        )
        self._initialize_real_loads()
        # stores the number of time till the appliances can be used
        # again after refresh period
        # This is specially useful for the fridges and freezers cycles,
        # Such that they all start synchronized
        delays = [
            np.random.randint(0, delay, size=self.n_households)
            if delay > 0
            else np.zeros(self.n_households, dtype=int)
            for delay in self.appliances["after_cycle_delay"]
        ]
        self.n_steps_till_refresh = np.asarray(delays, dtype=int).T
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

    def _parse_subgroups(self, n_households, subgroup_list, n_households_list):
        if (subgroup_list is None) and (n_households_list is None):
            return
        if (subgroup_list is None) or (n_households_list is None):
            raise AttributeError(
                "If you are using subgroups, you need to specify both "
                "'subgroup_list' and 'n_households_list'."
            )
        if sum(n_households_list) != n_households:
            raise ValueError(
                "'n_households' must be equal to "
                "the sum of 'n_households_list' "
            )

        self.subgroup_list = subgroup_list

        n_households_list = np.array(n_households_list, dtype=int)
        # keep in memory from which subgroup the hh are
        self.hh_types = np.concatenate(
            [i * np.ones(n, dtype=int) for i, n in enumerate(n_households_list)]
        ).reshape(-1)

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


        if real_profiles_algo == 'only_always_on':
            self._sample_always_on_load_profiles()
            return

        if real_profiles_algo == 'only_switched_on':
            self._sample_switched_on_load_profiles()
            return


        if real_profiles_algo != 'uniform':
            raise ValueError(UNIMPLEMENTED_ALGO_IN_METHOD.format(
                real_profiles_algo,
                type(self).__name__ + '.sample_real_load_profiles'
            ) + "Only 'uniform' is implemented at the moment.")

        self._sample_switched_on_load_profiles()
        self._sample_always_on_load_profiles()

    def _sample_switched_on_load_profiles(self):
        # Loads from the dataset
        try:
            real_loads_app_dict = self.data.load_real_profiles_dict('switchedON')
        except NotImplementedError:
            raise NotImplementedError((
                ALGO_REQUIRES_LOADING_METHOD + '\n'
                + USE_OTHER_ALGOS_FOR_ALGONAME
            ).format(
                algo='',
                simulator=type(self).__name__,
                loading_method="load_real_profiles_dict('switchedON')",
                dataset=self.data,
                other_algos=['only_always_on', 'nothing'],
                algo_name='real_profiles_algo',
            ))
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
        try:
            real_loads_app_dict = self.data.load_real_profiles_dict('full')
        except NotImplementedError:
            raise NotImplementedError((
                ALGO_REQUIRES_LOADING_METHOD + '\n'
                + USE_OTHER_ALGOS_FOR_ALGONAME
            ).format(
                algo='',
                simulator=type(self).__name__,
                loading_method="load_real_profiles_dict('full')",
                dataset=self.data,
                other_algos=['only_switched_on', 'nothing'],
                algo_name='real_profiles_algo',
            ))

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
        self.n_steps_left[mask_real_loads_on] = np.random.randint(
            0, self.load_duration[mask_real_loads_on]
        )

    def get_mask_available_appliances(self):
        # available if in the house and not already used and not in refresh time
        return np.logical_and.reduce(
            (
                self.available_appliances,
                self.n_steps_left <= 0,
                self.n_steps_till_refresh <= 0,
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

        Implementation in child should control when to call switch ON
        and switch OFF events, and call this step method
        by using `super().step()`.
        """

        self._update_iteration_variables()
        super().step()

    def _update_iteration_variables(self):
        """Updates the incremental variables."""
        # update the variables iteration
        self.n_steps_left[self.n_steps_left > 0] -= 1
        self.n_steps_till_refresh[self.n_steps_till_refresh > 0] -= 1

    def get_current_usage(self) -> np.ndarray:
        """Return the current used appliances in each households.

        Returns:
            usage_array, an boolean array of size size =
            (n_household, n_appliances)
            True if the appliance is currently used by the household
            else false
        """
        return (self.n_steps_left > 0) & self.available_appliances

    def get_current_power_consumptions(self):
        """Return the power consumed by each appliances in each household.

        Appliance is either in stand-by, either consuming.

        Returns:
            usage_array, an float array of size size =
            (n_household, n_appliances) with value being the power
            in Watts.
        """
        power_consumptions = np.zeros_like(self.n_steps_left, dtype=float)

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
        be given as input.
        """
        # Find the duration of the cycle
        cycle_duration = self.load_duration[
            indexes_household, indexes_appliance
        ]
        # Start the corresponding appliances
        self.n_steps_left[
            indexes_household, indexes_appliance
        ] = cycle_duration

        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
            cycle_duration
            + self.appliances["after_cycle_delay"][indexes_appliance]
        )

    def _switch_on_constant_loads(self, indexes_household, indexes_appliance):
        """Switch on load based on constant consumption."""
        # First sample for how long they will stay on
        cycle_duration = self._sample_durations(
            indexes_household, indexes_appliance
        )
        self.n_steps_left[
            indexes_household, indexes_appliance
        ] = cycle_duration
        # adds the time till refresh is possible
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
            cycle_duration
            + self.appliances["after_cycle_delay"][indexes_appliance]
        )

    def _sample_durations(self, indexes_household, indexes_appliance):
        """Sample the duration of a load.

        TODO: sample different kinds of load differently ?
        """
        durations = self.appliances["mean_duration"][indexes_appliance]

        # The real load profiles have a duration equal to their length
        mask_real_loads = self.load_duration[indexes_household, indexes_appliance] > 0
        durations[mask_real_loads] = self.load_duration[indexes_household, indexes_appliance][mask_real_loads]

        return durations

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
            self.load_duration - self.n_steps_left, dtype=int
        )

        # Will store the current load
        current_load = np.zeros_like(self.n_steps_left, dtype=float)

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
        # TODO add the power factor
        return self.get_power_demand()

    @cached_getter
    def get_energy_consumption(self):
        return np.sum(self.get_current_power_consumptions(), axis=-1)


class SubgroupApplianceSimulator(AppliancesSimulator):
    """Simulator for appliances differentiating subgroups.

    It uses
        - the simulated household occupancy
        - the activity profiles of different subgroups
    to compute
    the switch on probability of each appliance.


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
        step_size: datetime.timedelta = None,
        initial_active_occupancy: np.ndarray = None,
        equipped_sampling_algo: str = 'subgroup',
        *args,
        **kwargs
    ):
        """Create a simulator for the appliances."""

        self.data = data

        appliances_dict = self.data.load_appliance_dict()

        # Total number of households
        n_hh = np.sum(n_households_list)

        self._parse_subgroups(n_hh, subgroup_list, n_households_list)

        # checks the step size
        if step_size is None:
            step_size = data.step_size
        if step_size != data.step_size:
            raise ValueError((
                "Step sizes should be the same in {} and in the "
                " arguments of {}."
            ).format(data, self))

        # initialize the appliances similarly for all the households
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

    def sample_switched_on_load_profiles(self):
        pass

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

    def _initialize_real_loads(self):
        """Init the appliances that use real loads."""

        mask_real_loads_on = (
            (self.load_duration > 0)
            & self.get_current_usage()
        )
        # Sets the initial duration randomly sampled in the profiles
        self.n_steps_left[mask_real_loads_on] = np.random.randint(
            0, self.load_duration[mask_real_loads_on]
        )

    def _create_activities_labels(self):
        # Finds the related activity index of appliances
        activities_labels, inverse = np.unique(
            self.appliances["related_activity"], return_inverse=True
        )
        self.activities_labels = activities_labels
        self.activities_inverse = inverse


    def on_before_refresh_time(self) -> None:
        """Update the activity pdfs for the next day."""
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
        cycle_duration = self._sample_durations(indexes_household, indexes_appliance)
        self.n_steps_left[
            indexes_household, indexes_appliance
        ] = cycle_duration
        # adds the time till refresh
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
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
        self.n_steps_left[indexes_household, indexes_appliance] = durations
        # adds the time till refresh is possible
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
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
        self.n_steps_left[mask_need_switchoff] = 0
        # start or continue a refresh period for them

        self.n_steps_till_refresh[mask_need_switchoff] = np.minimum(
            self.n_steps_till_refresh[mask_household_need_switchoff][
                :, mask_app_need_switchoff
            ],
            self.appliances["after_cycle_delay"][mask_app_need_switchoff],
        ).reshape(-1)


    @cached_getter
    def get_current_water_consumptions(self):
        """Getter for the hot water consumption of each appliance (litres/min).

        Assumes water appliances don't have standby consumptions.

        Returns:
            np.ndarray, shape=(n_households, n_appliances)
        """
        water_consumptions = np.zeros_like(self.n_steps_left, dtype=float)

        # Adds the water consumption
        water_consumptions += (self.n_steps_left >= 1) * self.appliances[
            "mean_water_consumption"
        ]
        # water takes into account short duration events (less than 1 min)
        water_consumptions += (
            self.n_steps_left
            * ((self.n_steps_left > 0) & (self.n_steps_left < 1))
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

    @Callbacks.before_refresh_time
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
    Appliances that are turned ON and OFF randomly depending on
    the activity can be simulated by
    :py:class`.ProbabiliticActivityAppliancesSimulator`

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
        self, n_households: int, initial_activities_dict,
        data=GermanDataHerus(),
        subgroup_list: Subgroups = None,
        n_households_list: List[int] = None,
        **kwargs
    ) -> None:
        """Create the simulator.

        Sample the appliances, but only the ones that are related to the
        activity.
        """
        appliance_dict = data.load_appliance_dict()
        self.data = data

        self._parse_subgroups(n_households, subgroup_list, n_households_list)

        super().__init__(
            n_households,
            self._remove_non_compatible_appliances(
                appliance_dict, initial_activities_dict
            ),
            **kwargs
        )

        self.initialize_starting_state(initial_activities_dict, **kwargs)

    def _remove_non_compatible_appliances(
        self,
        appliance_dict: AppliancesDict,
        initial_activities_dict: ActivitiesDict
    ) -> AppliancesDict:
        """Remove appliances that cannot be simulated by this."""

        # Makes sure all related activities are included
        mask_app_for_this_sim = np.isin(
            appliance_dict['related_activity'],
            list(initial_activities_dict.keys()) + ALWAYS_ON_ACTIVITES
            )
        if sum(~mask_app_for_this_sim) > 0:
            warnings.warn((
                "{} have missing related activities : {} "
                " from the ones given in the "
                " 'initial_activities_dict' with activities {}."
            ).format(
                appliance_dict['name'][~mask_app_for_this_sim],
                appliance_dict['related_activity'][~mask_app_for_this_sim],
                initial_activities_dict.keys()
            ))

        if 'probabilistic' in appliance_dict:
            # Needs probabilitic key to remove these appliances
            mask_app_for_this_sim &= (~appliance_dict['probabilistic'])

        activity_appliance_dict = {}  # Dict with only compatible apps
        for key, items in appliance_dict.items():
            if key == 'number':
                # Number is not an array, account for removed act
                activity_appliance_dict[key] = int(sum(mask_app_for_this_sim))
                continue
            # Copy all items of appliance dict to the new dict
            activity_appliance_dict[key] = items[mask_app_for_this_sim]

        return activity_appliance_dict

    def initialize_starting_state(self, initial_activities_dict, **kwargs):
        """Initialize the appliances states based on activity.

        Activities that are occuring will have their"""
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
                np.repeat(hh_ind, len(app_ind)),
                np.tile(app_ind, len(hh_ind)),
            )

        # Turn on level appliances as they are always used
        self._is_used[:, np.isin(
            self.appliances['related_activity'], ALWAYS_ON_ACTIVITES
        )] = True

    def _check_simulated_activities(self, activities_dict):
        """One time check of matching simulated activities with appliances.

        Warns if some appliances related activities are not simulated, ie
        are not present.
        """

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
        mask_used_before = self.n_steps_left > 0
        super()._update_iteration_variables()
        mask_reload = (
            # Used before but not after update
            (mask_used_before & ~(self.n_steps_left > 0))
            & self.appliances['use_variable_loads'][np.newaxis, :],
        )

        durations = self.load_duration[mask_reload]
        # Reload the times left
        self.n_steps_left[mask_reload] += durations
        self.n_steps_till_refresh[mask_reload] += durations

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

    Attributes:
        occured_cycles, An array storing how many cycles already occurred
            this week

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

    occured_cycles: np.ndarray

    def __init__(
        self, n_households, initial_activities_dict,
        data=GermanDataHerus(),
        subgroup_list: Subgroups = None,
        n_households_list: List[int] = None,
        **kwargs
    ) -> None:
        """Create the simulator.

        Sample the appliances, but only the ones that are related to the
        activity.
        """
        appliance_dict = data.load_appliance_dict()
        self.data = data

        self._parse_subgroups(n_households, subgroup_list, n_households_list)

        super().__init__(
            n_households,
            self._remove_non_compatible_appliances(
                appliance_dict, initial_activities_dict
            ),
            **kwargs
        )

        self._assign_dictated_start()

        # Assign the target used per week
        self.target_weekly_cycle = self._assign_target_weekly_cycles()
        self.switch_on_probs = self._sample_switch_on_probs()

        self.initialize_starting_state(initial_activities_dict, **kwargs)

    def _assign_dictated_start(self):
        mask_dictated_start = np.zeros(self.appliances['number'], dtype=bool)
        if 'previous_appliance_type' in self.appliances:
            mask_dictated_start[
                self.appliances['previous_appliance_type'] != 'nan'
            ] = True
        if 'previous_activity' in self.appliances:
            mask_dictated_start[
                self.appliances['previous_activity'] != 'nan'
            ] = True

        self.apps_dictated_start = np.where(mask_dictated_start)[0]

    def _assign_target_weekly_cycles(self):
        """Assign to each household and appliance a target of weekly cycles.

        This implementation only use appliance_dict['target_cycle_year']
        as target.
        It should be improved to be loaded from the dataset based on
        the subgroups in a future implementation.
        """
        # Should use subgroups instead of this to improve, as in Bottaccioli
        weekly_targets = self.appliances['target_cycle_year'] / 365.25 * 7.
        return np.broadcast_to(
            weekly_targets, (self.n_households, len(weekly_targets))
        )

    def _remove_non_compatible_appliances(
        self,
        appliance_dict: AppliancesDict,
        initial_activities_dict: ActivitiesDict
    ) -> AppliancesDict:
        # Check if probabilistic is given, else assume all appliances are it.
        if 'probabilistic' not in appliance_dict:
            warnings.warn((
                "'probabilistic' is not in appliance dict of {}. \n "
                "{} will assume they are all probabilistic."
            ).format(self.data, self))
            appliance_dict['probabilistic'] = np.ones_like(
                appliance_dict['type'], dtype=bool
            )

        # Is the opposite as activity related, so we can just change the
        # value of the probabilistic array
        appliance_dict['probabilistic'] = (~appliance_dict['probabilistic'])
        new_dic = ActivityApplianceSimulator._remove_non_compatible_appliances(
            self, appliance_dict, initial_activities_dict
        )
        new_dic['probabilistic'] = (~new_dic['probabilistic'])

        return new_dic

    def _sample_switch_on_probs(self) -> np.ndarray:
        # Sample the probability of a switch on event is occuring
        probs = np.empty_like(self.available_appliances, dtype=float)

        # For each subgroup, finds the target of consumption
        for subgroup in self.subgroup_list:
            mask_hh_subgroup = self.hh_types == subgroup
            # The switchon targets for each appliance, TODO: check other targes
            target_switchons = self.data.load_yearly_target_switchons(subgroup)
            # Total probability of occurance of this activity
            activity_prob = self.data.load_activity_probability(subgroup)
            for activity, act_prob in activity_prob.items():
                # Each appliance will have a target
                mask_this_act = self.appliances['related_activity'] == activity
                # Possible switchon steps in the target period (year)
                n_possible_steps = (
                    365.25 * 24 * 60 * 60 / self.step_size.total_seconds()
                ) * sum(act_prob)  # Only steps of this activity
                #
                probs[
                    mask_hh_subgroup[:, np.newaxis],
                    mask_this_act[np.newaxis, :]
                ] = target_switchons[mask_this_act] / n_possible_steps

        # TODO: add special probs for one time probs
        return probs

    def initialize_starting_state(self, initial_activities_dict, **kwargs):
        self.previous_act_dict = initial_activities_dict.copy()

        # Initialize the number of times left that the appliances should be used
        start_weekday = self.current_time.weekday()
        self.occured_cycles = np.zeros_like(
            self.available_appliances, dtype=int
        )
        for days_passed in range(start_weekday):
            # Simulates as if the days where already passed
            start_mask = self._sample_if_occuring_today(7 - days_passed)
            self.occured_cycles += start_mask

        self.today_cycles = self._sample_if_occuring_today(7 - start_weekday)

        super().initialize_starting_state(self, **kwargs)

    def _sample_if_occuring_today(self, days_left: int):
        # Correction of Bottaccioli formulation, uses the already occuring
        # number of cycles
        probs = (self.target_weekly_cycle - self.occured_cycles) / days_left
        return np.random.uniform(size=self.occured_cycles.shape) < probs

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
            activities_dict: A dictionary containing the different activities
                performed by the household
        """
        # update the variables iteration
        mask_decrement_time = self.n_steps_left > 0
        self.n_steps_left[mask_decrement_time] -= 1
        self.n_steps_till_refresh[self.n_steps_till_refresh > 0] -= 1

        turned_off_hh, turned_off_app = np.where(
            mask_decrement_time & ~(self.n_steps_left > 0)
        )
        # Start the appliances that start after another appliance
        if 'previous_appliance_type' in self.appliances:
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

            # Switch off
            mask_hh_act_stops = mask_change & (n_performing == 0)
            mask_app_stop = (
                mask_apps_this_act & self.appliances['inactive_switch_off']
            )
            if np.any(mask_hh_act_stops):
                # turn off all appliances of this activity if it ends
                self.switch_off(*np.where(
                    mask_hh_act_stops[:, np.newaxis]
                    & mask_app_stop[np.newaxis, :]
                ))
            if 'previous_activity' in self.appliances:
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

    def _update_iteration_variables(self):
        # Don't update, as already taken care of in step method for
        # this simulator
        pass

    def _compute_switch_on_probs(self, indexes_household, indexes_appliance):
        """Compute switch on probs for appliances that can be started.

        These are the probabilites that an appliance is started when
        at least one person in the household is performing the activity
        corresponding to this appliance, at each simulation step.
        Note: the occurence of the activity during the day is
        already simulated by the activity simulator so we don't need
        to do anything there.
        """


        return (
            (
                ~self.get_current_usage()[indexes_household, indexes_appliance]
                & ~np.isin(indexes_appliance, self.apps_dictated_start)
            )
            # TODO find a good solution for that, dont use crest
            * self.switch_on_probs[indexes_household, indexes_appliance]
        )

    def _compute_switch_on_probs_for_one_time(
        self, indexes_household: np.ndarray, indexes_appliance: np.ndarray
    ) -> np.ndarray:
        """Compute probability of switch on events for one-time appliances.

        Appliances that can be switched on only after another one
        or when activity stops.
        """
        return (
            (~self.get_current_usage()[indexes_household, indexes_appliance])
            * self.today_cycles[indexes_household, indexes_appliance]
        )

    def switch_off(self, indexes_household, indexes_appliance):
        # Don't know for how long they will stay on
        self.n_steps_left[indexes_household, indexes_appliance] = 0
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
            self.appliances['after_cycle_delay'][indexes_appliance]
        )

    def _get_start_after_activity(
        self, mask_hh_act_stops, activity
    ):
        """When the activity stops."""

        ind_hh, ind_app = np.where(
            mask_hh_act_stops[:, np.newaxis]
            & (self.appliances['previous_activity'] == activity)[np.newaxis, :]
        )

        # Compute probability of switch on
        switch_on_probs = self._compute_switch_on_probs_for_one_time(
            ind_hh, ind_app
        )
        # Sample the switchon probs
        mask_switch_on = (
            np.random.uniform(size=len(switch_on_probs)) < switch_on_probs
        )
        return (
            ind_hh[mask_switch_on], ind_app[mask_switch_on]
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
            self.appliances['previous_appliance_type'] != 'nan'
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
                indexes_app_start,
                app_id * np.ones(sum(mask_stopped), dtype=int)
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
