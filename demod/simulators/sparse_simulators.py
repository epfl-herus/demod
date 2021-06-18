"""Sparse states simulators.

This module contains simulators for Activity based on a sparse
implementation of the transition probability matrices.
"""
from demod.datasets.base_loader import DatasetLoader
from ..utils.data_types import DataInput
import os
import datetime
from typing import List, Union
import warnings

import numpy as np

from .base_simulators import (
    MultiSimulator,
    SimLogger,
    Simulator,
    cached_getter,
)
from ..datasets.GermanTOU.loader import GTOU
from ..utils.monte_carlo import monte_carlo_from_cdf
from ..utils.distribution_functions import check_valid_cdf
from ..utils.sim_types import *
from ..utils.sparse import SparseTPM
from .util import OLD_DATASET_PATH

MAX_PEOPLE_HOUSEHOLD = (
    6 + 1
)  # always need to get one above the real max personas


# define the sparse states simulator
class SparseStatesSimulator(Simulator):
    """States simulator.

    Provides the basis for a states simulator that used a sparse
    implementation.

    Attributes:
        tpm: The set of TPMs used by the simulator.
        state_labels: The labels of the states in the TPM.
    """

    tpm: SparseTPM
    state_labels: StateLabels

    def assign_sparse_tpm_and_labels(
        self, sparse_tpm: SparseTPM, labels: StateLabels = None
    ) -> None:
        """Assign the TPMs to the simulator.

        Args:
            sparse_tpm: The set of TPMs to be used by the simulator.
            labels : The labels of the states in the TPM.
                Defaults to None.

        Raises:
            TypeError: Wrong types in the inputs
            ValueError: TPM has wrong shape
            ValueError: Labels have wrong length
        """
        if not isinstance(sparse_tpm, SparseTPM):
            raise TypeError("transition_matrices must be a SparseTPM")
        # checks that transitions matrices are NxN, where N = n_states
        if sparse_tpm.shape[-1] != sparse_tpm.shape[-2]:
            raise ValueError(
                "last two elements of transition_matrices must \
                be the same size (n possible states)"
            )
        self.tpm = sparse_tpm
        self.n_states = sparse_tpm.shape[-1]
        self.n_times = sparse_tpm.shape[0]

        # get the labels if correctly given or generates them
        if labels is not None:
            if len(labels) != self.n_states:
                raise ValueError(
                    "Length of labels is not the same as the \
                    number of states"
                )
            self.state_labels = labels
        else:
            self.state_labels = np.arange(self.n_states)

    def _initialize_time(self, time: datetime.datetime):
        if (time is None) or (not isinstance(time, datetime.datetime)):
            raise TypeError("time kwarg must be of type datetime.datetime")

        self.time = time

    def __init__(
        self,
        n_households: int,
        sparse_tpm: SparseTPM,
        starting_state_pdf: Union[List[float], np.ndarray],
        labels: StateLabels = None,
        time_aware: bool = False,
        start_datetime: datetime.datetime = None,
        use_quarters: bool = False,
        use_7days: bool = False,
        use_week_ends_days: bool = True,
        **kwargs
    ) -> None:
        """Create a sparse states simulator.

        Args:
            n_households: The number of households to simulate
            sparse_tpm: The TPMs used by the simulator
            labels: The labels of the states. Defaults to None.
            time_aware: Whether the simulator should keep track of the
                time as datetime object. Defaults to False.
            time: The time at the start of the simulation.
                Defaults to None.
            use_quarters: Distinguish the subgroups depending on the
                quarters of a year. Defaults to False.
            use_7days: Distinguish the subgroups between the 7 days of
                the weeks. Defaults to False.
            use_week_ends_days: Distringuish the subgroups between
                weekdays and weekends. Defaults to True.
        """
        super().__init__(n_households, **kwargs)

        self.assign_sparse_tpm_and_labels(sparse_tpm, labels=labels)

        # handles the time aware simulators

        # if given a start date, make it time aware
        if start_datetime is not None:
            time_aware = True
        self.time_aware = time_aware
        if time_aware:
            self._initialize_time(start_datetime)
            # initialize the use of the quarters
            self.use_quarters = use_quarters
            # initialize the use of number of days
            if use_7days and use_week_ends_days:
                warnings.warn(
                    "Cannot use 7 days and week-end-days splits,\
                    will not differentiate the days"
                )
                self.n_different_days_to_use = 0
            elif use_7days:
                self.n_different_days_to_use = 7
            elif use_week_ends_days:
                self.n_different_days_to_use = 2
            else:
                self.n_different_days_to_use = 0

            # gets the initial time step
            start_datetimestep = self._getstart_time_step_from_datetime(
                start_datetime
            )
        else:
            start_datetimestep = 0

        self.initialize_starting_state(
            starting_state_pdf, start_time_step=start_datetimestep, checkcdf=False
        )

    def _getstart_time_step_from_datetime(
        self, start_datetime: datetime.datetime
    ):
        # gets the timestep from the German TOU
        # which begins at 4 am with 10 minutes steps
        start_timestep = start_datetime.minute // 10
        hour = start_datetime.hour
        start_timestep += 6 * (hour - 4 if hour >= 4 else hour + 20)
        return int(start_timestep)

    def get_n_doing_state(self, state: Union[int, str, list]):
        """Get the number of residents doing the desired state or activity.

        Args:
            state:
                the requested state or activity,
                or the list of the requested states

        Raises:
            ValueError: If the given value of input is not valid
            Exception: If the simulator object has not been initialized
                with the :py:attr:`activity_labels` attributes
            TypeError: If the input state has an undesired type

        Returns:
            The number of occupants performing the activity, or the list
            of ndarrays if :py:obj:`state` was given as a list
        """
        if isinstance(state, list):
            return [self.get_n_doing_state(s) for s in state]
        elif isinstance(state, int):
            act_number = state
            return np.array(
                self.state_labels[self.current_states]
                % (MAX_PEOPLE_HOUSEHOLD ** (act_number + 1))
                // (MAX_PEOPLE_HOUSEHOLD ** (act_number)),
                dtype=int,
            )

        elif isinstance(state, str):
            if hasattr(
                self, "activity_labels"
            ):  # chekc that activity labels has been properly initialized
                if np.isin(state, self.activity_labels):
                    # gets the position of this state in the list
                    act_number = int(*np.where(self.activity_labels == state))
                    return self.get_n_doing_state(
                        act_number
                    )  # recursively get the value
                else:
                    raise ValueError(
                        "Invalid activity name : "
                        + state
                        + ", not in "
                        + str(self.activity_labels)
                    )
            else:
                raise Exception(
                    str(self)
                    + " does not have 'activity_labels', so cannot get the desired state."
                )
        else:
            raise TypeError(
                "Invalid type for get_state, must int, str or list of int/str"
            )

    def get_n_doing_activity(self, activity: str):
        """Return the number of residents doing the desired activity."""
        return self.get_n_doing_state(activity)

    def initialize_starting_state(
        self, starting_state_pdf, start_time_step=0, checkcdf=True
    ):
        """Initialize the starting state.

        Performs some steps to reach start_time_step.

        Parameters:
            starting_state_pdf : numpy.ndarray
                An array containing the pdf for each of the state at the time 0.
            checkcdf : bool
                optional. Will check if teh transition_matrices have correct values using
                the check_valid_cdf function

        Raises:
            AssertionError
                If the parameters have wrong types or shapes
        """
        # generates from the distribution of states
        assert (
            len(starting_state_pdf) == self.n_states
        ), "the starting states do not correspond to the size of the transition matrices"

        # get the starting state cdf
        starting_state_cdf = np.cumsum(starting_state_pdf)
        # broadcast to the number of households as they all have this same cdf
        starting_state_cdf = np.broadcast_to(
            starting_state_cdf, (self.n_households, self.n_states)
        )
        if checkcdf:
            check_valid_cdf(starting_state_cdf)

        # sample the starting state from its cdf
        starting_state_ = monte_carlo_from_cdf(starting_state_cdf)
        assert (
            max(starting_state_) < self.n_states and min(starting_state_) >= 0
        ), "the starting states do not correspond to the size of the transition matrices"
        self.current_states = starting_state_

        if self.time_aware:
            self.update_new_day()

        # generates cumulative distribution functions for the transition matrices

        if checkcdf:
            self.tpm.check_valid_cdf()

        # first intialize the sparse state, then udpate the steps
        super(SparseStatesSimulator, self).initialize_starting_state(
            start_time_step=start_time_step
        )

    def update_new_day(self):
        raise NotImplementedError(
            "Need to implement the actions occuring \
            when a new day arrives"
        )

    def step(self, increment=datetime.timedelta(minutes=10)):

        if self.time_aware:
            self.time = self.time + increment  # update the time if time aware
            if (
                self.current_time_step % 144 == 0
            ):  # if it has been reinitialized
                self.update_new_day()

        # update the time step
        super().step()

        # draw a montecarlo estimate on the current time

        self.current_states = self.tpm.sparse_monte_carlo(
            self.current_time_step - 1,  # -1 because it was already updated
            self.current_states,
        )


class SparseActivitySimulator(SparseStatesSimulator):
    """Should be able to simulate any kind of activitys.

    Not fully implemented yet.
    TODO: add handling of occupna;cy and active occ through masks from data
    TODO: test

    Args:
        SparseStatesSimulator: [description]
    """
    def __init__(self, n_households, subgroup_kwargs, data="Germany"):
        """Creates a sparse simulator for activities.

        Currently only implemented for Germany.

        Args:
            n_households (int): The number of households.
            subgroup_kwargs (dict): The subgroup to be simulated.
            data (str, optional): The dataset to be used. Defaults to 'Germany'.

        Raises:
            ValueError: If the data name is unknown.
        """

        if data == "Germany":

            path = OLD_DATASET_PATH + subgroup_kwargs
            # load the tpm
            sparse_tpm = SparseTPM.load(path)
            labels = np.load(path + "_labels.npy")
            activity_labels = np.load(path + "_activity_labels.npy")
            initial_pdf = np.load(path + "_initialpdf.npy")

            self.active_occupancy_mask = np.array(
                [
                    False
                    if (act == "HWH" or act == "HOH" or act == "not active")
                    else True
                    for act in activity_labels
                ]
            )

        else:
            raise ValueError("Unkown data name")

        self.subgroup_kwargs = subgroup_kwargs.copy()
        self.activity_labels = activity_labels

        super().__init__(n_households, sparse_tpm, initial_pdf, labels=labels)

    def get_active_occupancy(self):
        if self.active_occupancy_mask is None:
            raise ValueError(
                "No active occupancy mask was defined in simulator"
            )

        non_active_occupants = np.zeros(self.n_households, dtype=np.uint64)
        for act_number, is_not_active_occ in enumerate(
            self.active_occupancy_mask
        ):
            if not is_not_active_occ:
                non_active_occupants += (
                    self.state_labels[self.current_states]
                    % (MAX_PEOPLE_HOUSEHOLD ** (act_number + 1))
                    // (MAX_PEOPLE_HOUSEHOLD ** (act_number))
                )
        # the number of residents minus the number of non active or non occupants will be the number of active occupants
        return self.subgroup_kwargs["n_residents"] - non_active_occupants


class SparseTransitStatesSimulator(SparseStatesSimulator):
    """Simulates the Occupancy and Activity based on Transit.

    The states are simulated as household states using the number
    of residents doing an activity.
    Replaces the 'at home' state of a :ref:`overview_4_States`
    by 'HWH' and 'HOH', which represent
    transit activity for a work activity (home-work-home)
    or an other activity (home-other-home).
    See :ref:`overview_transit_occupancy`.
    You can access the number of persons in HOH by using
    :py:obj:`self.get_n_doing_activity('HOH')`.
    :py:attr:`~demod.utils.cards_doc.Params.data` must
    be initialized with 'Sparse9States' as
    :py:attr:`~demod.utils.cards_doc.Loader.activity_type`.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.subgroup`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.time_aware`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`

    Data
        :py:meth:`~demod.utils.cards_doc.Loader.load_sparse_tpm`
    Step input
        None.
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_active_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`
        :py:meth:`~demod.utils.cards_doc.Sim.get_n_doing_activity`
    Step size
        10 Minutes.
    """

    def __init__(
        self,
        n_households: int,
        subgroup_kwargs: Subgroup,
        data: DataInput = GTOU('Sparse9States'),
        **kwargs
    ) -> None:
        """Creates a Transit state simulator.

        Args:
            n_households: The number of households to simulate.
            subgroup_kwargs: The subgroups to simulate
            data: The dataset to be used. Defaults to 'Germany'.

        Raises:
            ValueError: [description]
        """
        self.data = data

        # Check that we have the number of occupants int subgroup_kwargs
        # as necessary, warn if not
        if not "n_residents" in subgroup_kwargs.keys():
            warnings.warn(
                "some methods of the simulator might not work if "
                "'n_residents' is not specified in 'subgroup_kwargs'"
            )
            self.n_residents = MAX_PEOPLE_HOUSEHOLD
        else:
            self.n_residents = subgroup_kwargs["n_residents"]

        # load the data

        (
            sparse_tpm,
            labels,
            activity_labels,
            initial_pdf,
        ) = self.data.load_sparse_tpm(subgroup_kwargs)

        # data is now loaded, saves it in the object
        self.subgroup_kwargs = subgroup_kwargs.copy()
        self.activity_labels = activity_labels

        super().__init__(
            n_households, sparse_tpm, initial_pdf, labels=labels, **kwargs
        )

    @cached_getter
    def get_active_occupancy(self) -> np.ndarray:
        """Get the active occupancy of the households.

        The number of people who are at home and active.
        This is calculated based on the persons who are not away.

        Returns:
            The current number of active occupants in each household

        Note:
            In transit model, we consider :py:obj:`active_occupancy` as
            the minimum number of at_home and active states,
            but if one person is out/active
            and one is at home/not_active, the
            :py:obj:`active_occupancy` will be 1, when it should be 0.
            See more as `ADD LINK`.
        """
        at_home = self.get_occupancy()
        active = self.get_n_doing_state("active")
        return np.minimum(at_home, active)

    @cached_getter
    def get_occupancy(self):
        """Get the occupancy of the households.

        The number of people who are at home is
        :py:attr:`n_residents` - (the number of people in transit
        states).

        Returns:
            ndarray: The current number of occupants in each household
        """
        hwh = self.get_n_doing_state("HWH")
        hoh = self.get_n_doing_state("HOH")
        away_residents = hwh + hoh

        return np.array(self.n_residents - away_residents)

    def get_thermal_gains(self):
        """Compute the thermal power produced by the residents.

        Returns:
            float: The thermal power produced in Watts.
        """
        active_gains = 147.0
        dormant_gains = 84.0
        active_occupants = self.get_active_occupancy()
        dormants = (
            self.n_residents
            - self.get_n_doing_state("HOH")
            - self.get_n_doing_state("HWH")
            - active_occupants
        )
        return dormant_gains * dormants + active_gains * active_occupants

    def update_new_day(self):
        """Update the internal TPMs for a new day.

        Also handles the change of states and labels due to the change
        of the TPMs.
        Manages cases when dead transition occur between two TPMs.
        """
        assert hasattr(
            self, "time"
        ), "No date params in sparse state simulator"
        # print('updating the day', self.time)
        # update the day
        if self.n_different_days_to_use == 0:
            self.subgroup_kwargs["weekday"] = 0
        elif self.n_different_days_to_use == 2:
            if self.time.isoweekday() <= 5:
                self.subgroup_kwargs["weekday"] = [1, 2, 3, 4, 5]
            else:
                self.subgroup_kwargs["weekday"] = [6, 7]
        elif self.n_different_days_to_use == 7:
            self.subgroup_kwargs["weekday"] = self.time.isoweekday()

        # update the quarter if required
        if self.use_quarters:
            self.subgroup_kwargs["quarter"] = (self.time.month - 1) // 3
        else:
            self.subgroup_kwargs["quarter"] = None

        # reload and assign the new states TPM
        sparse_tpm, labels, _, _ = self.data.load_sparse_tpm(
            self.subgroup_kwargs
        )

        # get the current states
        old_states_labels = self.state_labels[self.current_states]
        # print('old labels', old_states_labels)
        # print('new labels', labels)
        # gets the positions of the different states inthe states labels
        old_states_labels, inverse = np.unique(
            old_states_labels, return_inverse=True
        )
        # print('old labels unique and inv', old_states_labels, inverse)
        # finds where the old labels go in the new labels
        old_labels_position_in_new = np.asarray(
            [
                np.where(lab == labels)[0][0]
                if np.where(lab == labels)[0].size == 1
                else -1
                for lab in old_states_labels
            ],
            dtype=int,
        )
        # print('old_labels_position_in_new', old_labels_position_in_new)
        # print(
        #     'number_of_lost_transtitions : ', np.sum(old_labels_position_in_new==-1),
        #     'out of : ', len(old_labels_position_in_new) )
        # TODO: change the assignement of too dangerous
        mask_lost_transition = old_labels_position_in_new == -1
        old_labels_position_in_new[mask_lost_transition] = labels[0]

        # convert states to the new ones
        self.current_states = old_labels_position_in_new[inverse].reshape(-1)
        # print('new states', labels[self.current_states])

        # change the iterators over the cdf
        # the dead states will mean that every one is sleeping
        sparse_tpm.dead_state_value = 0

        self.assign_sparse_tpm_and_labels(sparse_tpm, labels)


class SubgroupsActivitySimulator(MultiSimulator):
    """Multisimulator for Activity simulators.

    Simulates different subsimulators of the activity.

    :py:attr:`~demod.utils.cards_doc.Params.subgroups_list` and
    :py:attr:`~demod.utils.cards_doc.Params.n_households_list` can be
    loaded from a dataset through
    :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
    and :py:func:`~demod.simulators.util.sample_population`.


    Params
        :py:attr:`~demod.utils.cards_doc.Params.subgroups_list`
        :py:attr:`~demod.utils.cards_doc.Params.n_households_list`
        :py:attr:`~demod.utils.cards_doc.Params.subsimulator`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.time_aware`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
    Data
        :py:meth:`~demod.utils.cards_doc.Loader.load_sparse_tpm`
    Step input
        None.
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_active_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`
        :py:meth:`~demod.utils.cards_doc.Sim.get_n_doing_activity`
    Step size
        10 Minutes.
    """
    def __init__(
        self,
        subgroups_list: Subgroups,
        n_households_list: List[int],
        subsimulator: Simulator = SparseTransitStatesSimulator,
        logger: SimLogger = None,
        **kwargs
    ):
        """Create a simulator for multiple sub Activity simulators.

        You can simply pass the :py:obj:`subsimulator` class that you
        want to implement, as well as specifiying how many households
        of each subgroups should be simulated.
        TODO: Time aware simulator. It will know which callbacks should be set
        depending on the implemented call back methods implemented in
        the subsimulator class.
        TODO: Make sure that compatible with other simulators.
        TODO: Write test.
        TODO: remove get method.
        TODO: Decide which kind of simulators are allowed as subsimulators.

        Args:
            subgroups_list: List of the subgroups dict.
            n_households_list: The number of housholds in each subgroups.
            subsimulator: The simulator class to use for simulating the subgroups. Defaults to SparseActivitySimulator.
            logger: A logger object to log the results. Defaults to None.

        Raises:
            TypeError: [description]
        """
        if not (
            isinstance(subsimulator, SparseStatesSimulator)
            or issubclass(subsimulator, SparseStatesSimulator)
        ):
            raise TypeError(
                "subsimulator must be an instance of object SparseStatesSimulator"
            )
        simulators_list = [
            subsimulator(n_hh, subgroup_kwargs, **kwargs)
            for subgroup_kwargs, n_hh in zip(subgroups_list, n_households_list)
            if n_hh > 0
        ]

        super().__init__(simulators_list, logger=logger)
        super().initialize_starting_state()

    def get_n_doing_state(self, state):
        if isinstance(state, list):
            return [self.get_n_doing_state(s) for s in state]
        elif isinstance(state, int) or isinstance(state, str):
            # merges the results of all the subsimulators
            return np.concatenate(
                [s.get_n_doing_state(state) for s in self.simulators]
            )

        else:
            raise TypeError(
                "Invalid type for get_state, must int, str or list of int/str"
            )

    def get_n_doing_activity(self, activity):
        return self.get_n_doing_state(activity)
