"""Activity simulators package.

Provides some basic implementations of simulators for the activity.
Usually based on Markov Chain models.
"""
from __future__ import annotations
import datetime
import itertools
from typing import Any, Dict, List

import numpy as np

from .base_simulators import (
    Callbacks,
    MultiSimulator,
    Simulator,
    GetMethod,
    TimeAwareSimulator,
    cached_getter
)
from ..utils.distribution_functions import check_valid_cdf
from ..utils.monte_carlo import monte_carlo_from_cdf, monte_carlo_from_pdf
from ..datasets.GermanTOU.loader import GTOU
from ..datasets.tou_loader import LoaderTOU
from ..utils.subgroup_handling import add_time, subgroup_households_to_persons
from ..utils.sim_types import Subgroups


class ActivitySimulator(Simulator):
    """Base simulator for the activity.

    Simply provides  :py:data:`GetMethod`'s useful for all simulators.

    Attributes:
        activity_labels: A list containing the label of the available
            activites.
    """

    activity_labels: List[Any]

    def get_occupancy(self) -> np.array:
        """Return the number of occupants in the households.

        Occupants means that the person is in the house.
        """
        raise NotImplementedError()

    def get_active_occupancy(self) -> np.array:
        """Return the number of active occupants in the households.

        Active occupant means that the person is in the house and that
        this person is active.
        A person is active if not inactive(sleeping or sick).
        """
        raise NotImplementedError()

    @ cached_getter
    def get_thermal_gains(self) -> np.array:
        """Return the thermal gains from the occupants of the house.

        The thermal gains are expressed in Watts.
        """
        raise NotImplementedError()

    def get_performing_activity(self, activity_name: str) -> np.array:
        """Return the number of occupants doing :py:obj:`activity_name`.

        If the given :py:obj:`activity_name` corresponds to an
        activity in :py:attr:`activity_labels`.

        Args:
            activity_name: The desired activity.
        """
        raise NotImplementedError()


class MarkovChain1rstOrder(Simulator):
    """Base simulator for the activity of households.

    Implements a 1^{rst} order Markov chain. The simulator iterates
    over the :py:obj:`transition_prob_matrices`.


    Attributes:
        n_states: The number of states availables.
        current_states: The current states at which the households are.
    """

    n_states: int
    current_states: np.ndarray
    _cdf_iterator: itertools.cycle
    corresponding_loader: str = 'load_tpm'

    def __init__(
            self, n_households: int, n_states: int,
            transition_prob_matrices: np.ndarray,
            labels: List[Any] = None, **kwargs
    ):
        """Initialize a simulator for a group of households.

        Parameters:
            n_households: The number of households to be simulated.
            n_states: The number of states
            transition_prob_matrices:
                An array containing the matrices for the state
                transitions at each steps.
                It must be of shape
                :py:obj:`n_times` * :py:attr:`n_states` *
                :py:attr:`n_states`.
                Transitions probability matrix must start with transtion
                from t-1 to t. where t is the start.
            labels:
                optional. A list of labels for the states


        Raises:
            ValueError:
                If the parameters have wrong shapes
            TypeError:
                If the parameters have wrong types
        """
        super().__init__(n_households, **kwargs)
        # get the transition matrices
        if type(transition_prob_matrices) is not np.ndarray:
            raise TypeError('transition_prob_matrices must be numpy array')
        if (
            transition_prob_matrices.shape[-1]
            != transition_prob_matrices.shape[-2]
        ):
            raise ValueError(
                'last two elements of transition_prob_matrices'
                ' must be the same size (n possible states)'
            )
        self.transition_prob_matrices = transition_prob_matrices

        if transition_prob_matrices.shape[-1] != n_states:
            raise ValueError(
                'Shape of the transition matrices must be'
                ' n_states*n_states'
            )
        self.n_states = n_states

        # get the labels if correctly given or generates them
        if labels is not None:
            if len(labels) != self.n_states:
                raise ValueError(
                    'Length of labels is not the same as the '
                    'number of States'
                )
            self.state_labels = labels
        else:
            self.state_labels = np.arange(self.n_states)

    def initialize_starting_state(
            self, starting_state_pdf: np.ndarray,
            start_time_step: int = 0, checkcdf: bool = True) -> None:
        """Initialize the starting states of the households.

        Parameters:
            starting_state_pdf:
                An array containing the pdf for each of the state at the
                    time 0.
            checkcdf:
                optional. Will check if teh transition_prob_matrices
                have correct values using
                the :py:func:`check_valid_cdf` function

        Raises:
            AssertionError:
                If the parameters have wrong types or shapes
        """
        super().initialize_starting_state(start_time_step=start_time_step)
        # generates from the distribution of states
        assert len(starting_state_pdf) == self.n_states, (
            'the starting states'
            ' do not correspond to the size of the transition matrices')

        # get the starting state cdf
        starting_state_cdf = np.cumsum(starting_state_pdf)
        # broadcast to the number of households as they all have this same cdf
        starting_state_cdf = np.broadcast_to(
            starting_state_cdf, (self.n_households, self.n_states))
        if checkcdf:
            check_valid_cdf(starting_state_cdf)
        self.starting_state_cdf = starting_state_cdf

        # sample the starting state from its cdf
        starting_state_ = monte_carlo_from_cdf(starting_state_cdf)
        assert (
            max(starting_state_) < self.n_states and min(starting_state_) >= 0
        ), (
            'the starting states do not correspond to the size of'
            'the transition matrices'
        )
        self.current_states = starting_state_

        self._set_tpm(self.transition_prob_matrices, checkcdf=checkcdf)

    def _set_tpm(
        self, tpms: np.ndarray, checkcdf: bool = True,
        new_labels: np.ndarray = None
    ) -> None:
        """Set TPMs for the Markov Chain.

        Generates cumulative distribution functions for the TPMs and
        saves them as :py:attr:`_cdf_iterator`, that is then used by
        :py:meth:`step`.

        Args:
            tpms: Transition probability matrices
            checkcdf: Whether to check that the values of the tpms are
                okay. Defaults to True.
            new_labels: if given, will update states and labels according
                to the new labels.
        """
        # Transform the tpms into cdf
        cdf = np.cumsum(tpms, axis=-1)
        if checkcdf:
            check_valid_cdf(cdf)
        # Store cdfs as iterable cycle (will start over at the end)
        self._cdf_iterator = itertools.cycle(cdf)

        # Call the first cdf as the tpms statrt with transitions
        # from t-1 to t
        next(self._cdf_iterator)

        if new_labels is not None:
            self._assign_new_labels(new_labels)

    def _assign_new_labels(self, new_labels: np.ndarray) -> None:
        """Convert the current states for the new labels.

        Will use the old labels and the new labels to try to find
        a matching.
        This might create some incompatibilities as the old states
        and the new states might not be the same.

        Args:
            new_labels: the new labels to assign
        """
        # Updates the states values
        old_states_labels = self.state_labels[self.current_states]
        # gets the positions of the different states inthe states labels
        old_states_labels, inverse = np.unique(
            old_states_labels, return_inverse=True
        )
        # finds where the old labels go in the new labels
        old_labels_position_in_new = np.asarray(
            [
                np.where(lab == new_labels)[0][0]
                if np.where(lab == new_labels)[0].size == 1
                else -1
                for lab in old_states_labels
            ],
            dtype=int,
        )

        mask_lost_transition = old_labels_position_in_new == -1
        # Lost transitions are given the 0 new label, # Dead state label
        old_labels_position_in_new[mask_lost_transition] = 0
        # convert states to the new ones
        self.current_states = old_labels_position_in_new[
            inverse
        ].reshape(-1)
        self.state_labels = new_labels

    def step(self) -> None:
        """Perfom a step Markov chain step.

        Update the current states using the appropriate value form the
        Transition Probability matrix.
        """
        # draw a montecarlo estimate on the current time
        cdf = next(self._cdf_iterator)
        self.current_states = monte_carlo_from_cdf(cdf[self.current_states, :])
        # update the time
        super().step()

    def get_n_doing_state(self, state: int):
        return self.current_states == state

    def get_n_doing_activity(self, activity: Any):
        return self.get_n_doing_state(
            list(self.state_labels).index(activity)
        )


class SemiMarkovSimulator(MarkovChain1rstOrder):
    """Semi Markov chain Simulator.

    Similar to a first order Markov Chains, but it also
    includes the duration of the states.
    This is sometimes refer to as a 2nd order markov chain.

    In the current implementations,
    the duration matrices can depend on only the new
    sampled states, or also on the previous state.
    It would be nice to sample either from duration matrices or from
    duration fuunctions. (ex Weibul)

    The SemiMarkovSimulator is not time-homogeneous, as the
    probabilities changes with respect to time.

    Corresponding data loader:
    :py:meth:`~demod.datasets.tou_loader.LoaderTOU.load_tpm_with_duration`


    Attributes:
        n_subjects : int
            The number of subjects to be simulated.
        n_states : int
            The number of states availables.
        current_states : ndarray(int)
            The current states
    """

    _use_previous_state_for_duration_flag: bool

    def __init__(
        self, n_subjects: int, n_states: int,
        transition_prob_matrices: np.ndarray,
        duration_pdfs: np.ndarray,
        labels: List[Any] = None,
        **kwargs
    ) -> None:
        """Initialize a simulator for a number of subjects.

        Parameters:
            n_subjects : int
                The number of subjjects to be simulated.
            n_states: int
                The number of states
            transition_prob_matrices:
                An array containing the matrices for the state
                transitions at each steps.
                Shape = (n_times, n_states, n_states).
                Transitions $T_{i,i}$ to the same state should not occur.

            duration_pdfs:
                An array containing the matrices for the duration of
                the new state.
                Shape = (n_times, n_states, n_states, n_times) or
                (n_times, n_states, n_times).
                In the first case, the previous states also impacts on
                the next duration. In the second case, only the new
                states impacts.
                The last axis contains the pdf.
            labels : list
                optional. A list of labels for the states


        Raises:
            ValueError
                If the parameters have wrong shapes
            TypeError
                If the parameters have wrong types
        """
        super().__init__(
            n_subjects, n_states, transition_prob_matrices,
            labels=labels, **kwargs
        )
        self.duration_pdfs = duration_pdfs

    def initialize_starting_state(
        self,
        starting_state_pdf: np.ndarray,
        starting_durations_pdfs: np.ndarray,
        start_time_step: int = 0, checkcdf: bool = True
    ) -> None:
        """Initialize the starting state.

        Parameters:
            starting_state_pdf:
                An array containing the pdf for each of the states
                at the time 0.
                Shape = (n_states)
            starting_durations_pdfs:
                An array containing the pdf of the times left
                at the start for each states.
                Shape = (n_states, n_times)
            checkcdf:
                optional. Will check if the pdfs
                have correct values using
                the check_valid_cdf function.

        Raises:
            AssertionError
                If the parameters have wrong types or shapes
        """
        super().initialize_starting_state(
            starting_state_pdf,
            start_time_step=start_time_step,
            checkcdf=checkcdf,
        )
        # Adds the time based
        self.times_left = monte_carlo_from_pdf(
            # Get the pdf of times left corresponding to each current states
            starting_durations_pdfs[self.current_states, :]
        ) - 1  # Duration are uploaded after step, and changed when == 0
        self._set_duration_pdfs(self.duration_pdfs)

    def _set_duration_pdfs(
        self, duration_pdfs: np.ndarray, checkcdf: bool = True,
    ) -> None:
        """Set the pdfs of the duration of states.

        Parameters:
            duration_pdfs:
                An array containing the matrices for the duration of
                the new state.
                Shape = (n_times, n_states, n_states, n_times) or
                (n_times, n_states, n_times).
                In the first case, the previous states also impacts on
                the next duration. In the second case, only the new
                states impacts.
                The last axis contains the pdf.
        """
        self._use_previous_state_for_duration_flag = (
            True if len(duration_pdfs.shape) == 4 else False
        )

        if duration_pdfs.shape[1] != self.n_states:
            raise ValueError(
                'Shape[1] of the duration matrices must be n_states'
            )
        # generates cumulative distribution functions for the times
        cdf = np.cumsum(duration_pdfs, axis=-1)
        if checkcdf:
            check_valid_cdf(cdf)
        self._times_cdfs = cdf
        # store cdfs as iterable cycle (will repeat when finished)
        self._times_cdf_iterator = itertools.cycle(cdf)
        # Call the first cdf as the tpms start with transitions
        # from t-1 to t
        next(self._times_cdf_iterator)

    def step(self):
        """Perform one step of the SemiMarkov model.

        Checks the subjects that require an update.
        Then it updates the new states of these subjects, and from that
        it samples the duration of that new state.
        """
        # Bypass the first order change of states
        Simulator.step(self)

        mask_change_state = self.times_left < 1
        previous_change_states = self.current_states[mask_change_state]
        # draw a montecarlo estimate on the current time
        cdf_state = next(self._cdf_iterator)

        self.current_states[mask_change_state] = monte_carlo_from_cdf(
            cdf_state[previous_change_states, :]
        )
        # draw a montecarlo estimate of the time spent on this state
        cdf_time = next(self._times_cdf_iterator)

        if self._use_previous_state_for_duration_flag:
            # Select the new cdf based also on the previous state
            self.times_left[mask_change_state] = monte_carlo_from_cdf(
                cdf_time[
                    previous_change_states,
                    self.current_states[mask_change_state],
                    :
                ]
            )
        else:
            # Select the new cdf based only on the new state
            self.times_left[mask_change_state] = monte_carlo_from_cdf(
                cdf_time[self.current_states[mask_change_state], :]
            )
        # update the time
        self.times_left -= 1


class SubgroupsIndividualsActivitySimulator(
    ActivitySimulator,
    MultiSimulator,
    TimeAwareSimulator
):
    """Multisimulator for simulating households activity based on individuals.

    Simulates different subsimulators of the activity of the people.
    Then groups them into households.
    Keeps track of the time and when to updates the internal simulators.
    Handles the data flows from the dataset and the subsimulators.

    Note that it does not take into account the correlation between
    the residents, as each resident is simulated independently
    of the others.

    :py:attr:`~demod.utils.cards_doc.Params.subgroups_list` and
    :py:attr:`~demod.utils.cards_doc.Params.n_households_list` can be
    loaded from a dataset through
    :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
    and :py:func:`~demod.simulators.util.sample_population`.


    You can simply pass the
    :py:attr:`~demod.utils.cards_doc.Params.subsimulator`
    class that you
    want to implement, as well as specifiying how many households
    of each subgroups should be simulated.
    At the moment, only :py:class:`.MarkovChain1rstOrder` and
    :py:class:`.SemiMarkovSimulator` are accepted as subsimulators.

    If at least one of the three parameters
    :py:attr:`~demod.utils.cards_doc.Params.use_week_ends_days`
    :py:attr:`~demod.utils.cards_doc.Params.use_7days`
    :py:attr:`~demod.utils.cards_doc.Params.use_quarters`
    is True, the simulator will check if it should
    update its parameters every simulated day.

    This simulator only simulates the activities and how many persons are in
    it. If you want to simulate occupancy, you need to override the following
    methods :
    :py:meth:`~demod.utils.cards_doc.Sim.get_occupancy`
    :py:meth:`~demod.utils.cards_doc.Sim.get_active_occupancy`
    :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`


    Params
        :py:attr:`~demod.utils.cards_doc.Params.subgroups_list`
        :py:attr:`~demod.utils.cards_doc.Params.n_households_list`
        :py:attr:`~demod.utils.cards_doc.Params.subsimulator`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.use_week_ends_days`
        :py:attr:`~demod.utils.cards_doc.Params.use_7days`
        :py:attr:`~demod.utils.cards_doc.Params.use_quarters`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:attr:`~demod.utils.cards_doc.Loader.refresh_time`
        :py:meth:`~demod.datasets.tou_loader.LoaderTOU.load_tpm`
        or :py:meth:`~demod.datasets.tou_loader.LoaderTOU.tpm_with_duration`
    Step input
        None.
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_n_doing_activity`
    Step size
        10 Minutes.
    """

    def __init__(
        self,
        subgroups_list: Subgroups,
        n_households_list: List[int],
        subsimulator: Simulator = MarkovChain1rstOrder,
        data: LoaderTOU = GTOU('DemodActivities_0'),
        use_week_ends_days: bool = False,
        use_7days: bool = False,
        use_quarters: bool = False,
        **kwargs
    ):
        """Create a simulator for multiple sub Activity simulators.

        Args:
            subgroups_list: List of the subgroups dict, the list can contain
                hh subgroup dictionaries, or can contains lists of person
                subgroups.
            n_households_list: The number of housholds in each subgroups.
            subsimulator: The simulator class to use for simulating the
                subgroups. Defaults to SparseActivitySimulator.
            logger: A logger object to log the results. Defaults to None.

        Raises:
            TypeError: [description]
        """
        self.data = data
        # Save the update time parameters
        self.use_week_ends_days = use_week_ends_days
        self.use_7days = use_7days
        self.use_quarters = use_quarters
        self.time_aware = (use_week_ends_days or use_7days or use_quarters)

        # Check the subsimulator and define what should be done
        self._parse_subsimulator(subsimulator)

        # Read the subgroups
        unique_persons, persons_counted = self._parse_subgroups_input(
            subgroups_list, n_households_list,
        )

        self.subgroups_persons = unique_persons

        simulators_list = self._initialize_subsimulators(
            subsimulator, unique_persons, persons_counted
        )

        MultiSimulator.__init__(self, simulators_list)
        self.n_households = sum(n_households_list)
        self.activity_labels = simulators_list[0].state_labels
        # This could be read from the dataset, but as far as I know,
        # all TOU are 10 min based ?
        step_size = datetime.timedelta(minutes=10)
        if 'step_size' in kwargs:

            if step_size != kwargs['step_size']:
                raise ValueError(
                    "'step_size' = {} was specified in {}'"
                    ". It uses the step_size = {}, "
                    " which is not the same.".format(
                        kwargs['step_size'],
                        self,
                        step_size,
                    ))

            kwargs = kwargs.copy()
            kwargs.pop('step_size')

        TimeAwareSimulator.__init__(
            self, self.n_households,
            step_size=step_size,
            **kwargs
        )
        self.initialize_starting_state(
            initialization_time=data.refresh_time
        )

    def _parse_subsimulator(self, subsimulator):
        """Define how the type of subsimulator changes the simulation."""
        if subsimulator == MarkovChain1rstOrder:
            self._initialize_subsimulators = (
                self._initialize_sub_MarkovChain1rstOrder
            )
            self._update_tpms = self._update_tpms_MarkovChain1rstOrder

        elif subsimulator == SemiMarkovSimulator:
            self._initialize_subsimulators = (
                self._initialize_sub_SemiMarkovSimulator
            )
            self._update_tpms = self._update_tpms_SemiMarkovSimulator

        else:
            raise TypeError(
                "subsimulator must be an instance of object "
                "MarkovChain1rstOrder or SemiMarkovSimulator not ''{}'".format(
                    subsimulator
                )
            )

    def _parse_subgroups_input(self, subgroups_list, n_households_list):
        """Parse the subgroups given as input for the simulation.

        Return the person subgroups, and the number of each subgroup
        that has to be simulated.
        """
        subgroup_persons = subgroup_households_to_persons(
            subgroups_list,
        )

        self.n_residents = np.concatenate([
            subgroup['n_residents'] * np.ones(n_households, dtype=int)
            for subgroup, n_households
            in zip(subgroups_list, n_households_list)
        ]).reshape(-1)

        # Counts the persons
        unique_persons, person_numbers = np.unique(
            np.concatenate(subgroup_persons), return_counts=True
        )

        # Adds an empty list of each person type that maps the persons
        # simulated in this multisimulator to the households
        hh_of_person = [[] for _ in person_numbers]  # Fill in the loop down

        # Tracks how many persons of each sim where counted
        persons_counted = np.zeros_like(person_numbers)
        _past_hh_counts = 0  # Tracks how many hh have been visited
        for hh_persons, n_hh in zip(subgroup_persons, n_households_list):
            for pers_subgroup in hh_persons:
                # Find which will be the subsim that simulates this person
                _ind_subsim = list(unique_persons).index(pers_subgroup)
                # This persone suubgrouup has been counted n_hh times
                persons_counted[_ind_subsim] += n_hh
                # Appends the household number
                for hh in range(n_hh):
                    hh_of_person[_ind_subsim].append(_past_hh_counts + hh)

            _past_hh_counts += n_hh

        self.hh_of_persons = np.concatenate(hh_of_person)

        return unique_persons, persons_counted

    def _update_subgroups_with_time(self):
        """Update the subgroup of the sub simulators using current_time."""
        self.subgroups_persons = [
            add_time(
                subgroup, self.current_time,
                use_week_ends_days=self.use_week_ends_days,
                use_7days=self.use_7days,
                use_quarters=self.use_quarters,
            ) for subgroup in self.subgroups_persons
        ]

    def initialize_starting_state(
        self, initialization_time: datetime.datetime,
    ) -> None:
        """Initialize the starting state.

        The time aware part is intialized, as well as the subsimulators.
        """
        TimeAwareSimulator.initialize_starting_state(
            self, initialization_time=initialization_time
        )

    def _initialize_sub_MarkovChain1rstOrder(
        self, subsimulator, unique_persons, persons_counted
    ) -> List[Simulator]:
        subsimulators = []
        for subgroup, n_persons in zip(unique_persons, persons_counted):
            (  # Load the data
                tpm,
                labels,
                initial_pdf,
            ) = self.data.load_tpm(subgroup)
            # Instantiate
            sim = subsimulator(
                n_persons,  # n_subjects
                len(labels),  # n_states
                tpm,  # transition_prob_matrices
                labels,  # labels
            )
            # Initialize
            sim.initialize_starting_state(initial_pdf)
            # Store in memory
            subsimulators.append(sim)

        return subsimulators

    def _initialize_sub_SemiMarkovSimulator(
        self, subsimulator, unique_persons, persons_counted
    ) -> List[Simulator]:
        """Instantiate and intialize semi markov subsimulators.

        This could be used for any simulator that has the same signature.
        """
        subsimulators = []
        for subgroup, n_persons in zip(unique_persons, persons_counted):
            (  # Load the data
                tpm,
                duration_pdfs,
                labels,
                initial_pdf,
                initial_duration_pdfs
            ) = self.data.load_tpm_with_duration(subgroup)
            # Instantiate
            sim = subsimulator(
                n_persons,  # n_subjects
                len(labels),  # n_states
                tpm,  # transition_prob_matrices
                duration_pdfs,  # duration_pdfs
                labels,  # labels
            )
            # Initialize
            sim.initialize_starting_state(initial_pdf, initial_duration_pdfs)
            # Store in memory
            subsimulators.append(sim)

        return subsimulators

    def _initialize_subsimulators(
        self, subsimulator, unique_persons, persons_counted
    ) -> List[Simulator]:
        raise NotImplementedError(
            'Abstract method that should be changed depending on '
            'the subsimulator used'
            )

    def _update_tpms_MarkovChain1rstOrder(self, sim, subgroup) -> None:
        (  # Load the data
            tpm,
            labels,
            initial_pdf,
        ) = self.data.load_tpm(subgroup)
        # Change TPM
        sim._set_tpm(tpm, new_labels=labels)

    def _update_tpms_SemiMarkovSimulator(self, sim, subgroup) -> None:
        # Load the data
        (
            tpm,
            duration_pdfs,
            labels,
            _,
            _
        ) = self.data.load_tpm_with_duration(subgroup)
        # Change TPM
        sim._set_tpm(tpm, new_labels=labels)
        sim._set_duration_pdfs(duration_pdfs)

    def _update_tpms(self, sim, subgroup) -> None:
        raise NotImplementedError(
            'Abstract method that should be changed depending on '
            'the subsimulator used'
            )

    def _create_multi_getter(self, getter_name: str) -> GetMethod:
        """Create a getter methods that retrieves households from the persons.

        Overrides the Multisimulator default multigetters, to map
        the individuals simulated by the suubsimulators to households.

        Args:
            getter_name: The name of the getter method
        """
        parent_getter = MultiSimulator._create_multi_getter(self, getter_name)

        def getter():
            getted_array = parent_getter(self)  # Call the parent getter
            # Assume Persons getter can only return 1s and 0s
            mask_out = np.array(getted_array, dtype=bool)
            # Counts how many persons in the households are 1s
            u, c = np.unique(self.hh_of_persons[mask_out], return_counts=True)
            out = np.zeros(self.n_households, dtype=int)
            out[u] = c
            return out

        # also assing the doc
        getter.__doc__ = parent_getter.__doc__

        return getter

    def get_n_doing_activity(self, activity):
        """Return the number of people in the desired state."""
        if isinstance(activity, list):
            return [self.get_n_doing_activity(a) for a in activity]
        # merges the results of all the subsimulators
        persons_states = np.concatenate(
            [s.get_n_doing_activity(activity) for s in self.simulators]
        )
        # Assume Persons getter can only return 1s and 0s
        mask_out = np.array(persons_states, dtype=bool)
        # Counts how many persons in the households are 1s
        u, c = np.unique(self.hh_of_persons[mask_out], return_counts=True)
        states_in_hh = np.zeros(self.n_households, dtype=int)
        states_in_hh[u] = c
        return states_in_hh

    def get_performing_activity(self, activity_name: str) -> np.array:
        return self.get_n_doing_activity(activity_name)

    @ Callbacks.after_refresh_time
    def step(self) -> None:
        """Update the simulator."""
        # Simply calls the parents (Multi for the sims and TimeAware for time)
        return super().step()

    def on_after_refresh_time(self) -> None:
        """Update the tpms."""
        # Check if we want to update the tpms
        if self.time_aware:
            self._update_subgroups_with_time()
            # Updates the TPMs
            [
                self._update_tpms(sim, subgroup) for sim, subgroup
                in zip(self.simulators, self.subgroups_persons)
            ]

    @ cached_getter
    def get_activity_states(self) -> Dict[str, np.ndarray]:
        """Return a dictionary containing the persons in each state."""
        states = {
            lab: self.get_n_doing_activity(lab) for lab in self.activity_labels
        }
        states['active_occupancy'] = self.get_active_occupancy(states)
        return states

    def get_occupancy(self, states=None) -> np.array:
        """Return the active occupancy of an activity simulator.

        Reads the state 'away' and get_occupancy to deduce it.
        """
        if states is None:
            states = self.get_activity_states()
        return self.n_residents - states['away']

    def get_active_occupancy(self, states=None) -> np.array:
        """Return the active occupancy of an activity simulator.

        Reads the state 'sleeping' and get_occupancy to deduce it.
        """
        if states is None:
            states = self.get_activity_states()
        occupancy = (
            self.get_occupancy() if states is None else
            self.get_occupancy(states)
        )
        return occupancy - states['sleeping']
