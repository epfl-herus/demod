"""Activity simulators package.
"""
import itertools
from typing import Any, List

import numpy as np

from .base_simulators import Simulator, GetMethod, cached_getter
from ..utils.distribution_functions import check_valid_cdf
from ..utils.monte_carlo import monte_carlo_from_cdf


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

    def __init__(
            self, n_households: int, n_states: int,
            transition_prob_matrices: np.ndarray,
            labels: List[Any] = None, **kwargs):
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
        if (transition_prob_matrices.shape[-1]
            != transition_prob_matrices.shape[-2]):
            raise ValueError('last two elements of transition_prob_matrices'
                ' must be the same size (n possible states)')
        self.transition_prob_matrices = transition_prob_matrices

        if transition_prob_matrices.shape[-1] != n_states:
            raise ValueError('Shape of the transition matrices must be'
            ' n_states*n_states')
        self.n_states = n_states

        # get the labels if correctly given or generates them
        if labels is not None:
            if len(labels) != self.n_states:
                raise ValueError('Length of labels is not the same as the '
                'number of States')
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
        assert len(starting_state_pdf) == self.n_states, ('the starting states'
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
        assert max(starting_state_) < self.n_states and min(starting_state_) >= 0, 'the starting states do not correspond to the size of the transition matrices'
        self.current_states = starting_state_

        self._set_tpm(self.transition_prob_matrices, checkcdf=checkcdf)



    def _set_tpm(self, tpms: np.ndarray, checkcdf: bool = True) -> None:
        """Set TPMs for the Markov Chain.

        Generates cumulative distribution functions for the TPMs and
        saves them as :py:attr:`_cdf_iterator`, that is then used by
        :py:meth:`step`.

        Args:
            tpms: Transition probability matrices
            checkcdf: Whether to check that the values of the tpms are
                okay. Defaults to True.
        """
        #
        cdf = np.cumsum(tpms, axis=-1)
        if checkcdf:
            check_valid_cdf(cdf)
        self._cdf_iterator = itertools.cycle(cdf)  # store cdfs as iterable cycle


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


# TODO impelment this
class TimedStatesSimulator(MarkovChain1rstOrder):
    """Timed states Markov chain Simulator.

    !!! THIS IS NOT IMPLEMENTED YET !!!
    Simulator for a number of Households given as input.
    Includes the duration of the states.
    This is sometimes refer as a 2nd order markov chain.
    It would be nice to sample either from duration matrices or from
    duration fuunctions. (ex Weibul)

    Attributes:
        n_households : int
            The number of households to be simulated.
        n_states : int
            The number of states availables.
        current_states : ndarray(int)
            The current states
    """
    def __init__(self, n_households, n_states, transition_prob_matrices, duration_matrices, labels=None):
        """
        Initialize a simulator for a group of households.

        Parameters:
            n_households : int
                The number of households to be simulated.
            n_states: int
                The number of states
            transition_prob_matrices : numpy.ndarray
                An array containing the matrices for the state transitions at each
                steps. matrices must be of shape n_states*n_states
            duration_matrices : numpy.ndarray
                An array containing the matrices for the duration of the new state.
                matrices must be of shape n_states*n_times, n_times is the possible durations
            labels : list
                optional. A list of labels for the states


        Raises:
            ValueError
                If the parameters have wrong shapes
            TypeError
                If the parameters have wrong types
        """

        super(TimedStatesSimulator, self).__init__(n_households, n_states, transition_prob_matrices, labels=labels)


        if duration_matrices.shape[1] != n_states:
            raise ValueError('Shape[1] of the duraition matrices must be n_states')
        self.duration_matrices = duration_matrices

    def initialize_starting_state(self, starting_state_pdf, start_time_step=0, checkcdf=True):
        """
        Initialize the starting state.

        Parameters
        ----------
        starting_state_pdf : numpy.ndarray
            An array containing the pdf for each of the state at the time 0.
        checkcdf : bool
            optional. Will check if teh transition_prob_matrices have correct values using
            the check_valid_cdf function

        Raises
        ------
        AssertionError
            If the parameters have wrong types or shapes
        """
        super().initialize_starting_state(starting_state_pdf, start_time_step=start_time_step, checkcdf=checkcdf)

        # generates cumulative distribution functions for the times
        cdf = np.cumsum(self.duration_matrices, axis=-1)
        if checkcdf:
            check_valid_cdf(cdf)
        self._times_cdfs = cdf
        self._times_cdf_iterator = itertools.cycle(cdf)  # store cdfs as iterable cycle

        # TODO : find a way to properly initialize that value
        self.times_left = np.zeros_like(self.current_states, dtype=int)

    def step(self):
        mask_change_state = self.times_left < 1
        # draw a montecarlo estimate on the current time
        cdf_state = next(self._cdf_iterator)
        self.current_states[mask_change_state] = monte_carlo_from_cdf(cdf_state[self.current_states[mask_change_state], :])
        # draw a montecarlo estimate of the time spent on this state
        cdf_time = next(self._times_cdf_iterator)
        self.times_left[mask_change_state] = monte_carlo_from_cdf(cdf_time[self.current_states[mask_change_state], :])
        # update the time
        self.current_time_step += 1
        self.times_left -= 1

