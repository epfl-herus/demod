"""Modules for handling Sparse TPMs.

Very useful when the TPMs are large and have a lot of 0 values in them.
"""

from __future__ import annotations

from typing import Tuple
import warnings
import numpy as np


class SparseTPM():
    """A sparse implementation of the Transition probability Matrices.

    Class to store a set of Transition Probability Matrices at
    different times that is very large and sparse.
    Provides methods for iteration over time, indexing,
    and Monte Carlo sampling.

    Attributes:
        times: The timestep of each transitions
        inds_from: The states indices from which the transition
            is performed
        inds_to: The states indices to which the transition is performed
        values: The probabilities of this transition to occur
        n_states: The total number of states
        n_times: The total number of times
        dead_state_value: The states which is assigned in case of
            dead states

    """

    times: np.ndarray
    inds_from: np.ndarray
    inds_to: np.ndarray
    values: np.ndarray
    n_times: int
    n_states: int
    shape: Tuple[int, int, int]
    dead_state_value: int
    _iter_count: int

    def __init__(
        self,
        times: np.ndarray,
        inds_from: np.ndarray,
        inds_to: np.ndarray,
        values: np.ndarray,
        n_states: int = None,
        n_times: int = None,
        dead_state_value: int = None
    ) -> None:
        """Create a sparse transition probability matrix.

        Args:
            times: The timestep of each transitions, dtype=uint
            inds_from: The states indices from which the transition
                is performed, dtype=uint
            inds_to: The states indices to which the transition
                is performed, dtype=uint
            values: The probabilities of this transition to occur,
                dtype=float
            n_states: The total number of states. Defaults to None.
            n_times: The total number of times. Defaults to None.
            dead_state_value: The states which should be assigned in
                case of dead states. Defaults to None.

        Raises:
            ValueError: If a value is incorrect.
        """
        length = len(times)
        if not (
                length == len(inds_from)
                & length == len(inds_to)
                & length == len(values)):
            raise ValueError('Inputs must match in length')
        self.times = np.array(times, dtype=np.uint16)
        self.inds_from = np.array(inds_from, dtype=np.uint16)
        self.inds_to = np.array(inds_to, dtype=np.uint16)
        self.values = np.array(values)

        if length == 0:  # an empty matrix
            n_states = 0
            n_times = 0

        if n_states is None:
            # assign n_states to the maximum of the two states
            n_states = max(np.max(inds_from), np.max(inds_to)) + 1
        self.n_states = int(n_states)

        if n_times is None:
            # assign n_times to the maximum transition times
            n_times = max(times) + 1
        self.n_times = int(n_times)

        self.shape = (n_times, n_states, n_states)

        self.dead_state_value = dead_state_value

        self._iter_count = 0

    def __getitem__(self, key: int):
        """Get the Transition matrix at the requested time."""
        if isinstance(key, int):
            key = self._check_int_time_value(key)
            # gets the desired key
            mask_ = self.times == key
            # build an empty matrix for that transition
            return_ = np.zeros((self.n_states, self.n_states))
            # assign the values for the corresponding key
            return_[
                self.inds_from[mask_],
                self.inds_to[mask_]
            ] = self.values[mask_]

        else:
            raise TypeError('indexing of sparse tpm type was not understood,\
                must be int')

        return return_

    def _check_int_time_value(self, time: int):
        return time % self.n_times

    def __setitem__(self, key: int, newvalue: np.ndarray):
        """Set a transition matrix at the requested time."""
        if isinstance(key, int):
            key = self._check_int_time_value(key)

            if newvalue.shape != (self.n_states, self.n_states):
                raise ValueError('invalid size of the assigned matrix')

            mask_remove = self.times == key
            # generate the new pattern and remove the old
            ind_from, ind_to = np.nonzero(newvalue)

            self.inds_from = np.r_[self.inds_from[~mask_remove], ind_from]
            self.inds_to = np.r_[self.inds_to[~mask_remove], ind_to]
            self.times = np.r_[
                self.times[~mask_remove],
                np.full_like(ind_from, key)
                ]
            self.values = np.r_[
                self.values[~mask_remove],
                np.array(newvalue[ind_from, ind_to])
                ]
        else:
            raise TypeError('indexing of sparse tpm type was not understood, \
                 must be int or')

    def __iter__(self):
        """Iterate over the transition matrices."""
        self._iter_count = 0
        return self

    def __next__(self):
        """Get the next transition matrix."""
        if self._iter_count < self.n_times:
            return_ = self[self._iter_count]
            self._iter_count += 1
            return return_
        else:
            raise StopIteration

    def sparse_monte_carlo(self, time: int, states: np.ndarray) -> np.ndarray:
        """Apply a MC sampling from the current states given as input.

        Args:
            time: The current time at which the transition probabilities
                are used.
            states: The current states, dtype=int.

        Returns:
            The new states after the monte carlo simulation, dtype=int.
        """
        time = self._check_int_time_value(time)
        # print('times, states', time, states)

        # gets the pdf that we want
        pdf_indices, inv = np.unique(states, return_inverse=True)
        # get mask of the pdfs we required
        # print('pdf_ind, inv', pdf_indices, inv)
        # print('states from available ', self.inds_from[self.times == time])
        mask = (self.times == time) & (np.isin(self.inds_from, pdf_indices))
        # assume states has no dead states
        if np.any(~np.isin(states, self.inds_from[mask])):
            warnings.warn('Are some dead states at step {}'.format(time))
            if self.dead_state_value is not None:
                # change the dead states and resample a montecarlo draw
                states[
                    ~np.isin(states, self.inds_from[mask])
                ] = self.dead_state_value
                return self.sparse_monte_carlo(time, states)

        # compute a flattened cdf containing all the states cdf
        flattened_cdf = np.cumsum(self.values[mask])
        # print('cdf',flattened_cdf)

        # monte carlo part
        # sample the random values, and add their location in the flattened cdf
        rand = np.random.uniform(size=states.shape) + inv
        # print('r', rand)
        # look for the position in the array
        new_states = self.inds_to[mask][np.searchsorted(flattened_cdf, rand)]
        # print('new states', new_states)
        return np.array(new_states)

    def save(self, path: str) -> None:
        """Save the TPM at the specified path.

        Can be then loaded via the load() method.

        Args:
            path: The path to which the TPM should be saved.
        """
        np.save(path+'_tpm_times', self.times)
        np.save(path+'_tpm_inds_from', self.inds_from)
        np.save(path+'_tpm_inds_to', self.inds_to)
        np.save(path+'_tpm_values', self.values)

    @ staticmethod
    def load(path: str) -> SparseTPM:
        """Load a TPM from the specified path.

        The TPM can be saved via the save method.

        Args:
            path: The path to which storing the TPM.

        Returns:
            SparseTPM: the sparse TPM at the requested path.
        """
        times = np.load(path+'_tpm_times.npy',)
        inds_from = np.load(path+'_tpm_inds_from.npy', )
        inds_to = np.load(path+'_tpm_inds_to.npy', )
        values = np.load(path+'_tpm_values.npy', )
        return SparseTPM(times, inds_from, inds_to, values)
