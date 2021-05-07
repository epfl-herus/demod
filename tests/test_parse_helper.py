from demod.utils.parse_helpers import (
    states_to_transitions,
    states_to_tpms
)
import unittest

import numpy as np


class TestStatesToTPM(unittest.TestCase):
    def test_simple(self):
        states = np.array([
            [0, 0],
            [0, 1],
        ], dtype=int)
        tpms = states_to_tpms(
            states, first_tpm_modification_algo='nothing'
        )
        print(tpms)
        self.assertTrue(np.all(
            tpms == np.array([
                [
                    [1., 0.],
                    [1., 0]
                ],
                [
                    [0.5, 0.5],
                    [0., 1.]
                ]
            ])
        ))
    def test_last_strategy(self):
        states = np.array([
            [0, 0],
            [0, 1],
        ], dtype=int)
        tpms = states_to_tpms(
            states, first_tpm_modification_algo='last'
        )
        print(tpms)
        self.assertTrue(np.all(
            tpms == np.array([
                [
                    [0.5, 0.5],
                    [0., 1.]
                ],
                [
                    [0.5, 0.5],
                    [0., 1.]
                ]
            ])
        ))

class TestStatesToTransitions(unittest.TestCase):
    def test_simple(self):
        """Test a simple case with one transition."""
        states = np.array([
            [0, 0],
            [0, 1],
        ], dtype=int)
        transition_dic = states_to_transitions(states)
        self.assertTrue(np.all(
            transition_dic['persons'] == np.array([1, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['times'] == np.array([0, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['new_states'] == np.array([0, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['old_states'] == np.array([1, 0], dtype=int)
        ))

    def test_same_transitions(self):
        states = np.array([
            [0, 0],
            [0, 1]
        ], dtype=int)
        transition_dic = states_to_transitions(states, include_same_state=True)

        self.assertTrue(np.all(
            transition_dic['persons'] == np.array([0, 1, 0, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['times'] == np.array([0, 0, 1, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['new_states'] == np.array([0, 0, 0, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['old_states'] == np.array([0, 1, 0, 0], dtype=int)
        ))
