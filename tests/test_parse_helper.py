from demod.utils.parse_helpers import (
    states_to_transitions
)
import unittest

import numpy as np


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
