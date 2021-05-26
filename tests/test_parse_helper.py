from demod.utils.parse_helpers import (
    counts_to_pdf,
    get_initial_durations_pdfs,
    states_to_tpms_with_durations,
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

    def test_durations(self):
        states = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=int)
        transition_dic = states_to_transitions(states, return_duration=True)
        print(transition_dic)
        self.assertTrue(np.all(
            transition_dic['persons'] == np.array([1, 1, 2, 2], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['times'] == np.array([0, 1, 1, 2], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['old_states'] == np.array([1, 0, 1, 0], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['new_states'] == np.array([0, 1, 0, 1], dtype=int)
        ))
        self.assertTrue(np.all(
            transition_dic['durations'] == np.array([1, 2, 1, 2], dtype=int)
        ))

class TestCountsToPDF(unittest.TestCase):
    def test_counts_to_pdf(self):
        a = np.array([
            [0, 3, 1],
            [1, 2, 1]
        ])
        pdfs = counts_to_pdf(a)
        self.assertTrue(
            np.all(pdfs == np.array([
                [0., 0.75, 0.25],
                [0.25, 0.5, 0.25]
            ]))
        )

    def test_0_replaces(self):
        a = np.array([
            [0, 0, 0],
            [1, 2, 1]
        ])
        pdfs = counts_to_pdf(a)
        self.assertTrue(
            np.all(pdfs == np.array([
                [1., 0., 0.],
                [0.25, 0.5, 0.25]
            ]))
        )

    def test_higher_shape(self):
        a = np.array([
            [
                [0, 2, 2],
                [1, 2, 1]
            ], [
                [0, 3, 1],
                [1, 2, 1]
            ],
        ])
        pdfs = counts_to_pdf(a)
        self.assertTrue(
            np.all(pdfs == np.array([
                [
                    [0., 0.5, 0.5],
                    [0.25, 0.5, 0.25]
                ], [
                    [0., 0.75, 0.25],
                    [0.25, 0.5, 0.25]
                ],
            ]))
        )


class TestTPMsWithDuration(unittest.TestCase):
    def test_simple(self):
        states = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ], dtype=int)
        tpms, dur0, dur1 = states_to_tpms_with_durations(
            states, first_tpm_modification_algo='nothing',
            ignore_end_start_transitions=False
        )
        print(states_to_transitions(states, return_duration=True))
        print(tpms, dur0, dur1)
        self.assertTrue(np.all(
            tpms == np.array([
                [
                    [1., 0.],
                    [1., 0.]
                ], [
                    [0., 1.],
                    [0., 1.]
                ], [
                    [1., 0.],
                    [1., 0.]
                ]
            ])
        ))
        self.assertTrue(np.all(
            dur0 == np.array([
                [
                    [0., 0.5, 0., 0.5],
                    [1., 0., 0., 0.]
                ], [
                    [1., 0., 0., 0.],
                    [0., 0.5, 0.5, 0.]
                ], [
                    [0., 0., 1., 0.],
                    [1., 0., 0., 0.]
                ]
            ])
        ))


class TestInitialDuration(unittest.TestCase):
    def test_simple(self):
        states = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ], dtype=int)
        pdf = get_initial_durations_pdfs(states)
        print(pdf)
        self.assertTrue(np.all(pdf == np.array(
            [
                [0., 0.5, 0., 0.5],
                [0., 0.5, 0.5, 0.],
            ]
        )))