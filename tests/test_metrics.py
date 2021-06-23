import unittest
import numpy as np


from demod.metrics.loads import (
    cumulative_changes_in_demand, diversity_factor, load_duration, profiles_similarity, time_coincident_demand
)
from demod.metrics.states import (
    sparsity,
    average_state_metric,
    state_duration_distribution_metric,
    levenshtein_edit_distance,
)


class SparsityTest(unittest.TestCase):
    def test_(self):
        s = sparsity(np.array([0.0, 1.0, 0.0, 1.0]))
        self.assertEqual(s, 0.5)

    def test_multidim(self):
        s = sparsity(
            np.array(
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        self.assertEqual(s, 0.5)

    def test_sparse(self):
        s = sparsity(np.zeros((2, 3, 4)))
        self.assertEqual(s, 1.0)


class AverageStateTest(unittest.TestCase):
    def test_same(self):
        e = average_state_metric(
            np.array([[0, 1, 1], [0, 1, 1]]),
            np.array([[0, 1, 2], [0, 1, 1], [0, 1, 0]]),
        )
        self.assertEqual(e, 0)

    def test_simple_case(self):
        e = average_state_metric(
            np.array([[0, 1]]),
            np.array([[0, 1], [1, 1]]),
        )
        self.assertEqual(e, 0.25)

    def test_array_output(self):
        a = average_state_metric(
            np.array([[0, 1]]),
            np.array([[0, 1], [1, 1]]),
            average_over_timestep=False,
        )
        self.assertTrue(np.all(a == np.array([0.5, 0.0])))


class StatesDurationTest(unittest.TestCase):
    def test_same(self):
        dic = state_duration_distribution_metric(
            np.array([[0, 1, 1], [0, 1, 1]]),
            np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]]),
        )
        self.assertEqual(dic[0], 0)
        self.assertEqual(dic[1], 0)

    def test_simple_case(self):
        e = state_duration_distribution_metric(
            np.array([[0, 1]]),
            np.array([[0, 1], [1, 1]]),
        )
        self.assertEqual(e[1], 0.5 / 3.0)
        self.assertEqual(e[0], 0.0)

    def test_array_output(self):
        a = state_duration_distribution_metric(
            np.array([[0, 1]]),
            np.array([[0, 1], [1, 1]]),
            average_over_timestep=False,
        )
        self.assertTrue(np.all(a[1] == np.array([0.0, 0.5, 0.0])))


class ProfilesSimilarityTest(unittest.TestCase):
    def test_basic(self):
        simulated_profiles = np.array(
            [
                [1.0, 2.0],
                [2.0, 2.0],
            ]
        )
        target_profiles = np.array(
            [
                [1.0, 2.0],
            ]
        )
        dist = profiles_similarity(simulated_profiles, target_profiles)
        self.assertEqual(dist.shape, (2, 1))
        self.assertTrue(np.all(dist == np.array([[0], [1]])))

    def test_levenshtein(self):
        d = levenshtein_edit_distance(
            np.array([0, 2, 1]),
            np.array([0, 2, 3]),
        )
        self.assertEqual(d, 1)

    def test_profiles_similarity_with_levenshtein(self):
        arr = profiles_similarity(
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                ]
            ),
            levenshtein_edit_distance,
        )
        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [
                        [1, 0, 2],
                        [0, 1, 1],
                    ]
                )
            )
        )

    def test_profiles_similarity(self):
        arr = profiles_similarity(
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                ]
            ),
        )
        from math import sqrt

        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [
                        [1, 0, sqrt(2)],
                        [0, 1, 1],
                    ]
                )
            )
        )


class LoadMetricsTest(unittest.TestCase):
    def test_cumulative_changes_in_demand(self):
        cum_changes, bin_eges = cumulative_changes_in_demand(
            np.array([
                [0, 1, 2, 0]
            ]),
            bins=3, normalize=False,
        )
        self.assertTrue(np.all(cum_changes == np.array([
            [0., 2./3., 1.]
        ])))
        print(bin_eges)
        self.assertTrue(np.all(bin_eges == np.array([
            [0., 2./3., 4./3., 2.]
        ])))

    def test_cumulative_changes_in_demand_many_arrays(self):
        cum_changes, bin_eges = cumulative_changes_in_demand(
            np.array([
                [0, 1, 2, 0],
                [0, 2, 0, 2],
            ]),
            bins=3, normalize=False,
        )
        self.assertTrue(np.all(cum_changes == np.array([
            [0., 1./3., 1.]
        ])))
        print(bin_eges)
        self.assertTrue(np.all(bin_eges == np.array([
            [0., 2./3., 4./3., 2.]
        ])))

    def test_cumulative_changes_in_demand_specified_bins(self):
        cum_changes, bin_eges = cumulative_changes_in_demand(
            np.array([
                [0, 1, 2, 0],
                [0, 2, 0, 2],
            ]),
            normalize=False, bin_edges=[0.5, 1.5, 2.5]
        )
        print(cum_changes)
        self.assertTrue(np.all(cum_changes == np.array([
            [1./3., 1.]
        ])))

    def test_diversity_factor(self):
        d = diversity_factor(np.array([
            [0., 1., 2.],
            [3., 1., 2.],
        ]))
        self.assertEqual(d, 5./4.)

    def test_diversity_factor_1(self):
        d = diversity_factor(np.array([
            [0., 1., 2.],
            [0., 1., 2.],
        ]))
        self.assertEqual(d, 1)

    def test_time_coincident_demand(self):
        d = time_coincident_demand(np.array([
            [0., 1., 2.],
            [0., 1., 2.],
        ]))
        self.assertEqual(d, 2)
        d = time_coincident_demand(np.array([
            [0., 4., 2.],
            [5., 2., 0.],
        ]))
        self.assertEqual(d, 3.)

    def test_load_durations(self):
        loads, durations = load_duration(np.array([
            [0., 1., 2.],
        ]))
        self.assertTrue(np.all(loads == np.array([0, 1, 2])))
        print(durations)
        self.assertTrue(np.all(durations == np.array([[3, 2, 1]])))

    def test_load_durations_2(self):
        loads, durations = load_duration(np.array([
            [0., 1., 2.],
            [3., 3., 3.],
        ]))
        self.assertTrue(np.all(loads == np.array([0, 1, 2, 3])))
        print(durations)
        self.assertTrue(np.all(durations == np.array([
            [3, 2, 1, 0],
            [3, 3, 3, 3],
        ])))

    def test_load_durations_with_loads(self):
        loads, durations = load_duration(np.array([
            [0., 1., 2.],
            [3., 3., 3.],
        ]), loads=[0, 2])
        self.assertTrue(np.all(loads == np.array([0, 2])))
        print(durations)
        self.assertTrue(np.all(durations == np.array([
            [3, 1],
            [3, 3],
        ])))
