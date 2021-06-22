import unittest
import numpy as np


from demod.metrics.states import sparsity, average_state_metric, state_duration_distribution_metric

class SparsityTest(unittest.TestCase):
    def test_(self):
        s = sparsity(np.array([
            0., 1., 0., 1.
        ]))
        self.assertEqual(s, 0.5)

    def test_multidim(self):
        s = sparsity(np.array([
            [0., 1., 0., 1.],
            [1., 1., 1., 1.],
            [0., 0., 0., 0.],
        ]))
        self.assertEqual(s, 0.5)
    def test_sparse(self):
        s = sparsity(np.zeros((2,3,4)))
        self.assertEqual(s, 1.)

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
            average_over_timestep=False
        )
        self.assertTrue(np.all(a == np.array([0.5, 0.])))

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
        self.assertEqual(e[1], 0.5/3.)
        self.assertEqual(e[0], 0.0)
    def test_array_output(self):
        a = state_duration_distribution_metric(
            np.array([[0, 1]]),
            np.array([[0, 1], [1, 1]]),
            average_over_timestep=False
        )
        self.assertTrue(np.all(a[1] == np.array([0.0, 0.5, 0.0])))
