import unittest
import numpy as np


from demod.metrics.states import sparsity

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
