
import unittest





import numpy as np



from demod.utils.sparse import SparseTPM



class TestSparseTPM(unittest.TestCase):

    def test_empty_instance(self):

        empty_array = np.array([])
        sparse_tpm = SparseTPM(empty_array, empty_array, empty_array, empty_array)

    def test_instance(self):

        times = np.array([1])
        inds_from = np.array([2])
        inds_to = np.array([4])
        values = np.array([0.3])
        sparse_tpm = SparseTPM(times, inds_from, inds_to, values)



if __name__ == '__main__':
    unittest.main()
