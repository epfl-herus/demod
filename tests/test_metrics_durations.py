"""These tests should be removed by utils"""
import unittest





import numpy as np


from demod.metrics.states import get_states_durations, get_durations_by_states




class TestDurations(unittest.TestCase):
    # tests the differents helper functions on different set
    def test_get_states_duration(self):
        # test state durations
        test_array = np.array([
            [ 1, 1, 0, 2],
            [ 1, 1, 1, 2],
            [ 1, 1, 1, 1]]) # pattern all same

        durations, durations_labels = get_states_durations(test_array)
        self.assertTrue(np.all(durations == np.array([2,1,1,3,1,4])))
        self.assertTrue(np.all(durations_labels == np.array([1,0,2,1,2,1])))

    def test_get_durations_by_states(self):
        # test state durations
        test_array = np.array([
            [ 1, 1, 0, 2],
            [ 1, 1, 2, 2],
            [ 1, 1, 1, 1],
            [ 1, 5, 5, 5]])

        dic = get_durations_by_states(test_array)
        self.assertTrue(np.all(dic[0] == np.array([1])))
        self.assertTrue(np.all(dic[1] == np.array([2, 2, 1, 4])))
        self.assertTrue(np.all(dic[2] == np.array([1, 2])))
        self.assertTrue(np.all(dic[5] == np.array([3])))

#   def test_24_hour_occupancy(self):
#       # test 24 hour occupancy
#       test_array = np.array([
#           np.zeros(144),
#           np.random.randint(0,2, size=144),
#           2*np.ones(144),
#           np.random.randint(2,4, size=144),
#           np.random.randint(0,4, size=144)], dtype=int)
#
#       test_labels = np.array([0,1,10,23]) # must be 2 or 3 for having occupancy
#       occupancy_24 = CREST_get_24h_occupancy(test_array, test_labels)
#       self.assertTrue(np.all(occupancy_24 == np.array([False, False, True, True, False])))



if __name__ == '__main__':
    unittest.main()
