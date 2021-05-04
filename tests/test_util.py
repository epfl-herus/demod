from datetime import date, datetime, timedelta

import numpy as np
from demod.simulators.util import subgroup_add_time, sample_population
from demod.utils.appliances import get_ownership_from_dict, find_closest_type, remove_start
import unittest


class TestAppliancesUtility(unittest.TestCase):
    long_type = 'a_b_c_app'
    single_type = 'app'
    def test_remove_start(self):
        new = remove_start(self.long_type)
        self.assertEqual(new, 'b_c_app')
        new = remove_start(self.single_type)
        self.assertFalse(new)

    def test_find_closest(self):
        new = find_closest_type(self.long_type, [self.single_type, 'asdfsd'])
        self.assertEqual(new, self.single_type)
    def test_find_fails(self):
        new = find_closest_type(self.long_type, ['nope', 'asdfsd'])
        self.assertFalse(new)

    def test_ownership_from_dict(self):
        ownership_dict = {'app': 0.3, 'c_app': 0.2, 'c_app_2': 0.1}
        appliances_dict = {
            'type': [self.long_type, self.single_type, self.long_type]
        }
        probs = get_ownership_from_dict(appliances_dict, ownership_dict)
        print(probs)
        self.assertTrue(np.all(probs == np.array([0.2, 0.3, 0.1])))

    def test_complex_ownership_from_dict(self):
        ownership_dict = {'app': 0.3, 'app_2': 0.2, 'app_3': 0.1}
        appliances_dict = {
            'type': [self.long_type, self.single_type, self.long_type]
        }
        probs = get_ownership_from_dict(appliances_dict, ownership_dict)
        print(probs)
        self.assertTrue(np.all(probs == np.array([0.3, 0.2, 0.1])))

    def test_ownership_from_dict_doc(self):
        # code snippet of the documentation
        appliances_dict = {'type': ['hob', 'electric_hob']}
        ownership_dict = {'hob': 0.9 , 'hob_2': 0.1}
        probs = get_ownership_from_dict(appliances_dict, ownership_dict)
        print(probs)
        self.assertTrue(np.all(probs == np.array([0.9, 0.1])))

    def test_too_many_appliances_ownership_from_dict(self):
        """Checks that ownership of a i-eth app that is not planned gives 0.
        """
        ownership_dict = {'app': 0.3, 'app_2': 0.2}
        appliances_dict = {
            'type': ['app', 'app', 'app', 'app']
        }
        probs = get_ownership_from_dict(appliances_dict, ownership_dict)
        self.assertTrue(np.all(probs == np.array([0.3, 0.2, 0.0, 0.0])))






class TestUtilityFunctions(unittest.TestCase):

    def test_subgroup_add_time(self):
        dic = {}
        d = date(2014,1,1)
        # default
        dic_out = subgroup_add_time(dic, d)
        self.assertEqual(dic['weekday'], [1,2,3,4,5])
        # check retuurn is the same
        self.assertEqual(dic, dic_out)
        # try specific day
        subgroup_add_time(dic, d, use_7days=True, use_week_ends_days=False)
        self.assertEqual(dic['weekday'], 3)
        # try with datetime object as well
        dt = datetime(2014,1,1,0,0,0)
        subgroup_add_time(dic, dt)
        self.assertEqual(dic['weekday'], [1,2,3,4,5])
        # try weekend
        d_e = date(2014,1,4)
        subgroup_add_time(dic, d_e)
        self.assertEqual(dic['weekday'], [6,7])
        # try quarter only
        dic_2 = {}
        subgroup_add_time(dic_2, d, use_quarters=True, use_week_ends_days=False)
        self.assertEqual(dic_2['quarter'], 1)
        self.assertNotIn('weekday', dic_2)
        # try quarter computation
        dates = [date(2014,1,15) + timedelta(days=30*i) for i in range(12)]
        quarters = [subgroup_add_time({}, d, use_quarters=True)['quarter'] for d in dates]
        self.assertEqual(quarters, sum([3*[i+1] for i in range(4)], []))
        # check that is does not override other attribuutes
        dic_full = {'test':3}
        subgroup_add_time(dic_full, d, use_7days=True, use_week_ends_days=False)
        self.assertEqual(dic_full['test'], 3)
        # check error if uuse both 7 days and weekendays
        self.assertRaises(ValueError, subgroup_add_time, dic, d, use_7days=True, use_week_ends_days=True)


class TestPopulationSampling(unittest.TestCase):
    test_pdf = np.array([0.2, 0.5, 0.3])
    algo = 'real_population'

    def test_numbers(self):
        out = sample_population(1, self.test_pdf, self.algo)
        print(out)
        self.assertEqual(sum(out), 1)
        out = sample_population(10, self.test_pdf, self.algo)
        print(out)
        self.assertEqual(sum(out), 10)
        out = sample_population(10, np.array([1]))
        print(out)
        self.assertEqual(sum(out), 10)

    def test_output_size(self):
        out = sample_population(1, self.test_pdf, self.algo)
        self.assertEqual(len(out), len(self.test_pdf))
        out3 = sample_population(10, self.test_pdf, self.algo)
        self.assertEqual(len(out3), len(self.test_pdf))
        out2 = sample_population(10, np.array([0.2, 0.8]), self.algo)
        self.assertEqual(len(out2), 2)
        out1 = sample_population(10, np.array([1]), self.algo)
        self.assertEqual(len(out1), 1)

    def test_wrong_pdf(self):
        self.assertRaises(ValueError, sample_population,
        1, np.array([0.3, 0.2]), self.algo)

    def test_real_population_values(self):
        """Test only the real population algo for ensuring correct values"""
        out = sample_population(10, self.test_pdf, 'real_population')
        self.assertTrue(np.all(np.array([2,5,3]) == out))

        out = sample_population(10, np.array([1]), 'real_population')
        self.assertTrue(np.all(np.array([10]) == out))

        # check cases with small numbers
        out = sample_population(1, np.array([0.3, 0.7]), 'real_population')
        self.assertTrue(np.all(np.array([1, 0]) == out))
        out = sample_population(2, np.array([0.3, 0.7]), 'real_population')
        self.assertTrue(np.all(np.array([1, 1]) == out))
        out = sample_population(2, np.array([0.5, 0.5]), 'real_population')
        self.assertTrue(np.all(np.array([1, 1]) == out))

    def test_other_algos(self):
        default_algo = self.algo
        self.algo = 'monte_carlo'
        self.test_numbers()
        self.test_output_size()
        self.test_wrong_pdf()

        self.algo = default_algo

    def test_wrong_algo(self):
        self.assertRaises(
            NotImplementedError, sample_population,
            1, self.test_pdf, population_sampling_algo='wrong_algo'
            )


class MotherTest(unittest.TestCase):
    A = 2
    def test_lol(self):
        self.assertEqual(self.A, 2)

class Dauughter(MotherTest):
    pass
