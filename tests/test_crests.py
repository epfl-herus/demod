
import unittest
import sys, os
from datetime import datetime, timedelta, time


from demod.simulators.crest_simulators import CrestOccupancySimulator, Crest4StatesModel

from test_base_simulators import BaseSimulatorChildrenTests, TimeAwareSimulatorChildrenTests


import numpy as np



class TestDataLoader(unittest.TestCase):
    def test_load_tpm(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        for n_res in [1, 2, 3, 4, 5, 6]:
            for day_type in [[1,2,3,4,5], [6, 7]]:
                subgroup = {'n_residents': n_res, 'weekday':day_type}
                tpm, labels, start_pdf = data.load_tpm(subgroup)
                n_states = int((n_res + 1) ** 2)
                self.assertEqual(tpm.shape, (144, n_states, n_states))
                self.assertEqual(len(labels), n_states)
                self.assertEqual(len(start_pdf), n_states)

    def test_load_appliances_dict(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)

        app_dict = data.load_appliance_dict()


    def test_load_activity_probability_profiles(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        for n_res in [1, 2, 3, 4, 5, 6]:
            for day_type in [[1,2,3,4,5], [6, 7]]:
                subgroup = {'n_residents': n_res, 'weekday':day_type}
                app_dict = data.load_activity_probability_profiles(subgroup)


    def test_new_versions(self):
        old_versions = ['2.3.3']
        for version in old_versions:
            self.test_load_tpm(version)

    def test_ownership(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        ownership_dict = data.load_appliance_ownership_dict()
        from demod.utils.appliances import get_ownership_from_dict
        values = get_ownership_from_dict(
            {'type': ['chest_freezer', 'tv', 'tv', 'tv']},
            ownership_dict
        )
        self.assertTrue(
            np.all(values == np.array([0.163, 0.977, 0.58,  0.18]))
        )

    def test_population(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        data.load_population_subgroups(
            'resident_number'
        )



class TestCrestOccupancySimulator(BaseSimulatorChildrenTests):
    sim = CrestOccupancySimulator  # Use the example Simulator, as Simulator cannot be call only by init and step
    args = [1, 3, 'd']  # n_households, max_residents, daytype
    kwargs = {}
    n_households = 1
    args_step = []  # no step inputs
    kwargs_step = {}
    unimplemented_getters = ['get_performing_activity']

    def test_day_types(self):
        args = self.args.copy()
        args[2] = 'e'
        self.sim(*args)


class TestCrestMultiOccupancySimulator(
        TimeAwareSimulatorChildrenTests,
        unittest.TestCase):
    default_start_datetime = datetime(2014, 1, 1, 0, 0, 0)
    default_initialization_time = time(0, 0, 0)
    default_step_size = timedelta(minutes=10)
    sim = Crest4StatesModel
    args = [1]  # n_households, max_residents
    kwargs = {}
    args_step = []  # two step inputs
    kwargs_step = {}
    unimplemented_getters = ['get_performing_activity']
    getter_args = {
        'get_mask_subgroup': [{'n_residents': 2}]
    }  # dic of the form "get_name_of_getter": [*args]

    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError, self.sim,
            *self.args,
            start_datetime=self.default_start_datetime,
            step_size=self.random_step_size, **self.kwargs)


    def test_accept_sampling_algorithm(self):
        kwargs = self.kwargs.copy()
        kwargs['population_sampling_algo'] = 'monte_carlo'
        self.sim(*self.args, **kwargs)




if __name__ == '__main__':
    unittest.main()
