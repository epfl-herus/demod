
import unittest
import sys, os
from datetime import datetime, timedelta, time


from demod.simulators.crest_simulators import CrestOccupancySimulator, Crest4StatesModel
from demod.datasets.GermanTOU.loader import GTOU

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
            self.test_clear_parsed_data(version)
            self.test_load_tpm(version)
            self.test_population(version)

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

    def test_washing_machines_loads(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        real_load_dict = data.load_real_profiles_dict('switchedON')

    def test_other_loaders(self, version=None):
        from demod.datasets.CREST.loader import Crest
        data = Crest() if version is None else Crest(version)
        data.load_crest_lighting()
        data.load_bulbs_config()

#   def test_clear_parsed_data(self, version=None):
#       from demod.datasets.CREST.loader import Crest
#       data = (
#           Crest(clear_parsed_data=True) if version is None
#           else Crest(version, clear_parsed_data=True)
#       )
#       self.test_population(version)
#       self.test_load_tpm(version)
#       self.test_washing_machines_loads(version)
#       self.test_load_activity_probability_profiles(version)
#
#
#       # self.test_ownership(version)
#       # self.test_load_appliances_dict(version)



class TestCrestOccupancySimulator(BaseSimulatorChildrenTests):
    sim = CrestOccupancySimulator  # Use the example Simulator, as Simulator cannot be call only by init and step
    args = [1, 3, 'd']  # n_households, max_residents, daytype
    kwargs = {}
    n_households = 1
    args_step = []  # no step inputs
    kwargs_step = {}
    unimplemented_getters = ['get_n_doing_state', 'get_performing_activity']
    getter_args = {
        'get_n_doing_activity': [10],  # From a 4 states model as default
    }

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
    unimplemented_getters = [
        'get_performing_activity',
        'get_n_doing_activity',
        'get_n_doing_state'
    ]
    getter_args = {
        'get_mask_subgroup': [{'n_residents': 2}],
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

    def test_with_GTOU(self):
        self.kwargs['data'] = GTOU()
        BaseSimulatorChildrenTests.run_base_tests(self)
        self.kwargs.pop('data')




if __name__ == '__main__':
    unittest.main()
