import unittest

import sys
import numpy as np
import datetime


from demod.datasets.CREST.loader import Crest
from demod.datasets.Germany.loader import GermanDataHerus
from demod.simulators.base_simulators import SimLogger, Simulator
from demod.simulators.appliance_simulators import (
    OccupancyApplianceSimulator,
    ActivityApplianceSimulator,
    ProbabiliticActivityAppliancesSimulator
)

from test_base_simulators import TimeAwareSimulatorChildrenTests

ONE_HH_ACT_DICT = {
    'sleeping': np.array([0]),
    'away': np.array([1]),
    'active_occupancy': np.array([1]),
    'ironing': np.array([1]),
    'cleaning': np.array([1]),
    'electronics': np.array([1]),
    'watching_tv': np.array([1]),
    'cooking': np.array([1]),
    'dishwashing': np.array([1]),
    'laundry': np.array([1]),
    'self_washing': np.array([1]),
}

class TestApplianceDatasets(unittest.TestCase):
    def test_german_data(self):
        data = GermanDataHerus()
        app_dic = data.load_appliance_dict()


class TestOccupancyApplianceSimulator(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    sim = OccupancyApplianceSimulator
    args = [[{'n_residents': 2}], [1]]
    kwargs = {}
    args_step = [np.array([1], dtype=int),]
    kwargs_step = {}
    unimplemented_getters = ['get_switchon_probs']
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_non_default_step_size(self):
        # cannot handle non default time step
        self.assertRaisesRegex(
            ValueError,
            'Step size must be 1 Minute for Appliance Simulator',
            self.sim, *self.args,
            start_datetime=self.default_start_datetime,
            step_size=self.random_step_size, **self.kwargs
        )

    def test_other_equipped_sampling_algo(self):
        self.kwargs['equipped_sampling_algo'] = 'basic'
        self.run_base_tests()
        self.kwargs['equipped_sampling_algo'] = 'all'
        self.run_base_tests()
        self.kwargs['equipped_sampling_algo'] = 'set_defined'
        self.kwargs['equipped_set_defined'] = [['FRIDGE1', 'FRIDGE2']]
        self.run_base_tests()
        self.kwargs.pop('equipped_sampling_algo')
        self.kwargs.pop('equipped_set_defined')

    def test_instantiation(self):
        super().test_instantiation()
        self.sim([{'n_residents': 2}], [1], )
        self.sim([{'n_residents': 2}, {'n_residents': 3}], [1,2],)
        self.sim([{'n_residents': 2}], [1], equipped_sampling_algo='basic')
        self.sim([{'n_residents': 2}], [1])

    def test_n_households(self):
        super().test_n_households()
        sim = self.sim([{'n_residents': 2}, {'n_residents': 3}], [1,2])
        self.assertEqual(sim.n_households, 3)

    def test_get_power_consumption(self):
        sim = self.sim(*self.args, **self.kwargs)
        c = sim.get_current_power_consumptions()
        # should test shape

    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError,
            self.sim, *self.args,
            start_datetime=self.default_start_datetime,
            step_size=self.random_step_size, **self.kwargs
        )

class TestOccupancyApplianceSimulatorCrestData(TestOccupancyApplianceSimulator):
    """Version of test subgroup appliances, but uses CREST data.

    Was required as compatibility breaks due to different default start
    time.

    Args:
        TimeAwareSimulatorChildrenTests: [description]
        unittest: [description]
    """
    sim = OccupancyApplianceSimulator
    kwargs = {
        'data': Crest(),
        'real_profiles_algo': 'only_switched_on'  # Crest has only switched on profiles
    }
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0)
    default_initialization_time = datetime.time(0, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_step_timestep(self):
        # needs to set the start of the simulation first
        self.kwargs['start_datetime'] = self.default_start_datetime
        super().test_step_timestep()
        self.kwargs.pop('start_datetime')

    def test_instantiation_with_default_datetime(self):
        # needs to set the start of the simulation first
        self.kwargs['start_datetime'] = self.default_start_datetime
        super().test_instantiation_with_default_datetime()
        self.kwargs.pop('start_datetime')




class TestActivityApplianceSimulator(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    sim = ActivityApplianceSimulator
    args = [1, ONE_HH_ACT_DICT]
    kwargs = {}
    args_step = [ONE_HH_ACT_DICT]
    kwargs_step = {}
    unimplemented_getters = ['get_switchon_probs']
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_initialization_time(self):
        n_steps = 3
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime
                + n_steps*self.default_step_size
            ),
            **self.kwargs)
        # Can be initialized at any time
        self.assertEqual(sim.current_time_step, 0)

    def test_initialization_time_over_day(self):
        n_steps = 1
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime - n_steps*self.default_step_size
            ),
            **self.kwargs)
        expected_steps = 0  # Can be initialized at any time
        self.assertEqual(sim.current_time_step, expected_steps)
        self.assertEqual(
            sim.current_time,
            self.default_start_datetime - n_steps*self.default_step_size
        )

    def test_algos_real_loads(self):
        self.kwargs['real_profiles_algo'] = 'nothing'
        self.run_base_tests()
        self.kwargs['real_profiles_algo'] = 'only_always_on'
        self.run_base_tests()
        self.kwargs['real_profiles_algo'] = 'uniform'
        self.run_base_tests()
        self.kwargs.pop('real_profiles_algo')

    def test_init_with_subgroups(self):
        self.kwargs['subgroups_list'] = [{'n_residents': 2}]
        self.kwargs['n_households_list'] = [1]
        self.run_base_tests()
        self.kwargs.pop('subgroups_list')
        self.kwargs.pop('n_households_list')


class TestProbabiliticActivityAppliancesSimulator(
    TimeAwareSimulatorChildrenTests, unittest.TestCase
):
    sim = ProbabiliticActivityAppliancesSimulator
    args = [1, ONE_HH_ACT_DICT]
    kwargs = {}
    args_step = [ONE_HH_ACT_DICT]
    kwargs_step = {}
    unimplemented_getters = ['get_switchon_probs']
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_algos_real_loads(self):
        self.kwargs['real_profiles_algo'] = 'nothing'
        self.run_base_tests()
        self.kwargs['real_profiles_algo'] = 'only_always_on'
        self.run_base_tests()
        self.kwargs['real_profiles_algo'] = 'uniform'
        self.run_base_tests()
        self.kwargs.pop('real_profiles_algo')

    def test_init_with_subgroups(self):
        self.kwargs['subgroups_list'] = [{'n_residents': 2}]
        self.kwargs['n_households_list'] = [1]
        self.run_base_tests()
        self.kwargs.pop('subgroups_list')
        self.kwargs.pop('n_households_list')

    def test_initialization_time(self):
        n_steps = 3
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime
                + n_steps*self.default_step_size
            ),
            **self.kwargs)
        # Can be initialized at any time
        self.assertEqual(sim.current_time_step, 0)

    def test_initialization_time_over_day(self):
        n_steps = 1
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime - n_steps*self.default_step_size
            ),
            **self.kwargs)
        expected_steps = 0  # Can be initialized at any time
        self.assertEqual(sim.current_time_step, expected_steps)
        self.assertEqual(
            sim.current_time,
            self.default_start_datetime - n_steps*self.default_step_size
        )

if __name__ == '__main__':
    unittest.main()
