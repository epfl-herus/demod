
from demod.datasets.CREST.loader import Crest
from demod.datasets.Germany.loader import GermanDataHerus
import unittest




import sys
import numpy as np
import datetime





from demod.simulators.base_simulators import SimLogger, Simulator
from demod.simulators.appliance_simulators import SubgroupApplianceSimulator


from test_base_simulators import TimeAwareSimulatorChildrenTests

class TestApplianceDatasets(unittest.TestCase):
    def test_german_data(self):
        data = GermanDataHerus()
        app_dic = data.load_appliance_dict()
        print(app_dic)


class TestSubgroupApplianceSimulator(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    sim = SubgroupApplianceSimulator
    args = [ [{'n_residents': 2}], [1],]
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
        self.kwargs.pop('equipped_sampling_algo')

    def test_instantiation(self):
        super().test_instantiation()
        self.sim( [{'n_residents': 2}], [1], )
        self.sim( [{'n_residents': 2}, {'n_residents': 3}], [1,2],)
        self.sim( [{'n_residents': 2}], [1], equipped_sampling_algo='basic')
        self.sim( [{'n_residents': 2}], [1])

    def test_n_households(self):
        super().test_n_households()
        sim = self.sim([{'n_residents': 2}, {'n_residents': 3}], [1,2])
        self.assertEqual(sim.n_households, 3)

    def test_get_power_consumption(self):
        sim = self.sim(*self.args, **self.kwargs)
        c = sim.get_current_power_consumptions()
        # should test shape


class TestSubgroupApplianceSimulatorCrestData(TestSubgroupApplianceSimulator):
    """Version of test subgroup appliances, but uses CREST data.

    Was required as compatibility breaks due to different default start
    time.

    Args:
        TimeAwareSimulatorChildrenTests: [description]
        unittest: [description]
    """
    sim = SubgroupApplianceSimulator
    kwargs = {'data': Crest()}
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






if __name__ == '__main__':
    unittest.main()
