

import unittest
import numpy as np
from demod.simulators.lighting_simulators import (
    FisherLighitingSimulator,
    CrestLightingSimulator
)
from demod.datasets.CREST.loader import Crest
from demod.datasets.Germany.loader import GermanDataHerus
from test_base_simulators import BaseSimulatorChildrenTests


class TestDatasets(unittest.TestCase):
    def test_crest_dataset(self):
        data = Crest()
        bulb_config = data.load_bulbs_config()
        self.assertEqual(bulb_config.shape, (100, 38))
        light_dic = data.load_crest_lighting()

    def test_german_data(self):
        data = GermanDataHerus()
        light_dic = data.load_crest_lighting()
        bulbs = data.load_bulbs()
        print(data.load_installed_bulbs_stats())




class FisherLightingTests(BaseSimulatorChildrenTests):
    sim = FisherLighitingSimulator
    args = [1]  # n_households
    kwargs = {}
    n_households = 1
    args_step = [np.array([1]), 50]  # two step inputs
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]

    def test_calculate_light_usage_value(self):
        sim = self.sim(*self.args, **self.kwargs)
        self.assertEqual(sim.calculate_light_usage_value(10), 1)
        self.assertEqual(sim.calculate_light_usage_value(30), 1)
        self.assertEqual(sim.calculate_light_usage_value(45), 0.5)
        self.assertEqual(sim.calculate_light_usage_value(60), 0)
        self.assertEqual(sim.calculate_light_usage_value(80), 0)

    def test_all_possible_active_occupants(self):
        sim = self.sim(7)
        sim.step(np.array([0, 1, 2, 3, 4, 5, 6]), 42.42)

    def test_no_active_occupants(self):
        """0 occupants will return 0 consumption."""
        sim = self.sim(7)
        sim.step(np.array([0, 0, 0, 0 ,0, 0, 0]), 1000)
        self.assertTrue(np.all(
            0 == sim.get_power_consumption()
        ))


class CrestLightingSimulatorTests(BaseSimulatorChildrenTests):
    sim = CrestLightingSimulator
    args = [3] # n_households
    kwargs = {}
    n_households = 3
    args_step = [np.array([1, 2, 0]), 50]  # two step inputs
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    def test_sampling_algo(self):
        self.kwargs['bulbs_sampling_algo'] = 'randn'
        self.kwargs['data'] = GermanDataHerus()
        self.run_base_tests()
        self.kwargs.pop('bulbs_sampling_algo')
        self.kwargs.pop('data')