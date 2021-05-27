import datetime
import unittest

import numpy as np

from demod.simulators.activity_simulators import (
    SemiMarkovSimulator,
    SubgroupsIndividualsActivitySimulator
)

from test_base_simulators import TimeAwareSimulatorChildrenTests

class SubgroupsIndividualsActivitySimulatorTests(TimeAwareSimulatorChildrenTests):
    sim = SubgroupsIndividualsActivitySimulator
    n_households = 6
    args = [
        [
            {'household_type': 2},
            {'household_type': 3, 'n_residents': 3},
            {'household_type': 4, 'n_residents': 5}
        ], [3, 2, 1]
    ]
    kwargs = {}
    args_step = []
    kwargs_step = {}
    unimplemented_getters = [
        'get_active_occupancy', 'get_occupancy', 'get_mask_subgroup',
        'get_n_doing_state', 'get_thermal_gains'
    ]
    getter_args = {
        'get_n_doing_activity': ['sleeping'],
        'get_performing_activity': ['sleeping'],
    }

    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    # From dataset
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=10)

    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError, super().test_non_default_step_size
        )

    def test_with_semi_markov(self):
        self.kwargs['subsimulator'] = SemiMarkovSimulator
        self.run_base_tests()
        self.kwargs.pop('subsimulator')

