import datetime
import unittest

import numpy as np

from demod.simulators.activity_simulators import (
    SubgroupsIndividualsActivitySimulator
)

from test_base_simulators import TimeAwareSimulatorChildrenTests

class SubgroupsIndividualsActivitySimulatorTests(TimeAwareSimulatorChildrenTests):
    sim = SubgroupsIndividualsActivitySimulator
    n_households = 3
    args = [
        [
            {'household_type': 2},
            {'household_type': 3, 'n_residents': 3},
            {'household_type': 4, 'n_residents': 5}
        ], [3, 2, 1]
    ]
    kwargs = {}
    args_step = []
    kwargs_step = []
    unimplemented_getters = []
    getter_args = {
        'get_n_doing_activity': [10],  # From a 4 states model as default
    }

    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    # From dataset
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)

    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError, super().test_non_default_step_size
        )

