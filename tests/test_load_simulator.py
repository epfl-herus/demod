import datetime
import unittest

# from .context import simulators
import numpy as np
from demod.simulators.load_simulators import LoadSimulator

from test_base_simulators import TimeAwareSimulatorChildrenTests


class TestLoadSimulator(
        TimeAwareSimulatorChildrenTests, unittest.TestCase
    ):
    sim = LoadSimulator
    args = [2]
    kwargs = {}
    args_step = []
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    n_households = 2
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)


    def test_get_demands(self):
        sim = LoadSimulator(*self.args, **self.kwargs)
        dhw = sim.get_dhw_heat_demand()
        sh = sim.get_sh_heat_demand()
        self.assertTrue(len(dhw)==self.n_households)
        self.assertTrue(len(sh)==self.n_households)
        self.assertTrue(isinstance(dhw, np.ndarray))
        self.assertTrue(isinstance(sh, np.ndarray))

    def test_accept_external_outputs(self):
        sim = LoadSimulator(*self.args, **self.kwargs)
        arr1 = np.array([1.,2.])
        arr2 = np.array([np.nan,2.])
        arr3 = np.array([3,4,5])
        sim.step(external_heat_outputs=arr1)
        sim.step(external_heat_outputs=arr2)
        sim.step(external_dhw_outputs=arr1)
        sim.step(external_dhw_outputs=arr2)
        sim.step(external_sh_outputs=arr1)
        sim.step(external_sh_outputs=arr2)
        # mix
        sim.step(external_dhw_outputs=arr1, external_sh_outputs=arr2)
        sim.step(
            external_heat_outputs=arr2,
            external_dhw_outputs=arr1,
            external_sh_outputs=arr2)
        # garbage input
        self.assertRaises(Exception, sim.step, external_heat_outputs=arr3)
        self.assertRaises(Exception, sim.step, external_dhw_outputs=arr3)
        self.assertRaises(Exception, sim.step, external_sh_outputs=arr3)

    def test_accept_external_cyl(self):
        sim = LoadSimulator(*self.args, **self.kwargs)
        arr1 = np.array([1., 2.])
        arr2 = np.array([np.nan, 2.])
        arr3 = np.array([3, 4, 5])
        sim.step(external_cylinder_temperature=arr1)
        sim.step(external_cylinder_temperature=arr2)
        self.assertRaises(
            Exception, sim.step, external_cylinder_temperature=arr3
        )

    def test_init(self):
        # check with defualt testing parameter
        sim = LoadSimulator(*self.args, **self.kwargs)
        # check with empty params
        sim = LoadSimulator(*self.args, **self.kwargs, include_climate=False)


    def test_climate_value_error(self):
        # check that the climate will raise it value error
        self.assertRaises(
            ValueError, LoadSimulator,
            *self.args, **self.kwargs,
            start_datetime=datetime.datetime.now()
        )

    def test_ignore_climate(self):
        # try to instantiate a simulator without climate
        sim = LoadSimulator(*self.args, **self.kwargs, include_climate=False)
        # check that climate simulator is not set
        self.assertIsNone(sim.climate)
        self.assertFalse(sim.include_climate)

        sim.step(
            external_outside_temperature=4.3, external_irradiance=50.3,
        )
        # check the error message when step is called without the two climate indicators
        self.assertRaisesRegex(
            ValueError, 'Inputs must be given for external_irradiance and temperature if Loadsimulator does not include a climate simulator.',
            sim.step)
        self.assertRaisesRegex(
            ValueError, 'Inputs must be given for external_irradiance and temperature if Loadsimulator does not include a climate simulator.',
            sim.step, external_outside_temperature=4.3,)
        self.assertRaisesRegex(
            ValueError, 'Inputs must be given for external_irradiance and temperature if Loadsimulator does not include a climate simulator.',
            sim.step, external_irradiance=50.3,)

    def test_initialization_time(self):
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 1)

    def test_initialization_time_over_day(self):
        sim = self.sim(
            *self.args, **self.kwargs,
            start_datetime=(
                self.default_start_datetime
                - datetime.timedelta(minutes=1)
            )
        )
        self.assertEqual(sim.current_time_step, 0)
