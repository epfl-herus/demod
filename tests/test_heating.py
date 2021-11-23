import unittest



import os
import sys
import numpy as np
import datetime



from demod.datasets.Germany.loader import GermanDataHerus
from demod.simulators.base_simulators import SimLogger, Simulator
from demod.simulators.heating_simulators import (
    Thermostats,
    HeatingSystem,
    BuildingThermalDynamics,
    HeatDemand,
    SystemControls,
    FiveModulesHeatingSimulator,

)


from test_base_simulators import BaseSimulatorChildrenTests

class TestDataLoading(unittest.TestCase):
    def test_german_data_herus(self):
        data = GermanDataHerus()
        data.load_buildings_dict()
        data.load_heating_system_dict()
        data.load_thermostat_dict()

class ThermostatsTest(BaseSimulatorChildrenTests, unittest.TestCase):
    sim = Thermostats
    args = [2] # n_households,
    n_households = 2
    kwargs = {
        'initial_temperatures': {
            'cylinder': np.array([50, 50-10]),
            'interior': np.array([20, 20-10]),
            'emitter':  np.array([40, 40-10]),
        }
    }
    args_step = [{
            'cylinder': np.array([50, 50-10]),
            'interior': np.array([20, 20-10]),
            'emitter':  np.array([40, 40-10]),
    }]
    kwargs_step = {}
    def test_instantiation_without_temperatures(self):
        sim = self.sim(1)
        states = sim.get_thermostat_states()
        for s in states.values():
            print(s)
            self.assertFalse(np.all(s))

class HeatingSystemTests(BaseSimulatorChildrenTests, unittest.TestCase):
    sim = HeatingSystem
    args = [2]
    kwargs = {}
    args_step = [
        {  # Controls
            'heater_on': np.array([True, False]),
            'heat_water_on': np.array([True, False]),
            'space_heating_on': np.array([True, False]),
        },
        {  # target_heat_demand
            'dhw': np.array([50, 50-10]),
            'space_heating': np.array([50, 50-10]),
        }

    ]
    kwargs_step = {}
    n_households = 2

    def test_initalization_with_outputs(self):
        self.kwargs['initial_heat_demand'] = {
            'dhw': np.array([0.42, 0.42]),
            'space_heating': np.array([0.42, 0.42])
        }
        self.kwargs['initial_controls'] = {
            'heater_on': np.array([True, False]),
            'heat_water_on': np.array([True, False]),
            'space_heating_on': np.array([True, False]),
        }
        self.run_base_tests()
        self.kwargs.pop('initial_heat_demand')
        self.kwargs.pop('initial_controls')

    def test_heat_correctly(self):
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)
        h_outputs = sim.get_heat_outputs()
        self.assertTrue(np.all(
            (
                self.args_step[1]['dhw']
                * self.args_step[0]['heat_water_on']
                + self.args_step[1]['space_heating']
                * self.args_step[0]['space_heating_on']
            ) * self.args_step[0]['heater_on']
            == (
                h_outputs['total']
            )
        ))

    def test_caps_too_large(self):
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(
            {  # Controls
                'heater_on': np.array([True, True]),
                'heat_water_on': np.array([True, False]),
                'space_heating_on': np.array([True, True]),
            },
            {  # target_heat_demand
                'dhw': np.array([1e10, 0.]),
                'space_heating': np.array([1e10, 1e10]),
            }
        )
        h_outputs = sim.get_heat_outputs()
        # Test capped
        self.assertEqual(h_outputs['dhw'][0], sim.max_Phi_h_dhw[0])
        # Test prioritize dhw
        self.assertEqual(h_outputs['space_heating'][0], 0.)
        # Test capped space heating only
        self.assertEqual(h_outputs['space_heating'][1], sim.max_Phi_h_space[1])




class BuildingThermalDynamicsTests(BaseSimulatorChildrenTests):
    sim = BuildingThermalDynamics
    args = [2, HeatingSystem(2), 5.3]
    kwargs = {}
    args_step = [
        4.2, 142,
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
        {  # heat outputs
            'space_heating': np.array([42.1, 42.1]),
            'dhw': np.array([42.1, 42.1]),
        },
    ]
    kwargs_step = {}
    n_households = 2

    def test_initialization_with_target_temperatures(self):
        self.kwargs['target_temperatures'] = np.array([21, 21])
        self.run_base_tests()
        self.kwargs.pop('target_temperatures')


    def test_initialization_with_different_step_size(self):
        self.kwargs['step_size'] = datetime.timedelta(hours=1)
        self.run_base_tests()
        self.kwargs.pop('step_size')

class HeatDemandTests(BaseSimulatorChildrenTests):
    sim = HeatDemand
    args = [
        2,
        HeatingSystem(2),
        BuildingThermalDynamics(2, HeatingSystem(2), 5.3),
    ]
    kwargs = {}
    args_step = [
        {
            'cylinder': np.array([50, 50-10]),
            'interior': np.array([20, 20-10]),
            'emitter':  np.array([40, 40-10]),
            'cold_water': 10.2
        },
        {
            'space_heating': np.array([50, 50-10]),
            'dhw':  np.array([40, 40-10]),
            'emitter':  np.array([40, 40-10]),

        },
        np.array([12.3, 42.1]),
    ]
    kwargs_step = {
        'outside_temperature': 4.2
    }
    n_households = 2
    def test_other_algos(self):
        self.kwargs['heatdemand_algo'] = 'room_estimation'
        self.run_base_tests()
        self.kwargs.pop('heatdemand_algo')

class SystemControlsTests(BaseSimulatorChildrenTests):
    sim = SystemControls
    args = [2, np.array([True, False])]
    kwargs = {}
    args_step = [
        np.array([0., 42.]),
        {
            'hot_water': np.array([True, False]),
            'space_heating': np.array([True, False]),
            'emitter': np.array([True, False]),
        },
    ]
    kwargs_step = {}
    n_households = 2

    def test_external_cylinder(self):
        """Test we can use the external cyclinder in the step function.
        """
        self.kwargs_step['has_external_cylinder'] = np.array([True, False])
        self.run_base_tests()
        self.kwargs_step.pop('has_external_cylinder')

    def test_combi_boiler_assignement(self):
        """Test that the combi boilers are correctly assigned.
        """
        # test correct assignement
        sim: SystemControls = self.sim(*self.args, **self.kwargs)
        print(sim.is_combi_boiler)
        self.assertTrue(np.all(sim.is_combi_boiler == self.args[1]))

        # test default to false
        sim: SystemControls = self.sim(2)
        self.assertTrue(np.all(~sim.is_combi_boiler))

    def test_step_behaviour(self):
        sim: SystemControls = self.sim(*self.args, **self.kwargs)
        dhw_demand = np.ones(2)
        no_dhw_demand = np.zeros(2)

        thermostats_states_all_on = {
            'hot_water': np.array([True, True]),
            'space_heating': np.array([True, True]),
            'emitter': np.array([True, True]),
        }
        thermostats_states_all_off = {
            'hot_water': np.array([False, False]),
            'space_heating': np.array([False, False]),
            'emitter': np.array([False, False]),
        }
        thermostats_states_space_but_not_emitter = {
            'hot_water': np.array([False, False]),
            'space_heating': np.array([True, True]),
            'emitter': np.array([False, False]),
        }

        sim.step(
            no_dhw_demand,
            thermostats_states_all_on
        )
        controls = sim.get_controls()

        self.assertTrue(
            np.all(controls['space_heating_on'])
            & np.all(controls['heater_on'])
        )
        self.assertTrue(
            # Combi boiler is off, normal is on to heat the cyl
            np.all(controls['heat_water_on'] == ~sim.is_combi_boiler)
        )

        # All should be on
        sim.step(
            dhw_demand,
            thermostats_states_all_on
        )
        controls = sim.get_controls()
        self.assertTrue(
            np.all(controls['space_heating_on'])
            & np.all(controls['heater_on'])
            & np.all(controls['heat_water_on'])
        )

        # Test with off thermostats
        sim.step(
            dhw_demand,
            thermostats_states_all_off
        )
        controls = sim.get_controls()

        self.assertTrue(
            np.all(~controls['space_heating_on'])
        )
        self.assertTrue(
            np.all(controls['heater_on'] == sim.is_combi_boiler)
            & np.all(controls['heat_water_on'] == sim.is_combi_boiler)
        )

        # Test when emitters cannot be heated more
        sim.step(
            no_dhw_demand,
            thermostats_states_space_but_not_emitter
        )
        controls = sim.get_controls()

        self.assertTrue(
            np.all(~controls['space_heating_on'])
            & np.all(~controls['heater_on'])
            & np.all(~controls['heat_water_on'])
        )



class HeatLoadSimulatorTests(BaseSimulatorChildrenTests):
    sim = FiveModulesHeatingSimulator
    args = [2, 5.3]
    kwargs = {}
    args_step = [
        4.2, 142,
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
        np.array([42.1, 42.1]),
    ]
    kwargs_step = {}
    n_households = 2

    def test_other_algos(self):
        self.kwargs['heatdemand_algo'] = 'room_estimation'
        self.run_base_tests()
        self.kwargs.pop('heatdemand_algo')


    def test_initialization_with_different_step_size(self):
        self.kwargs['step_size'] = datetime.timedelta(hours=1)
        self.run_base_tests()
        self.kwargs.pop('step_size')

    def test_accept_external_target_temperature(self):
        self.kwargs_step['external_target_temperature'] = np.array([13, 22])
        self.run_base_tests()
        # check the value is well assigned
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertTrue(np.all(
            sim.thermostats_sim.get_target_temperatures()['space_heating']
            == self.kwargs_step['external_target_temperature']
        ))

        self.kwargs_step['external_target_temperature'] = {
            'space_heating': np.array([13, 22]),
            'dhw':  np.array([40, 40-10]),
            'emitter':  np.array([40, 40-10]),
        }
        self.run_base_tests()
        # check the value is well assigned
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertTrue(np.all(
            sim.thermostats_sim.get_target_temperatures()['space_heating']
            == self.kwargs_step['external_target_temperature']['space_heating']
        ))
        self.assertTrue(np.all(
            sim.thermostats_sim.get_target_temperatures()['emitter']
            == self.kwargs_step['external_target_temperature']['emitter']
        ))
        self.assertTrue(np.all(
            sim.thermostats_sim.get_target_temperatures()['dhw']
            == self.kwargs_step['external_target_temperature']['dhw']
        ))

        # test half complete dict
        self.kwargs_step['external_target_temperature'] = {
            'space_heating': np.array([np.nan, 22]),
            'emitter':  np.array([40, np.nan]),
        }
        self.run_base_tests()

        self.kwargs_step.pop('external_target_temperature')
