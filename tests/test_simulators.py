
import unittest





import numpy as np


from demod.helpers import *
from demod.simulators.sparse_simulators import *
from demod.simulators.crest_simulators import *
from demod.simulators.base_simulators import *
from demod.simulators.lighting_simulators import *


from datetime import datetime

class SimulatorCommonTests(unittest.TestCase):
    def test_(self):
        pass

    def check_instantiation(self, *args, **kwargs):
        self.sim = self.simulator_class( *args, **kwargs)
        self.assertIsInstance(self.sim, Simulator)

    def check_step(self, *args, **kwargs):
        self.sim.step(*args, **kwargs)

    def check_accept_logger(self, *args, **kwargs):
        self.sim = self.simulator_class( *args, logger=SimLogger('current_time_step'), **kwargs)


class OccupnancySimulatorsCommonTests(SimulatorCommonTests):
    def test_(self):
        pass
    def check_get_active_occupancy(self):
        act_occ = self.sim.get_active_occupancy()
        self.assertEqual(self.sim.n_households, len(act_occ))



class TestCRESTOccupancy(OccupnancySimulatorsCommonTests, unittest.TestCase):
    simulator_class = CrestOccupancySimulator
    sim = simulator_class(10, 1, 'e')
    def test_instatiations(self):

        self.check_instantiation(10,1,'e')
        self.check_instantiation(10,1,'d')

    def test_step(self):
        self.check_step()



class TestSparseSimulators(OccupnancySimulatorsCommonTests):
    # tests the differents helper functions on different set

    simulator_class = SparseTransitStatesSimulator
    sim = simulator_class(10, {'n_residents': 2} )
    def test_instatiations(self):

        self.check_instantiation(10, {'n_residents': 2})
        # test an empty dict
        with self.assertWarns(Warning):
            self.check_instantiation(10, {} ) # warns that no n_residents is specified


    def test_step(self):
        self.check_step()

    def test_9_state_simulator(self):
        n_households = 100

        subgroup =  {'n_residents': 2}

        sim  = SubgroupsActivitySimulator(
            [subgroup], [n_households],
            subsimulator=SparseTransitStatesSimulator)
        sim.step()

        subgroups =  [
            {'n_residents': 2},
            {
                'weekday':                          [6,7],
                'n_residents':                      1,
            },
            {
                'weekday':                          1,
                'quarter':                          2,
            }]

        # check housholds with 3 subgroups
        sim  = SubgroupsActivitySimulator(
            subgroups, 3*[n_households],
            subsimulator=SparseTransitStatesSimulator)
        sim.step()

    def test_accept_logger(self):
        self.check_accept_logger(10, {'n_residents': 2})

    def test_time_awareness(self):
        sim = SparseTransitStatesSimulator(
            2, {'n_residents':2},
            time_aware=True, start_datetime=datetime(2000, 1, 1, 5, 10, 0))
        # check initialization to correct time step
        self.assertEqual(sim.current_time_step, 7)
        # check can perform steps
        sim.step()
        self.assertEqual(sim.current_time_step, 8)
        self.assertEqual(sim.time.hour,6)
        self.assertEqual(sim.time.minute,30)

        # check passes through a multi
        sim  = SubgroupsActivitySimulator(
            [{'n_residents': 2}], [3],
            subsimulator=SparseTransitStatesSimulator,
            time_aware=True,
            start_datetime=datetime(2000, 1, 1, 5, 10, 0))
        sim.step()
        self.assertEqual(sim.simulators[0].current_time_step, 8)



if __name__ == '__main__':
    unittest.main()
