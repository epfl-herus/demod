
from demod.datasets.DESTATIS.loader import Destatis
import unittest





import numpy as np
import os




from demod.simulators.base_simulators import *
from demod.simulators.sparse_simulators import SubgroupsActivitySimulator, SparseTransitStatesSimulator

from test_simulators import SimulatorCommonTests

class SubSim(Simulator):
    def get_something(self):
        return np.array(self.n_households*[1])
    def get_something_else(self):
        return np.array(self.n_households*[2])
    def get_with_doc(self):
        """Oh wow"""
        return np.array(self.n_households*[2])

class UserMultiSim(MultiSimulator):
    def get_something(self):
        return 42

class UserMultiSim2(MultiSimulator):
    pass

class MultiSimulatorTests(unittest.TestCase):
    def test_creates_getter(self):

        sim = MultiSimulator([SubSim(1)])
        self.assertTrue(hasattr(sim, 'get_something'))
        self.assertTrue(np.all(sim.get_something() == np.array([1])))
        self.assertEqual(sim.get_something(0), 1)
        self.assertTrue(hasattr(sim, 'get_something_else'))
        self.assertTrue(np.all(sim.get_something_else() == np.array([2])))
        self.assertEqual(sim.get_something_else(0), 2)

    def test_created_getter_no_overriding(self):
        # Tests that when creating the getter, any user-defined getter won't
        # be overriden by the default getters
        sim = UserMultiSim([SubSim(1)])
        self.assertTrue(hasattr(sim, 'get_something'))
        # No overriding, result is the one of the Usersim
        self.assertEqual(sim.get_something(), 42)
        # Test the other is overridden
        self.assertTrue(hasattr(sim, 'get_something_else'))
        self.assertTrue(np.all(sim.get_something_else() == np.array([2])))
        self.assertEqual(sim.get_something_else(0), 2)

    def test_reconstruction_in_getter(self):
        # This test is wierd,
        # it fails when I run only it, but when the class runs, it fails
        sim2 = UserMultiSim2([SubSim(2), SubSim(1), SubSim(3)])
        smth = sim2.get_something()
        print('smth',smth)
        self.assertEqual(len(smth), 6)

    def test_reconstruct_with_doc(self):
        sim = UserMultiSim([SubSim(1)])
        get_method_multi = getattr(sim, 'get_with_doc')
        get_method_sub = getattr(sim.simulators[0], 'get_with_doc')
        self.assertEqual(
            get_method_sub.__doc__, get_method_multi.__doc__)

    def test_step(self):
        sim = MultiSimulator([SubSim(1), SubSim(1), SubSim(1)])
        sim.initialize_starting_state()
        self.assertEqual(sim.current_time_step, 0)
        for s in sim.simulators:
            s.initialize_starting_state()
            self.assertEqual(s.current_time_step, 0)
        sim.step()
        self.assertEqual(sim.current_time_step, 1)
        for s in sim.simulators:
            self.assertEqual(s.current_time_step, 1)








class MultisimulatorCommonTests(SimulatorCommonTests):


    def check_inheritance(self):
        self.assertTrue(issubclass(self.simulator_class, MultiSimulator))

    def check_logger_only_in_main(self, *args, **kwargs):
        self.sim = self.simulator_class(*args, logger = SimLogger('current_time_step'), **kwargs)
        self.assertIsInstance(self.sim.logger, SimLogger) # checks that main simulator loggs data
        for s in self.sim.simulators:
            self.assertTrue(s.logger is None)


class SubgroupsActivitySimulatorTests(MultisimulatorCommonTests):
    simulator_class = SubgroupsActivitySimulator

    def test_instantiation(self):
        #os.chdir( os.getcwd()[:-4] ) # change the directory
        self.check_instantiation([{'n_residents':2}], [10], subsimulator=SparseTransitStatesSimulator)

    def test_logger_only_in_main(self):
        self.check_logger_only_in_main([{'n_residents':2}], [10], subsimulator=SparseTransitStatesSimulator)

    def test_getters(self):
        subgroups, counts = self._get_householdtypes(20)
        sim = self.simulator_class(
            subgroups,
            counts,
            subsimulator=SparseTransitStatesSimulator,
        )
        sim.get_active_occupancy()

    def _get_householdtypes(self, n_households):

        # randomly assign the type of households based on the german data
        data = Destatis()
        subgroups, pdf, _ = data.load_population_subgroups('household_types')
        t = np.random.randint(0, len(subgroups), size=n_households)
        counts = np.bincount(t, minlength=len(subgroups))
        mask_counts = counts > 0
        return np.array(subgroups)[mask_counts], counts[mask_counts]

if __name__ == '__main__':

    unittest.main()
