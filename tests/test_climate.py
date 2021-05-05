import sys
from demod.datasets.OpenPowerSystems.loader import OpenPowerSystemClimate
import unittest
import datetime

from test_base_simulators import TimeAwareSimulatorChildrenTests
from demod.simulators.weather_simulators import CrestIrradianceSimulator, RealInterpolatedClimate, RealClimate, CrestClimateSimulator



class CrestIrradianceSimulatorTests(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    """Tests for CrestIrradianceSimulator."""
    sim = CrestIrradianceSimulator # Use the example Simulator, as Simulator cannot be call only by init and step
    args = []  # no argument required
    kwargs = {}
    n_households = 1
    args_step = [] #
    kwargs_step = {}
    unimplemented_getters = ['get_outside_temperature']
    getter_args = {} # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0)
    default_initialization_time = datetime.time(0, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)
    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError,
            super().test_non_default_step_size
        )
    def test_initial_clearness(self):
        self.kwargs['initial_clearness'] = 0.42
        self.run_base_tests()
        self.kwargs['initial_clearness'] = 1.
        self.run_base_tests()
        self.kwargs.pop('initial_clearness')


class CrestClimateSimulatorTests(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    """Tests for Crest Climate simulator."""
    sim = CrestClimateSimulator
    args = []  # no argument required
    kwargs = {}
    n_households = 1
    args_step = [] #
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {} # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0)
    default_initialization_time = datetime.time(0, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)
    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError,
            super().test_non_default_step_size
        )
    def test_initial_clearness(self):
        self.kwargs['initial_clearness'] = 0.42
        self.run_base_tests()
        self.kwargs.pop('initial_clearness')


class TestClimateLoader(unittest.TestCase):
    def test_loading(self):
        data = OpenPowerSystemClimate('germany')
        data.load_historical_climate_data(datetime.datetime(2014, 1, 1))

    def test_loading_aware_datetime(self):
        data = OpenPowerSystemClimate('germany')
        start_datetime = datetime.datetime(
            2014, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
        )
        print(data.load_historical_climate_data(start_datetime ))

class RealClimateTests(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    """Tests for real climate simulator."""
    sim = RealClimate
    args = []  # no argument required
    kwargs = {}
    n_households = 1
    args_step = [] #
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {} # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters. they depend on the default dataset (germany)
    default_start_datetime = datetime.datetime(1980, 1, 1, 0, 0, 0)
    default_initialization_time = datetime.time(0, 0, 0)
    default_step_size = datetime.timedelta(hours=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_non_default_step_size(self):
        self.assertRaises(
            ValueError,
            super().test_non_default_step_size
        )

    def test_other_data(self):
        self.kwargs['data'] = OpenPowerSystemClimate('switzerland')
        self.run_base_tests()
        self.kwargs.pop('data')

    def test_initialization_time(self):

        n_steps = 3
        sim = self.sim(
            *self.args, start_datetime= self.default_start_datetime + n_steps*self.default_step_size,
            **self.kwargs)
        # Climate data intialize directly a the current_time
        self.assertEqual(sim.current_time_step, 0)

    def test_initialization_time_over_day(self):
        n_steps = 1
        sim = self.sim(
            *self.args,
            start_datetime = (
                self.default_start_datetime
                - n_steps*self.default_step_size
                # Extra day, because data starts the day
                # of default_start_datetime
                + datetime.timedelta(days=1)
            ),
            **self.kwargs)
        # Climate data intialize directly a the current_time
        self.assertEqual(sim.current_time_step, 0)
        self.assertEqual(
            sim.current_time,
            (
                self.default_start_datetime
                - n_steps*self.default_step_size
                + datetime.timedelta(days=1)
            )
        )

    def test_invalid_day(self):
        """Test what happens when the day is not in the dataset.
        """
        self.assertRaises(
            ValueError,
            self.sim,
            *self.args,
            start_datetime = datetime.datetime(1969, 7, 20, 20, 17),
            **self.kwargs
        )
        self.assertRaises(
            ValueError,
            self.sim,
            *self.args,
            start_datetime = datetime.datetime.now(),
            **self.kwargs
        )




class RealInterpolatedClimateTests(TimeAwareSimulatorChildrenTests, unittest.TestCase):
    """Tests for real climate simulator that uses interpolation."""
    sim = RealInterpolatedClimate
    args = []  # no argument required
    kwargs = {}
    n_households = 1
    args_step = [] #
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {} # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters. they depend on the default dataset (germany)
    default_start_datetime = datetime.datetime(1980, 1, 1, 0, 0, 0)
    default_initialization_time = datetime.time(0, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_other_data(self):
        self.kwargs['data'] = OpenPowerSystemClimate('switzerland')
        self.run_base_tests()
        self.kwargs.pop('data')

    def test_other_interpolation(self):
        self.kwargs['interpolation_kind'] = 3
        self.run_base_tests()
        self.kwargs['interpolation_kind'] = 'cubic'
        self.run_base_tests()
        self.kwargs.pop('interpolation_kind')

    def test_initialization_time(self):

        n_steps = 3
        sim = self.sim(
            *self.args, start_datetime= self.default_start_datetime + n_steps*self.default_step_size,
            **self.kwargs)
        # Climate data intialize directly a the current_time
        self.assertEqual(sim.current_time_step, 0)

    def test_initialization_time_over_day(self):
        n_steps = 1
        sim = self.sim(
            *self.args,
            start_datetime = (
                self.default_start_datetime
                - n_steps*self.default_step_size
                # Extra day, because data starts the day
                # of default_start_datetime
                + datetime.timedelta(days=1)
            ),
            **self.kwargs)
        # Climate data intialize directly a the current_time
        self.assertEqual(sim.current_time_step, 0)
        self.assertEqual(
            sim.current_time,
            (
                self.default_start_datetime
                - n_steps*self.default_step_size
                + datetime.timedelta(days=1)
            )
        )

    def test_invalid_day(self):
        """Test what happens when the day is not in the dataset.
        """
        self.assertRaises(
            ValueError,
            self.sim,
            *self.args,
            start_datetime = datetime.datetime(1969, 7, 20, 20, 17),
            **self.kwargs
        )
        self.assertRaises(
            ValueError,
            self.sim,
            *self.args,
            start_datetime = datetime.datetime.now(),
            **self.kwargs
        )

    def test_end_of_data(self):
        """Test the behaviour when arriving at the end of the data."""
        # Sets a huge step size to ensure fail.
        sim = self.sim(step_size=datetime.timedelta(days=1e6))
        self.assertRaises(
            ValueError,
            sim.step,
        )
    def test_getters_with_tzinfo(self):
        """Test instantiation with a non-naive datetime object.

        A non-naive datetime object holds a tzinfo,
        which gives information on the time-zone and daylight saving
        times.
        """
        # check using random time
        new_time = self.default_start_datetime
        new_time = new_time.replace(tzinfo=datetime.timezone.utc)
        sim = self.sim(*self.args, start_datetime=new_time, **self.kwargs)
        self.assertEqual(sim.current_time, new_time)
        old_T = sim.get_outside_temperature()
        sim.step()
        self.assertEqual(sim.current_time, new_time+self.default_step_size)
        self.assertNotEqual(sim.get_outside_temperature(), old_T)

if __name__ == '__main__':
    unittest.main()