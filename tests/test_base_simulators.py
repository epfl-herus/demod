from datetime import datetime, timedelta, time, timezone
from typing import Any, Dict, List
import unittest

from demod.simulators.base_simulators import ExampleSimulator, SimLogger, Simulator, TimeAwareExampleSimulator, TimeAwareSimulator, before_next_day_callback, after_next_day_callback, before_next_day_callback_4am, after_next_day_callback_4am



class BaseSimulatorChildrenTests(unittest.TestCase):
    """Helper Class for testing children of the base simulator.

    This class can be used as parent to test the behaviour of any
    :py:class:`Simulator`.

    Attributes:
        sim: The simuulator class that should be tested.
        n_households: The number of households that are supposed to
            be simulated in the test.
        args: The args used for the :py:meth:`__init__` method.
        kwargs: The kwargs used for the :py:meth:`__init__` method.
        args_step: Example args used for the :py:meth:`step` method
        kwargs_step: Example kwargs used for the :py:meth:`step` method
        unimplemented_getters: A list of any getter names, that should
            not have their behaviour tested by TestCase.
        getter_args: A dictionary that specifies some arguments that
            should be passed to specific getter funcitons.
            Of the form: {"get_name_of_getter": [*args], ...}

    """
    # Use the example Simulator,
    # as Simulator cannot be call only by init and step
    sim: Simulator = ExampleSimulator
    n_households: int = 1
    args: List[Any] = [1, 3] # n_households, max_residents
    kwargs: Dict[str, Any] = {}
    args_step: List[Any] = [1, -1]  # two step inputs
    kwargs_step: Dict[str, Any] = {}
    unimplemented_getters: List[str] = []
    getter_args: Dict[str, List[Any]]  = {}

    def get_instantiated_sim(self, *args, **kwargs):
        return self.sim(*args, **kwargs)

    def test_instantiation(self):
        """Test that the simulator can be instantiated with args and kwargs."""
        sim = self.sim(*self.args, **self.kwargs)

    def test_n_households(self):
        """Test that the number of households is well assigned."""
        sim = self.sim(*self.args, **self.kwargs)
        self.assertEqual(sim.n_households, self.n_households)

    def test_step(self):
        """Test that the simulator can perform a step."""
        sim = self.sim(*self.args, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)

    def test_step_timestep(self):
        """Test that the simulator updates the :py:attr:`current_time_step`
        when doing a step.
        """
        sim = self.sim(*self.args, **self.kwargs)
        # Time step after initialization
        current_step = sim.current_time_step
        sim.step(*self.args_step, **self.kwargs_step)
        # Checks did one and only one step
        self.assertEqual(sim.current_time_step, current_step + 1)

    def test_accept_logger(self):
        """Test that the simulator can accept a :py:class:`SimLogger`."""
        sim = self.sim(
            *self.args, **self.kwargs, logger=SimLogger('n_households')
        )
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(type(sim.logger), SimLogger)

    def test_getter_function_work(self):
        """Test all getter functions available in the simulator.

        Reads the attributes of the simulator to find the getters.
        If a getter should be ignored in the test, it can be
        specifies in the TestCase :py:attr:`unimplemented_getters`
        attribute.
        """
        sim = self.sim(*self.args, **self.kwargs)
        # finds out the getters methods
        getters = [
            method for method in dir(sim)
            if method.startswith('get_') and
            (method not in self.unimplemented_getters)
        ]
        for get_name in getters:
            get_method = getattr(sim, get_name)
            # check if getter needs argument
            if get_name in self.getter_args:
                get_method(*self.getter_args[get_name])
            else:
                get_method()

    def run_base_tests(self):
        """Run all basic tests.

        Useful if you want to run all test with different parameters.
        """
        self.test_instantiation()
        self.test_n_households()
        self.test_step()
        self.test_step_timestep()
        self.test_accept_logger()
        self.test_getter_function_work()




class TimeAwareSimulatorChildrenTests(BaseSimulatorChildrenTests):
    """Base tests for all children of TimeAwareSimulators.

    This method also has all the attributes from
    :py:class:`.BaseSimulatorChildrenTests`.

    Attributes:
        default_start_datetime: The date_time at which the simulators
            starts by default.
        default_initialization_time: The initialization time that is
            used by the simulator.
        default_step_size: The default step_size assigned to the
            simulator.
        random_time: A random date_time value that is used for some
            tests.
        random_step_size: A random step_size value that is used for
            some test.
        sim: The TimeAwareSimulator that should be tested.
    """

    default_start_datetime: datetime = datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time: time = time(4, 0, 0)
    default_step_size: timedelta = timedelta(minutes=1)
    random_time: datetime = datetime(2008, 4, 5, 13, 42, 26)
    random_step_size: timedelta = timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    sim: TimeAwareSimulator = TimeAwareExampleSimulator
    n_households: int = 1
    args: List[Any] = [1, 3]  # n_households, max_residents
    kwargs: Dict[str, Any] = {}
    args_step: List[Any] = [1, -1]  # two step inputs
    kwargs_step: Dict[str, Any] = {}
    unimplemented_getters: List[str] = []
    getter_args: Dict[str, List[Any]] = {}

    def test_same_time_init_start_in_test(self):
        """Test for the TestCase default values.

        Test that the :py:attr:`default_start_datetime`
        and the :py:attr:`default_initialization_time`
        have the same time value in the TestCase values.
        """
        self.assertEqual(
            self.default_start_datetime.time(),
            self.default_initialization_time,
            "Your test must have the same time in the defaults: \
            'default_initialization_time' and 'default_start_datetime'")

    def test_instantiation_with_default_datetime(self):
        """Test that sim is initialized at default start_datetime."""
        # check using datetime object
        sim = self.sim(*self.args, **self.kwargs)
        self.assertEqual(sim.current_time, self.default_start_datetime)

    def test_instantiation_with_other_datetime(self):
        """Test instantiation at another datetime."""
        # check using random time
        sim = self.sim(
            *self.args, start_datetime=self.random_time, **self.kwargs)
        self.assertEqual(sim.current_time, self.random_time)

    def test_instantiation_with_tzinfo(self):
        """Test instantiation with a non-naive datetime object.

        A non-naive datetime object holds a tzinfo,
        which gives information on the time-zone and daylight saving
        times.
        """
        # check using random time
        new_time = self.default_start_datetime
        new_time = new_time.replace(tzinfo=timezone.utc)
        sim = self.sim(*self.args, start_datetime=new_time, **self.kwargs)
        self.assertEqual(sim.current_time, new_time)

    def test_step_updates_time(self):
        """Test that :py:meth:`sim.step` updates :py:attr:`sim.current_time`.
        """
        sim = self.sim(
            *self.args,
            start_datetime=self.default_start_datetime,
            **self.kwargs
        )
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(
            sim.current_time, self.default_start_datetime+sim.step_size
        )
        self.assertEqual(sim.current_time_step, 1)

    def test_initialization_time(self):
        """Test that the simulator is running the requested number of
        steps when initialized.

        Default intialization should not require doing any
        initialization step.
        """
        n_steps = 3
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime
                + n_steps*self.default_step_size
            ),
            **self.kwargs)
        self.assertEqual(sim.current_time_step, n_steps)

    def test_initialization_time_over_day(self):
        """Test that the simulator can be initialized over a day.

        Test proper instantiation.
        Test the number of steps performed.
        """
        n_steps = 1
        sim = self.sim(
            *self.args,
            start_datetime=(
                self.default_start_datetime - n_steps*self.default_step_size
            ),
            **self.kwargs)
        expected_steps = (
            (timedelta(days=1) - self.default_step_size)
            // self.default_step_size
        )
        self.assertEqual(sim.current_time_step, expected_steps)
        self.assertEqual(
            sim.current_time,
            self.default_start_datetime - n_steps*self.default_step_size
        )

    def test_non_default_step_size(self):
        """Test the simulator behaviour with non default step_size.

        This test can be overriden in case, only a single step_size is
        allowed.
        """
        sim = self.sim(
            *self.args,
            start_datetime=self.default_start_datetime,
            step_size=self.random_step_size, **self.kwargs)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEquals(
            sim.current_time,
            self.default_start_datetime+self.random_step_size
        )

    def run_base_tests(self):
        super().run_base_tests()
        self.test_same_time_init_start_in_test()
        self.test_instantiation_with_default_datetime()
        self.test_instantiation_with_other_datetime()
        self.test_step_updates_time()
        self.test_initialization_time()
        self.test_initialization_time_over_day()
        self.test_non_default_step_size()





class TimeAwareSimulatorTests(unittest.TestCase):
    sim = TimeAwareSimulator
    default_start_datetime = datetime(2014, 1, 1, 4, 0, 0)
    args = [1] # n_households
    kwargs = {}
    args_step = []
    kwargs_step = {}

    def test_instantiation(self):
        sim = self.sim(*self.args, **self.kwargs)

    def test_initialtime(self):
        self.default_start_datetime
        sim = self.sim(*self.args, start_datetime=self.default_start_datetime, **self.kwargs)
        sim.initialize_starting_state(self.default_start_datetime, *self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 0)

    def test_default_stepsize(self):

        sim = self.sim(*self.args, start_datetime=self.default_start_datetime + timedelta(minutes=4), **self.kwargs)
        sim.initialize_starting_state(initialization_time=self.default_start_datetime, *self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 4)

    def test_different_stepsize(self):

        sim = self.sim(*self.args, step_size=timedelta(seconds=20), **self.kwargs)
        sim.initialize_starting_state(initialization_time=self.default_start_datetime-timedelta(minutes=3), *self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 9)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 10)
        self.assertEqual(sim.current_time, self.default_start_datetime+timedelta(seconds=20))

    def test_starting_time_with_initialization(self):
        sim = self.sim(*self.args, **self.kwargs)
        sim.initialize_starting_state(initialization_time=self.default_start_datetime - timedelta(minutes=23))
        self.assertEqual(sim.current_time, self.default_start_datetime)

    def test_future_raises_error(self):
        # tests that initializing from a future date won't work
        sim = self.sim(*self.args, **self.kwargs)
        self.assertRaises(
            ValueError,
            sim.initialize_starting_state,
            initialization_time=self.default_start_datetime + timedelta(minutes=1)
        )


    def test_initialize_without_date(self):
        # test initialization using a time object instead of datetime
        sim = self.sim(*self.args, **self.kwargs)
        sim.initialize_starting_state(*self.args_step, initialization_time=time(4, 0, 0),  **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 0)
        sim = self.sim(*self.args, **self.kwargs)
        sim.initialize_starting_state(*self.args_step, initialization_time=time(5, 0, 0),  **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 23*60)
        self.assertEqual(sim.current_time, self.default_start_datetime)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 23*60 + 1)
        self.assertEqual(sim.current_time, self.default_start_datetime + timedelta(minutes=1))


    def test_step(self):
        sim = self.sim(*self.args, **self.kwargs)
        sim.initialize_starting_state(initialization_time=self.default_start_datetime, *self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 0)
        sim.step(*self.args_step, **self.kwargs_step)
        self.assertEqual(sim.current_time_step, 1)
        self.assertEqual(sim.current_time, self.default_start_datetime + timedelta(minutes=1))

    def test_initialization_args(self):
        sim_class = self.sim
        class InhertitedSimulatro(sim_class):
            test_a = 0
            test_b = 0
            test_c = 0
            def step(self, a, b, *args_step, c=0, **kwargs_step,):
                self.test_a = a
                self.test_b = b
                self.test_c = c
                return super().step(*args_step, **kwargs_step)
        sim = InhertitedSimulatro(*self.args, **self.kwargs)
        sim.initialize_starting_state(
            1, 2, *self.args_step, c=3,
            initialization_time=self.default_start_datetime-timedelta(minutes=1), **self.kwargs_step)
        self.assertEqual(sim.test_a, 1)
        self.assertEqual(sim.test_b, 2)
        self.assertEqual(sim.test_c, 3)

class CallbacksNextDay(TimeAwareSimulator):
    test_trigger = False
    @ before_next_day_callback
    def step(self):
        return super().step()
    def on_before_next_day(self):
        self.test_trigger = True
class CallbacksNextDay4AM(TimeAwareSimulator):
    test_trigger = False
    @ before_next_day_callback_4am
    def step(self):
        return super().step()
    def on_before_next_day_4am(self):
        self.test_trigger = True

def _pass():
    pass

class CallBackTests(unittest.TestCase):
    default_start_datetime = datetime(2014, 1, 1, 4, 0, 0)
    minute_before_midnight_time = datetime(2014, 1, 1, 23, 59, 0)
    minute_before_4am_time = datetime(2014, 1, 1, 3, 59, 0)
    sim = TimeAwareSimulator
    callbacks = {
        before_next_day_callback: 'on_before_next_day',
        after_next_day_callback: 'on_after_next_day',
        before_next_day_callback_4am: 'on_before_next_day_4am',
        after_next_day_callback_4am: 'on_after_next_day_4am',
    }

    def test_passes_new_day(self):
        sim = self.sim(1, start_datetime=self.minute_before_midnight_time)
        sim.initialize_starting_state(self.minute_before_midnight_time)
        sim.step()
        self.assertEqual(sim.current_time, datetime(2014, 1, 2, 0, 0, 0))


    def test_callback_different_stepsize(self):
        sim = CallbacksNextDay(
            1,
            start_datetime=self.minute_before_midnight_time,
            step_size=timedelta(seconds=30)
        )
        sim.initialize_starting_state(self.minute_before_midnight_time)
        self.assertFalse(sim.test_trigger)
        sim.step()
        self.assertFalse(sim.test_trigger)
        sim.step()
        self.assertTrue(sim.test_trigger)

    def test_4am(self):
        sim = CallbacksNextDay4AM(1,start_datetime=self.minute_before_4am_time)
        sim.initialize_starting_state(self.minute_before_4am_time)
        self.assertFalse(sim.test_trigger)
        sim.step()
        self.assertTrue(sim.test_trigger)

    def test_before_after_executions(self):
        # defines a class for testing the execution
        class CallbacksExecution(TimeAwareSimulator):
            test_timestep = None
            test_time = None
        def set_test_times(self:CallbacksExecution):
            self.test_time = self.current_time
            self.test_timestep = self.current_time_step

        # test with the before execution
        setattr(CallbacksExecution, 'step', before_next_day_callback_4am(CallbacksExecution.step))
        setattr(CallbacksExecution, 'on_before_next_day_4am', set_test_times)
        sim_before = CallbacksExecution(1, start_datetime=self.minute_before_4am_time)
        sim_before.initialize_starting_state(self.minute_before_4am_time)
        sim_before.step()
        self.assertEqual(sim_before.test_time, self.minute_before_4am_time)
        self.assertEqual(sim_before.test_timestep, 0)

        # test with the after execution
        class CallbacksExecution(TimeAwareSimulator):
            test_timestep = None
            test_time = None
        def set_test_times(self:CallbacksExecution):
            self.test_time = self.current_time
            self.test_timestep = self.current_time_step

        setattr(CallbacksExecution, 'step', after_next_day_callback_4am(CallbacksExecution.step))
        setattr(CallbacksExecution, 'on_after_next_day_4am', set_test_times)
        sim_after = CallbacksExecution(1, start_datetime=self.minute_before_4am_time)
        sim_after.initialize_starting_state(self.minute_before_4am_time)
        sim_after.step()
        self.assertEqual(sim_after.test_time, self.minute_before_4am_time + timedelta(minutes=1))
        self.assertEqual(sim_after.test_timestep, 1)


    def test_callbacks_pass_return_values_to_step(self):

        for c in self.callbacks.keys():
            class CallbackArgs(TimeAwareSimulator):
                @ c
                def step(self, arg1, arg2, kwarg=True):
                    super().step()
                    return arg1, arg2, kwarg
            sim = CallbackArgs(1, start_datetime=self.minute_before_midnight_time)
            setattr(sim, self.callbacks[c], _pass)
            sim.initialize_starting_state(self.minute_before_midnight_time)
            arg1, arg2, kwarg = sim.step(1,2,kwarg=3)
            self.assertEqual(arg1, 1)
            self.assertEqual(arg2, 2)
            self.assertEqual(kwarg, 3)

    def test_chaining_callbacks(self):
        class CallbackChain(TimeAwareSimulator):
            counter = 0
            @ after_next_day_callback
            @ before_next_day_callback
            def step(self):
                return super().step()
            def on_before_next_day(self):
                self.counter += 1
            def on_after_next_day(self):
                self.counter += 1
        sim = CallbackChain(1, start_datetime=self.minute_before_midnight_time)
        sim.initialize_starting_state(self.minute_before_midnight_time)
        self.assertEqual(sim.counter, 0)
        sim.step()
        self.assertEqual(sim.counter, 2)



