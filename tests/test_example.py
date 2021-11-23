"""This package shows examples on how to use the test framework.

You can simply copy the code of this simulator to create your
tests for you simulator.
"""
import datetime
import unittest
from test_base_simulators import BaseSimulatorChildrenTests, TimeAwareSimulatorChildrenTests
from demod.simulators.base_simulators import ExampleSimulator, TimeAwareExampleSimulator


class ExampleSimulatorTests(BaseSimulatorChildrenTests, unittest.TestCase):
    sim = ExampleSimulator # Use the example Simulator, as Simulator cannot be call only by init and step
    args = [1, 3]  # n_households, max_residents
    kwargs = {}
    n_households = 1
    args_step = [1, -1] #two step inputs
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {} # dic of the form "get_name_of_getter": [*args]
    def test_(self):
        pass


class TimeAwareExampleSimulatorTests(TimeAwareSimulatorChildrenTests):
    sim = TimeAwareExampleSimulator
    args = [1, 3]  # n_households, max_residents
    kwargs = {}
    args_step = [1, -1]  # two step inputs
    kwargs_step = {}
    unimplemented_getters = []
    getter_args = {}  # dic of the form "get_name_of_getter": [*args]
    # Time aware parameters
    default_start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)
    default_initialization_time = datetime.time(4, 0, 0)
    default_step_size = datetime.timedelta(minutes=1)
    random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
    random_step_size = datetime.timedelta(
        hours=2, minutes=42, seconds=35, milliseconds=221)

    def test_(self):
        pass