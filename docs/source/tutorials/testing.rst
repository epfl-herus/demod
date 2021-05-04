===========
Testing
===========

This tutorial will show you how to use Demod testing functions
to test your simulator.

It requires that you have
:doc:`created your own simulator <./create_a_simulator>`.
It also requires that you downloaded the testing repository,
which you can have by
:ref:`installing Demod from source <installation_from_source>`.


Base Test Classes
-----------------

In the repository that you have downloaded from github,
you can open the test/ folder, which contains test utilities.

Demod uses the
`unittest library <https://docs.python.org/3/library/unittest.html>`_
.

Create a new pyhon file inside the test folder with this inside:

.. code-block:: python

    import unittest
    # Import a test class that already contains tests common
    # to all Simulator objects
    from test_base_simulators import BaseSimulatorChildrenTests
    from demod.simulators.your_simulator import YourSimulator

    # We create a test class for your simulator
    class YouTestClass(BaseSimulatorChildrenTests):
        # Specifies the simulator to test
        sim = YourSimulator
        # Args and kwargs for the __init__ method
        args = [1]  # n_households
        kwargs = {}
        # Number of households that should be simulated (based on the ars and kwargs)
        n_households = 1
        # Args and kwargs of the step method
        args_step = [np.array([1]), 50]  # two step inputs
        kwargs_step = {}
        # Getters methods which should not be tested
        unimplemented_getters = [
            'get_something_that_is_not_made_to_work',
            'get_another'
        ]
        # Arguments for getter methdos that requred some
        getter_args = {}  # dic of the form "get_name_of_getter": [*args]


Now that you have created the test class, you can simply add some
test method to it.
Here is an example of testing another algo

.. code-block:: python

    def test_with_another_algo(self):
        # Set a new kwarg to the test class
        self.kwargs['algo'] = 'test_algo'
        # Helper that runs all the base tests of the TestClass
        self.run_base_tests()
        # Remove the tested kwarg
        self.kwargs.pop('algo')


Running the test
~~~~~~~~~~~~~~~~

To run you test file you can add at the end of the file

.. code-block:: python

    if __name__ == '__main__':
        unittest.main()

and then you can simply run the file from the terminal to
see the test output.


Time Aware Simulators tests
----------------------------

Similarly as the previous section, you can import another
TestClass from which you can inherit.
You will also need to add other class parameters:

.. code-block:: python

    import unittest
    # Import a test class that already contains tests common
    # to all Simulator objects
    from test_base_simulators import TimeAwareSimulatorChildrenTests
    from demod.simulators.your_simulator import YourSimulator

    # We create a test class for your simulator
    class YouTestClass(TimeAwareSimulatorChildrenTests):
        sim = YourSimulator
        args = [1]  # n_households
        kwargs = {}
        n_households = 1
        args_step = [np.array([1]), 50]  # two step inputs
        kwargs_step = {}
        unimplemented_getters = []
        getter_args = {}  # dic of the form "get_name_of_getter": [*args]

        # Time aware parameters

        # The start_datetime that is implemented by default in you simualator
        default_start_datetime = datetime.datetime(2014, 1, 1, 0, 0, 0)
        # The initialization time that is implemented by default
        default_initialization_time = datetime.time(0, 0, 0)
        # The default step_size used in your simulation
        default_step_size = datetime.timedelta(minutes=1)
        # A random start_datetime at which your simulator should
        # be able to be started
        random_time = datetime.datetime(2008, 4, 5, 13, 42, 26)
        # A random step_size that will be tried to be assign to your
        # simulator
        random_step_size = datetime.timedelta(
            hours=2, minutes=42, seconds=35, milliseconds=221)


.. Note::
    Some of the test methdos might not work with your parameters.
    You can read the :doc:`test api documentation <../api/testing>` to
    check what each test method is doing and override it in you test
    class if necessarry.
