.. _create-time-aware-sim_label:

==============================
Create a Time Aware Simulator
==============================

This tutorial will help you to create a simulators
that keeps track of the current time using demod.
It assumes that you have followed the previous tutorial about
:doc:`creating a demod simulator <./create_a_simulator>`.


Demod helps handling time in the simulation when required.
It can do that for simulator inheriting from
:py:class:`~demod.simulators.base_simulators.TimeAwareSimulator`.

.. code-block:: python

    class TimeAwareExampleSimulator(TimeAwareSimulator):
        """Adds Time functionality to our previous example Simulator."""


The first thing we have to handle now is the constructor of the
simulator.
Let's first check the one of the
:py:class:`~demod.simulators.base_simulators.TimeAwareSimulator`.



.. module:: demod.simulators.base_simulators
    :noindex:


.. automethod:: TimeAwareSimulator.__init__
    :noindex:




There are two new parameters that are mandatory.
The :py:attr:`~demod.utils.cards_doc.Params.step_size` which
will tell how long a step of the simulator last,
and :py:attr:`~demod.utils.cards_doc.Params.start_datetime` which
specifies when the simulation starts.

.. Note::

    Demod uses the standard python library
    `datetime <https://docs.python.org/3/library/datetime.html>`_
    for handling the time, where
    `datetime.datetime <https://docs.python.org/3/library/datetime.html>`_
    represent a time stamp and
    `datetime.timedelta <https://docs.python.org/3/library/datetime.html#datetime-objects>`_
    represent time intervals.


Let's adapt the __init__ and methods:

.. code-block:: python

    def __init__(
            self, n_households, max_residents, *args,
            initialization_algo='random', **kwargs
        ):

        # The new parameters will simply be passed through  *args, **kwargs to
        # TimeAwareSimulator, we don't need any change about that.
        super().__init__(n_households, *args, **kwargs)
        self.occupants = np.empty(n_households, dtype=int)

        # Initialize the simulator
        self.initialize_starting_state(max_residents)


.. Note::
    Our example simulator can accept any step_size, but some simulator
    might allow only one step size, which can be specified in
    the __init__ method.
    You can also add a condition on the start_datetime, if
    for example the simulator is used for retrieving old climate data,
    the start_datetime must be included in the dataset.



We did not need to change many things, excepted that we added
a new :py:obj:`initialization_algo` argument with a default value.

Now let's see how we can use initialize_starting_state.

.. automethod:: TimeAwareSimulator.initialize_starting_state
    :noindex:

We can implement our initialization like this:

.. code-block:: python

    def initialize_starting_state(self, max_residents, initialization_algo):
        """This is the function initializing the starting state.

        Different methods, depending on the algorithm.
        """
        if initialization_algo == 'random':
            # Same as previous initialization
            self.occupants = np.random.randint(
                0, max_residents, size=self.n_households
            )
            super().initialize_starting_state(
                # Will not run any step
                # self.current_time tracks the time during simulation
                initialization_time=self.current_time
            )
        elif initialization_algo == 'all_inside_at_4am':
            # Now assume that all the residents are at home at 4 AM

            # Sets the residents to be all there
            self.occupants = max_residents * np.ones(self.n_households)

            # Call to the parent initialization
            super().initialize_starting_state(
                # Specifies that the method has been initialized for 4 AM
                initialization_time=datetime.time(4, 0, 0),
                # Sets dummy variables for the step function during
                # the initial steps
                arriving=np.zeros(self.n_households),
                leaving=np.zeros(self.n_households)
            )
        else:
            # Import a specific error message
            from demod.utils.error_messages import UNIMPLEMENTED_ALGO_IN_METHOD
            # Raise the error message
            raise NotImplementedError(UNIMPLEMENTED_ALGO_IN_METHOD.format(
                algo=initialization_algo,
                method=self.initialize_starting_state,
            ))

.. Note::
    Demod provides support for error messages, you can
    :ref:`learn more here <error_messages>`.


Finally we can look at the step function:

.. automethod:: TimeAwareSimulator.step
    :noindex:

The interesting thing is that you can add a callback to the step
function that will trigger another method when being called.


.. code-block:: python

    from demod.simulators.base_simulators import Callbacks

    @ Callbacks.before_next_day_4am
    def step(self, arriving, leaving):
        """Step function of the simulator.
        """
        self.occupants += arriving
        self.occupants -= leaving
        super().step()

    def on_before_next_day_4am(self):
        """This function is called by the Callbacks.before_next_day_4am

        It will be called every day at 4 am, right before the step
        function is called.
        """
        # We want to print the percentage households with no one at home.
        print("There is {} percent of the households that are empty".format(
            np.mean(self.occupants==0) * 100
        ))

You can see all the available callbacks function, and their corresponding
methods in :py:class:`demod.simulators.base_simulators.Callbacks`.


Finally as a bonus,
you can try to pass
:py:obj:`logger=SimLogger('current_datetime', 'your_getter)`
to your new simulator, to have a plot that used the time
in the x coordinates !

You can continue the tutorial by