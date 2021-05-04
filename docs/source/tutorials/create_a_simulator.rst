
.. _create-sim_label:

==============================
Creating your first simulator
==============================

This tutorial will help you to create your own simulator using demod.
It assumes that you have
:doc:`installed demod <../installation/installation>`, and
that you are familiar with both: python and
`numpy <https://numpy.org/>`_ .

A simulator is a python class that inherits from the
:py:class:`demod.simulators.base_simulators.Simulator`
class.

We can start by creating this object

.. code-block::

    import numpy as np
    from demod.simulators.base_simulators import Simulator

    class ExampleSimulator(Simulator):
        """This is a simple example simulator showing how to implement a sim.

        It implements a simple simulator that memorize the number of occupants
        inside the households.
        """

Now that we have created a Simulator and that we have decided what it
should do,
there are 3 methods that every :py:class:`Simulator` must implement.

The first one is the
constructor, :py:meth:`~demod.simulators.base_simulators.Simulator.__init__`.
If have never heard of
python's :py:meth:`__init__`, you can
`get more information <https://docs.python.org/3/reference/datamodel.html#object.__init__>`_
.

.. code-block:: python

    class ExampleSimulator(Simulator):
        """This is a simple example simulator showing how to implement a sim.

        It implements a simple simulator that memorize the number of occupants
        inside the households, based on arrivals and departures.

        """
        def __init__(self, n_hh, *args, **kwargs):
            """Initialization of the function parameters and set-up.

            Put here everything you want to be performed before the simulation.
            This can inclue loading files or setting variable.
            *args, **kwargs are additional arguments that are passed to the
            parent simulator object.

            Args:
                n_households: The number of households.
                    A parameter that is passed to any Simulator.
                max_residents: The maximum number of residents
                    that are allowed in the households.
            """
            # Initialize the parent instance of the simulator
            super().__init__(n_households, *args, **kwargs)

            # Attributes a variable that will store the position of the households
            # Empty array of the size of the number of households
            self.occupants = np.empty(n_households, dtype=int)

.. note::
    We use numpy arrays to store the variables and to do computation
    because it makes coding and scalability very easy.

Now, the constructor object can be created using


.. code-block:: python

    example_sim = ExampleSimulator(42)

However we did not initialize the value of the number of occupants (it
is still an empty array)
To initialize it, we can use the second simulator method :
:py:meth:`Simulator.initialize_starting_state`

.. code-block:: python

    def initialize_starting_state(self, max_residents):
        """This is the function initializing the starting state.

        The starting state defines which values the simulator
        should have at the beggining of the simulation.

        Args:
            max_residents: the maximum number of residents in the households
        """
        # Randomly initialize the number, using the n_household class attribute
        self.occupants = np.random.randint(0, max_residents, size=self.n_households)

        # Call to the parent initialization method
        super().initialize_starting_state()

Now that we have implemented the initialization method, we
can add it do the constructor

.. code-block:: python

    def __init__(self, n_households, max_residents, *args, **kwargs):

        super().__init__(n_households, *args, **kwargs)
        self.occupants = np.empty(n_households, dtype=int)

        # Initialize the simulator
        self.initialize_starting_state(max_residents)

Now, when we create the simulator, it will also run the initialization
of the starting state. Note that we added an input variable that
determines the maximum number of residents in a house at the start.

It is time to create the third method of the simulator :
:py:meth:`Simulator.step`

It performs an iteration of the
simulation.
It receives some variables as input, and uses them to update
its internal variables.

.. code-block:: python

    def step(self, arriving, leaving):
        """Step function of the simulator.

        Updates the number of residents based
        on the number of arrivals and departures.
        """
        self.occupants += arriving
        self.occupants -= leaving
        # Calls the parent step method.
        super().step()

We have implement the three mandatory methods, by now we also want
to be able to read the number of occupant.
For that we can create a getter.

.. code-block:: python

    def get_occupancy(self):
        """Return the number of occupants.
        """
        return self.occupants


You can create any getter you want, but make their name start
with :py:obj:`get_` and try to reuse the naming of getters that
already exist if possible.


Now we have finished, the simulator can be used in a simulation.
You could improve it by settings limits on the persons, or raising
errors depending on the values of the inputs (ex: no one can leave
if there is no one at home).
You can also :doc:`continue the tutorial <./time_aware>` where
we use demod to add time functionality in the simulator.


.. note::
    When you implement these 3 methods, you should always call the
    :py:meth:`super` corresponding method.
    This is because the base simulator contains some instructions
    that are helpful for all kinds of simulators, such
    as the logging of the values.


Base Simulator
~~~~~~~~~~~~~~

In demod, all simulations are performed by Simulator objects.
Therefore all simulators are children of the :py:class:`Simulator` class.


.. module:: demod.simulators.base_simulators
    :noindex:


The base simulator provides some functionality common to all
simulators:

.. autoclass:: Simulator
    :noindex:


You need to implement in your simulator the 3 following methods.

The constructor :py:meth:`Simulator.__init__`. If have never heard of
python's :py:meth:`__init__`, you can
`get more information <https://docs.python.org/3/reference/datamodel.html#object.__init__>`_
.


    .. automethod:: Simulator.__init__
        :noindex:



:py:meth:`Simulator.initialize_starting_state` initialize
the variable simulations and
computes their initial value at the start of the simulation.

    .. automethod:: Simulator.initialize_starting_state
        :noindex:

:py:meth:`Simulator.step` performs an iteration of the
simulation.
It receives as inputs requested variables.

    .. automethod:: Simulator.step
        :noindex:



