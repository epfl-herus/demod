============
Contributing
============


You can contribute by adding any data or simulator via GitHub.
There are just few rules that should be respected.


PEP guidelines
--------------

Demods respects the following PEP gidelines.

Style
~~~~~~~~~~~~~~~~

The style of the code must match `PEP 8`_.
In particular a maximum of 79 characters per line.

Type Annotations
~~~~~~~~~~~~~~~~

We use type annotations as according to `PEP 484`_ and `PEP 526`_ to annote
variables in demod.
The main benefit is that
type checkers and IDEs can take advantage of them for static code
analysis, which will make the inputs and outputs more readable.
Documentation can also automatically include the types.

.. _PEP 8:
   https://www.python.org/dev/peps/pep-0008/

.. _PEP 484:
   https://www.python.org/dev/peps/pep-0484/

.. _PEP 526:
    https://www.python.org/dev/peps/pep-0526/



Documenting your code
---------------------


Docstrings
~~~~~~~~~~

We use docstrings of the simulators to simply add new simulators and methods
in demod'API.

You can follow this example from `sphinx`_.

Google style with Python 3 type annotations::

    def func(arg1: int, arg2: str) -> bool:
        """Summary line.

        Extended description of function.

        Args:
            arg1: Description of arg1
            arg2: Description of arg2

        Returns:
            Description of return value

        """
        return True

    class Class:
        """Summary line.

        Extended description of class

        Attributes:
            attr1: Description of attr1
            attr2: Description of attr2
        """

        attr1: int
        attr2: str


.. _sphinx:
    https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#type-annotations


Simulators identity cards
~~~~~~~~~~~~~~~~~~~~~~~~~

Demod has a standardized notation for Simulators documentation
refered to as "cards".

See the following example, on how you can document your simulator:

.. code-block:: python

    class YouSimulator(TimeAwareSimulator):
        """One line description of your simulator.

        Longer description where you can explain how it works,
        or what are the different possible simulations modes.
        You can explain with which simulator it is compatible.
        You can also cite here the autors.

        Params
            # The following lines represent the parameters for
            # the instatiation of the simulator
            :py:attr:`~demod.utils.cards_doc.Params.n_households`
            :py:attr:`~demod.utils.cards_doc.Params.data`
            :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
            :py:attr:`~demod.utils.cards_doc.Params.population_sampling_algo`
            :py:attr:`~demod.utils.cards_doc.Params.logger`
        Data
            # These lines represent the data that the data object
            # must have
            :py:meth:`~demod.utils.cards_doc.Loader.load_tpm`
            :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
        Step input
            # The inputs for the step function
            :py:meth:`~demod.utils.cards_doc.Inputs.occupancy`
        Output
            # The outputs of the simulator, the getter methods
            :py:meth:`~demod.utils.cards_doc.Sim.get_active_occupancy`
            :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`
        Step size
            # What step size the simulator can handle
            10 Minutes


        """

.. note::
    The values in the id cards are links written using spinx and
    rst. They allow the creation of hyper links in the documentation.



Citing
------

When you add any component to Demod, don't forget to always cite
the source where you find the data, or the source where you
find inspiration for the code.

You can add it in the docstring of the Simulator or of the DatasetLoader.