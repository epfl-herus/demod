=============================
Utility Classes and Functions
=============================





Monte Carlo
~~~~~~~~~~~



.. module:: demod.utils.monte_carlo

.. autofunction:: monte_carlo_from_1d_cdf
.. autofunction:: monte_carlo_from_1d_pdf
.. autofunction:: monte_carlo_from_cdf
.. autofunction:: monte_carlo_from_pdf


Distibution functions
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: demod.utils.distribution_functions
    :members:


Get Methods
~~~~~~~~~~~

All simulators have some get methods which makes it usefull to
access data of the current simulation state.

TODO: add here the get methods and references to them so that we
can call them from the docstrings of the other classes

.. module:: demod.simulators.base_simulators
    :noindex:



.. autodata:: GetMethod
    :annotation: = for getting simulators attributes

.. autodecorator:: cached_getter


Logging simulation data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: demod.simulators.base_simulators
    :noindex:

.. autoclass:: SimLogger
    :members:

Sparse
~~~~~~~~

.. module:: demod.utils.sparse
    :noindex:

.. autoclass:: SparseTPM
    :members:


.. _error_messages:
Error messages
~~~~~~~~~~~~~~~

.. automodule:: demod.utils.error_messages
    :members:

.. _parse_helpers:
Parsing helpers
~~~~~~~~~~~~~~~~~~~~

.. automodule:: demod.utils.parse_helpers
    :members:



Other functions
~~~~~~~~~~~~~~~

.. module:: demod.simulators.util
    :noindex:

.. autofunction:: sample_population



Units Conversion
~~~~~~~~~~~~~~~~

.. automodule:: demod.utils.converters
    :members: