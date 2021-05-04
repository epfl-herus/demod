=============================
Utility Classes and Functions
=============================





Monte Carlo
~~~~~~~~~~~

demod provides some helper function for `Monte Carlo (MC) sampling`_
with numpy arrays and using discrete domains.

.. _Monte Carlo (MC) sampling: https://en.wikipedia.org/wiki/Monte_Carlo_method

Discrete Probability and Cumlative distribution functions (PDF and CDF)
can be used for
sampling.
See the following example::

    pdf = np.array([0.3, 0.65, 0.05])
    # 30% chance return 0, 65% chance return 1, 5% chance return 2
    out = monte_carlo_from_1d_pdf(pdf)

.. module:: demod.utils.monte_carlo

.. autofunction:: monte_carlo_from_1d_cdf
.. autofunction:: monte_carlo_from_1d_pdf
.. autofunction:: monte_carlo_from_cdf
.. autofunction:: monte_carlo_from_pdf




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


Other functions
~~~~~~~~~~~~~~~

.. module:: demod.simulators.util
    :noindex:

.. autofunction:: sample_population



Units Conversion
~~~~~~~~~~~~~~~~

.. automodule:: demod.utils.converters
    :members: