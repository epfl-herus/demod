========================
Abstract simulators
========================


Base simulators
----------------

The following simulators are abstract simulators that serve as basis for new
implementations of simulation.



.. module:: demod.simulators.base_simulators




.. autoclass:: Simulator
    :special-members: __init__
    :members:

.. autoclass:: MultiSimulator
    :members:
    :inherited-members:

.. autoclass:: TimeAwareSimulator
    :members:
    :inherited-members:


Callbacks for TimeAwareSimulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Callbacks
    :members:


Component simulators
---------------------

The following abstract simulators are dedicated to specific components.
Specific components are for example: activity of the residents, weather
or appliance consumption.

Activity
~~~~~~~~~

.. module:: demod.simulators.activity_simulators
    :noindex:

.. autoclass:: ActivitySimulator
    :members:

States
~~~~~~~~~


These simulators keeps track of the activity in households as being in a state.
At each step, a change of states is sampled for all households using
Monte Carlo sampling.

.. autoclass:: MarkovChain1rstOrder
    :members:



Sparse simulators
~~~~~~~~~~~~~~~~~

This implementation provides mainly the same features as states simulators
with better performances in the case of large
transition probability matrices.


.. module:: demod.simulators.sparse_simulators

.. autoclass:: SparseStatesSimulator
    :members:


Appliances
~~~~~~~~~~~~~~~~~~~

.. module:: demod.simulators.appliance_simulators
    :noindex:

.. autoclass:: AppliancesSimulator
    :members:

