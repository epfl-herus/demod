
===============
Multi Simulator
===============

Demod provides an abstact
:py:class:`~demod.simulators.base_simulators.MultiSimulator`
that can handle the simulation of different subsimulators
in parallel.




Creating a Time-Aware MultiSimulator
-------------------------------------

This tutorial shows you how to combine two base simulators:
:py:class:`~demod.simulators.base_simulators.MultiSimulator`
that is usefull for handling different
simulators and
:py:class:`~demod.simulators.base_simulators.TimeAwareSimulator`
 that handles the time.

We can use multiple inheritance for that::

    class MultiFunctionSim(TimeAwareSimulator, MultiSimulator):

Then we need to be careful in the intialization of the simulator.
In particular we will need to do 2 things, 1. define how the n_households
will be assigned in the different suubsimulators and then initialize those
subsimulators::

    def __init__(self, n_households: int, **kwargs):

        # dispatch the n_households in the subsimulators using a pdf
        n_hh_list = monte_carlo_from_1d_pdf(
            [0.3, 0.5, 0.2], n_samples=n_households)

        # initializes the subsimulator of MultiSimulator
        [Sim0(n_hh_list[0]),  Sim1(n_hh_list[1]), Sim2(n_hh_list[2])]

        # initialize the parents
        # this should work fine as TimeAwareSimulator will not
        # use simulator_list for setting n_households
        super().__init__(simulator_list, **kwargs)

        # Now you can initialize the simulator like TimeAwareSimulator
        super().initialize_starting_state( *initialization_args,
            initialization_time=datetime.time(0),
            **initialization_kwargs,)

Due to the Method resolution order
, this works well, as the
:py:attr:`simulator_list`
is only passed to :py:class:`MultiSimulator`, which converts it into
:py:attr:`n_households` for
:py:class:`Simulator`
for creating
you own new simulator.

Note that any :py:obj:`*args`, :py:obj:`**kwargs`, in
:py:meth:`super().initialize_starting_state`
are passed to the initialization that calls :py:meth:`step`  (
see :ref:`create-time-aware-sim_label`)