==========
Quickstart
==========


Do you **don't want to use** Python ?
:ref:`Learn how to use demod without python <withoutpython>`.

Do you want to **add load profiles** in your python project ?
:ref:`Get started with demod <runnig_a_simulation>`.

Do you want to create **your own simulation algorithms** ?
:doc:`Create your first demod simulator <../tutorials/create_a_simulator>`.

Do you want to **add your data** for generating tailored load profiles ?
:doc:`Create your demod datset <../tutorials/datasets>`.

.. _runnig_a_simulation:

Running a Simulation
--------------------

Once you have :ref:`installed demod <installation>`,
you can import the library in you python file by doing:

.. code-block:: python

    import demod

    from demod.simulators.load_simulators import LoadSimulator

    sim = LoadSimulator(n_households=1)

    for i in range(24 * 60):
        sim.step()
        print(sim.get_power_demand())

.. _changing_the_dataset:

Change the dataset
~~~~~~~~~~~~~~~~~~~

You can assign different datasets to your simulator.

Datasets in Demod are classes and they are often refered to
as DatasetLoader.

.. code-block:: python

    # Import the DatasetLoader
    from demod.datasets.CREST.loader import Crest

    # Passes the instantiated class to the simulator
    sim = LoadSimulator(n_households=1, data=Crest())


For easing your life with the dataset, you can use the
:ref:`excell file  prepared on purpose for demod <excell_input_file>`.

If you want to create your own simulators, you can get started with the
introduction in :doc:`../tutorials/create_a_simulator`.

.. _using_a_logger:

Using a logger to access the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following lines of code show how to use the
:py:class:`~demod.simulators.base_simulators.SimLogger`
object from the simulators.


.. code-block:: python

    from demod.simulators.base_simulators import SimLogger

    sim = LoadSimulator(
        n_households=1,
        logger=SimLogger('current_time', 'get_power_demand', 'get_temperatures')
    )

    for i in range(24 * 60):
        sim.step()

    # Plots all the logged data one by one
    sim.logger.plot()
    # plots all the data in column
    sim.logger.plot_column()
    # Gets array of the data, this can be used for your own post-processing
    elec_cons = sim.logger.get('get_power_demand')


If you simulate many households,
by default the data is aggregated over all the households, but you can
also access disaggregated data by setting:

.. code-block:: python

    SimLogger('get_power_demand', aggregated=False)


You can have more information about the logger at
:py:class:`demod.simulators.base_simulators.SimLogger`.



Handling multiple simulators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
:py:class:`~demod.simulators.load_simulators.LoadSimulator`
can be decomposed in different components
as explain in :doc:`the overview section <../overview/index>`.

Here we present how to perform a simulation with multiple
components simulators,
how to handle different timesteps,
how to handles the inputs and outputs from the different simulators.

First you need to choose the components among the different
available simulators, which you can find in the
:ref:`simulators API documentation <available_datasets>`.
You can also select a dataset from the
:doc:`available datasets <../api/data_api>`.

In this example, we will simulate the lighting in a household.

.. code-block:: python

    import datetime

    from demod.datasets.GermanTOU.loader import GTOU
    from demod.datasets.OpenPowerSystems.loader import OpenPowerSystemClimate

    from demod.simulators.crest_simulators import Crest4StatesModel
    from demod.simulators.weather_simulators import RealClimate
    from demod.simulators.lighting_simulators import FisherLighitingSimulator

    n_households = 10
    # Start of the simulation
    start_datetime = datetime.datetime(2014, 3, 1, 0, 0, 0)

    climate_sim = RealClimate(
        data=OpenPowerSystemClimate('Germany'),  # A climate dataset
        start_datetime=start_datetime  # Specifiy the start of the simulaiton
    )

    activity_sim = Crest4StatesModel(
        n_households,
        data=GTOU('4_States'),  # Time of use survey for germany
        start_datetime=start_datetime,  # Specifiy the start of the simulaiton
    )

    lighting_sim = FisherLighitingSimulator(
        n_households,
        # Gets the initial values from other simulators
        initial_active_occupancy=activity_sim.get_occupancy(),
        initial_irradiance=climate_sim.get_irradiance()
    )
    # No data was specified, it will use a default dataset.


Now that we have intialized the three simulators, with different data
we need to run the simulation.
However we have to be careful because the step_size of the
simulation is different for each simulator.
You can check the different step_size in the
:doc:`simulators API documentation <../api/simulators_api>`.

Running the simulation simply involves running the step function
for the desired amount of time. In this example we run it for two days.

.. code-block:: python

    for _ in range(2*24):
        # step size of one hour
        climate_sim.step()

        for __ in range(6):
            # step size of 10 minutes
            activity_sim.step()
            # two inputs are required for lighting step
            lighting_sim.step(
                active_occupancy=activity_sim.get_active_occupancy(),
                irradiance=climate_sim.get_irradiance()
            )

Note how we connected the inputs of the step function for
lighting simulator using the the corresponding getter methods.

You can find all the inputs and outputs of simulators also in the
:doc:`simulators API documentation <../api/simulators_api>`.


After the simulation is run, you
can :ref:`use a logger <using_a_logger>` to check what was simulated
.


.. _withoutpython:

Without Python Scripting
--------------------------
Available in a next release:
You can create simple load profiles by running loadprofile.exe .
The parameters of the created load profiles can be changed in the
input excell file : inputs.xls, where you can also decide the location
and the format of the generated profiles.

Please contact us if you would need this tool, so that we can get insights
about how we can design it better.

.. note::

    You will still need to have python and the corresponding
    library installed. See
    :doc:`installation instructions <../installation/installation>`.
