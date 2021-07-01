
.. _co-simulation:

=========================================
Integration with Co-simulation Frameworks
=========================================

Once you have created your simulator, we provide a simple adaptor API to
a co-simulation framework.

Co-simulation frameworks provide functionality to combine heterogeneous 
simulation  components  from  different  domains  in  an  integrated  
simulation.
Indeed, cos-imulation deisgn ensures *modularity* to the simulation, 
allowing to incorporate additional models such as networks 
(e.g., eletric grid andmobility infrastructure), 
devices (e.g., electric vehicles and batteries), 
agents (e.g.,industries  and  governance  authorities)  
and  platforms  (e.g.,  spot  market) and 
facilitating model use and modification.


Mosaik integration
------------------

`mosaik <https://mosaik.readthedocs.io/en/latest/overview.html>`_ is a 
framework which provides API to coordinated simulations
of Smart Grid scenarios.

You can access the
`GitHub repository of mosaik-demod <https://github.com/epfl-herus/mosaik-demod>`_
to get the source code and see a demo of usage.

The following instructions explain how to install the mosaik-demod
adapters and how to incorporate demod simulator into mosaik.


1. Install the python library providing the adaptors ::

    pip install mosaikdemod



2. Import the abstract adapter.
If the demod simulator simulates various households,
use the Household module.
For a single value simulated (ex. climate), use The SingleValue module.

.. code-block:: python

    from mosaikdemod.adaptors import AbstractHouseholdsModule
    from mosaikdemod.adaptors import AbstractSingleValueModule


3. import demod library

.. code-block:: python

    import demod


4. Inherit from the abstract module

.. code-block:: python

    class ComponentSimulator(AbstractHouseholdsModule):


5. Specify the attributes of the simulator that can be accessed

.. code-block:: python

    attributes_dict = {
        'attr_name_in_mosaik': 'get_demod_getter',
        'other__attr': 'get_smth_else',
        ...
    }

6. Specify the inputs of the simulator that can be accessed

.. code-block:: python

    step_inputs_dict = {
        'attr_name_in_mosaik': 'step_input_demod',
        'other_input': 'input_other',
        ...
    }

7. Override the :py:meth:`__init__()` method

.. code-block:: python

    def __init__(
        self,
        # Name of what is simulated used for mosaik instances
        simulated_component='CompName',
        # The simulator class you want to simulate
        default_simulator=demod.simulators.example_simulators.ExampleSimulator,
        # The mosaik step size (depend on your definition)
        step_size=60
    ):
        super().__init__(simulated_component, default_simulator, step_size)


8. Import your simulator to your mosaik scenario script.

.. code-block:: python

    # Define the Simulator
    sim_config = {
        ...
        'CompNameSimulator': {
            'python': 'python_file_of_the_sim:ComponentSimulator',
        },
        ...
    }

    # Instantiate the simulator
    sim = world.start('CompNameSimulator')


    # Instantiate the households with parameters
    component = actsim.HouseholdsGroupCompName(
        inputs_params={  # demod init params of sim
            'n_households': n_households,
            'start_datetime': START_DATETIME,
            ...
        }
    )
    # OR instantiate a  SingleValue simulator (remove HouseholdsGroup)
    component = actsim.CompName(
        inputs_params={  # demod init params of sim
            'start_datetime': START_DATETIME,
            ...
        }
    )


9. Connect the simulators. You can connect a whole household group to another one if you use 2 demod components.
Or you can also connect all the households individually
by calling the children method

.. code-block:: python

    # Connect 2 demod components
    # comp1 passes attr to comp2
    world.connect(component1, component2, 'attr_name_in_mosaik')

    # Connect 2 demod components with single value
    # component_single_value passes attr to comp2
    world.connect(component_single_value, component2, 'attr_name_in_mosaik')

    # Connect a single household using the children
    world.connect(component.children[42], other_mosaik_comp, 'attr_name_in_mosaik')



We recommend that you check the example files available at
`demo.py <https://github.com/epfl-herus/mosaik-demod/blob/master/demo.py>`_
and
`simulator_mosaik_modular.py <https://github.com/epfl-herus/mosaik-demod/blob/master/simulator_mosaik_modular.py>`_
.

