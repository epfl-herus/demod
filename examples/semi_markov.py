
#%%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import demod

from importlib import reload
from demod.utils.monte_carlo import monte_carlo_from_1d_pdf
from demod.datasets.GermanTOU.loader import GTOU
from demod.simulators.base_simulators import SimLogger
from demod.simulators.activity_simulators import MarkovChain1rstOrder, SemiMarkovSimulator, SubgroupsIndividualsActivitySimulator

#%%
reload(demod.simulators.activity_simulators)
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator
reload(demod.utils.subgroup_handling)
from demod.utils.subgroup_handling import subgroup_households_to_persons, Subgroup
# A list of lists of subgrouups
hh_subgroups = [
        {'household_type': 2},
        {'household_type': 3, 'n_residents': 3},
        {'household_type': 4, 'n_residents': 5}
    ]
subgroup_households_to_persons(hh_subgroups)
n_hh_list = [300, 200, 100]
reload(demod.simulators.activity_simulators)
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator, SemiMarkovSimulator, MarkovChain1rstOrder
sim = SubgroupsIndividualsActivitySimulator(
    hh_subgroups,
    n_hh_list,
    logger=SimLogger('get_states', 'current_time'),
    subsimulator=MarkovChain1rstOrder,
    use_7days=True
)

for i in range(9*144):
    sim.step()

dict_states = sim.logger.get('get_states')
time_axis = sim.logger.get('current_time')

for lab, states in dict_states.items():
    plt.plot(time_axis, states, label=lab)

plt.legend()
plt.show()



# %%
