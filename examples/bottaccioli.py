""" Attempt to reproduce Bottacioli paper.

doi 10.1109/ACCESS.2018.2886201
"""
# %%
import os
import sys

import matplotlib.pyplot as plt

import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from demod.simulators.util import sample_population
from demod.simulators.appliance_simulators import ActivityApplianceSimulator
from demod.simulators.base_simulators import SimLogger
from demod.datasets.Germany.loader import GermanDataHerus
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator, SemiMarkovSimulator, MarkovChain1rstOrder

# %%
n_households = 5

data = GermanDataHerus(version='vBottaccioli')

hh_subgroups, probs, _ = data.load_population_subgroups()

n_hh_list = sample_population(n_households, probs)

sim = SubgroupsIndividualsActivitySimulator(
    hh_subgroups,
    n_hh_list,
    logger=SimLogger('get_states', 'current_time', aggregated=False),
    subsimulator=MarkovChain1rstOrder,
    data=data,
    use_7days=True
)

sim_app = ActivityApplianceSimulator(
    n_households, initial_activities_dict=sim.get_states(),
    data=data,
    logger=SimLogger('get_current_power_consumptions', aggregated=False)
)

# %%


for i in range(2*144):
    sim.step()
    sim_app.step(sim.get_states())


# %%
dict_states = sim.logger.get('get_states')
time_axis = sim.logger.get('current_time')

power_consumptions = sim_app.logger.get('get_current_power_consumptions')


n_ieth_hh = 1
fig, axes  = plt.subplots(2,1, sharex=True)
axes[0].plot(power_consumptions[:, n_ieth_hh, :])
for state, array in sim.logger.get('get_states').items():
    axes[1].plot(array[:, n_ieth_hh], label = state)
plt.legend()
plt.show()

# %%

# TODO
# Make some appliance probabilistic in the activity
# Add the load patterns to ActivityApplianceSimulator
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Make a visualization of the activities like in bottacioli
# Define what to do for appliance that do not depend on activity