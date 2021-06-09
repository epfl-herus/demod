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
from demod.simulators.appliance_simulators import ActivityApplianceSimulator, ProbabiliticActivityAppliancesSimulator, SubgroupApplianceSimulator
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
    subsimulator=SemiMarkovSimulator,
    data=data,
    use_7days=True
)

sim_app = ActivityApplianceSimulator(
    n_households, initial_activities_dict=sim.get_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)

sim_CREST = SubgroupApplianceSimulator(
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    data=data,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=True)
)

sim_prob_app = ProbabiliticActivityAppliancesSimulator(
    n_households, initial_activities_dict=sim.get_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)

# %%

print(sim_app.appliances['use_variable_loads'])

for i in range(1*144):
    sim.step()
    for i in range(10):
        sim_app.step(sim.get_states())
        sim_prob_app.step(sim.get_states())
        sim_CREST.step(sim.get_active_occupancy())


# %%
dict_states = sim.logger.get('get_states')
time_axis = sim.logger.get('current_time')

power_consumptions = sim_app.logger.get('get_current_power_consumptions')
power_consumptions_prob = sim_prob_app.logger.get('get_current_power_consumptions')
time_axis_app = sim_app.logger.get('current_time')


sim_CREST.logger.plot()
# %%
for n_ieth_hh in range(n_households):
    fig, axes  = plt.subplots(2,1, sharex=True)
    for i, name in enumerate(sim_app.appliances['name']):
        # plots each appliance pattern
        if sim_app.available_appliances[n_ieth_hh, i]:
            axes[0].plot(time_axis_app, power_consumptions[:, n_ieth_hh, i], label=name)
    for i, name in enumerate(sim_prob_app.appliances['name']):
        # plots each appliance pattern
        if sim_prob_app.available_appliances[n_ieth_hh, i]:
            axes[0].plot(time_axis_app, power_consumptions_prob[:, n_ieth_hh, i], label=name)
    axes[0].legend()
    for state, array in sim.logger.get('get_states').items():
        axes[1].plot(time_axis, array[:, n_ieth_hh], label = state)
    plt.legend()
    plt.show()

# %%

# TODO
# Make some appliance probabilistic in the activity
# Add the load patterns to ActivityApplianceSimulator
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Make a visualization of the activities like in bottacioli
# Define what to do for appliance that do not depend on activity