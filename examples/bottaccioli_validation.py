""" Attempt to reproduce Bottacioli paper.

doi 10.1109/ACCESS.2018.2886201
"""
# %%

import os
import sys


import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from demod.utils.plotters import FIGSIZE
from demod.utils.appliances import merge_appliance_dict
from demod.simulators.util import sample_population
from demod.simulators.appliance_simulators import ActivityApplianceSimulator, ProbabiliticActivityAppliancesSimulator, OccupancyApplianceSimulator
from demod.simulators.base_simulators import SimLogger
from demod.datasets.Germany.loader import GermanDataHerus
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator, SemiMarkovSimulator, MarkovChain1rstOrder

# %%
n_households = 5000

data = GermanDataHerus(version='vBottaccioli')

hh_subgroups, probs, _ = data.load_population_subgroups()


n_hh_list = sample_population(n_households, probs)

sim = SubgroupsIndividualsActivitySimulator(
    hh_subgroups,
    n_hh_list,
    logger=SimLogger('get_activity_states', 'current_time'),
    subsimulator=SemiMarkovSimulator,
    data=data,
    use_week_ends_days=True
)
# Simulates appliances using the CREST method
sim_CREST = OccupancyApplianceSimulator(
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    data=data,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)
# Simulates appliances using the Bottaciolli method
sim_app = ActivityApplianceSimulator(
    n_households, initial_activities_dict=sim.get_activity_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)
# Simulates probablitstics appliances using the Bottaciolli method
sim_prob_app = ProbabiliticActivityAppliancesSimulator(
    n_households, initial_activities_dict=sim.get_activity_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)

# %%

for i in range(2*144):

    for j in range(10):
        sim_app.step(sim.get_activity_states())
        sim_prob_app.step(sim.get_activity_states())
        sim_CREST.step(sim.get_active_occupancy())
    sim.step()


# %%
dict_states = sim.logger.get('get_activity_states')
time_axis = sim.logger.get('current_time')

power_consumptions = sim_app.logger.get('get_current_power_consumptions')
power_consumptions_prob = sim_prob_app.logger.get('get_current_power_consumptions')
power_consumptions_crest = sim_CREST.logger.get('get_current_power_consumptions')
time_axis_app = sim_app.logger.get('current_time')

# %%

from demod.utils.plotters import FIGSIZE, plot_stacked_activities, plot_appliance_consumptions
# Could make functions out of these plots


# Plot the appliances
fig, axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
plt.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0)

app_dic = merge_appliance_dict(sim_app.appliances, sim_prob_app.appliances)

plot_appliance_consumptions(
    np.sum(np.append(  # Merge power consumption of the two sims
        power_consumptions,
        power_consumptions_prob,
        axis=2
    ), axis=1),
    app_dic,
    time_axis=time_axis_app,
    differentiative_factor='type',
    ax=axes[0]
)

plot_appliance_consumptions(
    np.sum(power_consumptions_crest, axis=1),
    sim_CREST.appliances,
    time_axis=time_axis_app,
    differentiative_factor='type',
    ax=axes[1]
)

# Plots the different activities

# plot_stacked_activities(
#     {key: arrays for key, arrays in dict_states.items()},
#     time_axis=time_axis,
#     ax=axes[2]
# )
plt.show()

# %%

# TODO
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Define what to do for appliance that do not depend on activity
