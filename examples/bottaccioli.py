""" Attempt to reproduce Bottacioli paper.

doi 10.1109/ACCESS.2018.2886201
"""
# %%
import os
import sys
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import Colormap

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

    for i in range(10):
        sim_app.step(sim.get_states())
        sim_prob_app.step(sim.get_states())
        sim_CREST.step(sim.get_active_occupancy())
    sim.step()


# %%
dict_states = sim.logger.get('get_states')
time_axis = sim.logger.get('current_time')

power_consumptions = sim_app.logger.get('get_current_power_consumptions')
power_consumptions_prob = sim_prob_app.logger.get('get_current_power_consumptions')
time_axis_app = sim_app.logger.get('current_time')


sim_CREST.logger.plot()
# %%
for n_ieth_hh in range(n_households):

    # Plot the appliances
    fig, axes  = plt.subplots(2,1, sharex=True)
    for i, name in enumerate(sim_app.appliances['name']):
        # plots each appliance pattern
        if sim_app.available_appliances[n_ieth_hh, i]:
            axes[0].step(time_axis_app, power_consumptions[:, n_ieth_hh, i], label=name)
    for i, name in enumerate(sim_prob_app.appliances['name']):
        # plots each appliance pattern
        if sim_prob_app.available_appliances[n_ieth_hh, i]:
            axes[0].step(time_axis_app, power_consumptions_prob[:, n_ieth_hh, i], label=name)
    axes[0].legend()

    # Plots the different activities
    max_number = 1
    colors = np.array(['','#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba'])
    for i, (state, array) in enumerate(dict_states.items()):
        mask_activity_occuring = array[:, n_ieth_hh] > 0
        axes[1].scatter(
            time_axis[mask_activity_occuring],
            i * np.ones(sum(mask_activity_occuring)),
            c = colors[array[:, n_ieth_hh][mask_activity_occuring]]
            )
        # Records the max number of residents
        max_number = max(max_number, max(array[:, n_ieth_hh]))
    axes[1].set_yticks(np.arange(0, len(dict_states.keys())))
    axes[1].set_yticklabels(dict_states.keys())
    # Adds the legend

    axes[1].legend(handles=[
        Circle((0,0), color=colors[i], label='{}'.format(i))
        for i in range(1, max_number+1)
    ])
    plt.show()

# %%

# TODO
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Define what to do for appliance that do not depend on activity
