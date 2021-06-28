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
n_households = 1000

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
    subgroups_list=hh_subgroups,
    n_households_list=n_hh_list,
    equipped_sampling_algo="subgroup",
    data=data,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)
# Simulates appliances using the Bottaciolli method
sim_app = ActivityApplianceSimulator(
    n_households, initial_activities_dict=sim.get_activity_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroups_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)
# Simulates probablitstics appliances using the Bottaciolli method
sim_prob_app = ProbabiliticActivityAppliancesSimulator(
    n_households, initial_activities_dict=sim.get_activity_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroups_list=hh_subgroups,
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
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=FIGSIZE)
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




plt.show()

# %%
fig, axes = plt.subplots(1,1)

# Total consumption of appliances
cons_bot_total = np.sum(np.append(  # Merge power consumption of the two sims
        power_consumptions,
        power_consumptions_prob,
        axis=2
    ), axis=0).sum(axis=0)
cons_crest_total = np.sum(power_consumptions_crest, axis=0).sum(axis=0)

x_axis = np.arange(app_dic['number'])
offset = 0.3
indices_reordered = np.array([
    list(app_dic['name']).index(name)
    for name in sim_CREST.appliances['name']
])
plt.bar(x_axis, height=cons_bot_total[indices_reordered], width=offset, label='Activity based')
plt.bar(x_axis + offset, height=cons_crest_total, width=offset, label='Occupancy based')
plt.xticks(x_axis + offset, app_dic['name'][indices_reordered], rotation=30)
plt.legend()
plt.show()
# %% plot in percentage
fig, axes = plt.subplots(1,1)

# Total consumption of appliances grouped by categories

categories_names = {
    'Desk': [
        'fixed_computer',
        'laptop_computer',
        'printer'
    ],
    'TV_audio': [ 'blueray_console',  'tv', 'tv_box', 'gaming_console', 'tablet'],
    'Cooking': [ 'electric_hob',  'gaz_hob', 'microwave', 'oven', 'toaster'],
    'Fridge': [ 'fridge'],
    'Light': [],
    'Drying': ['dryer'],
    'CirculationPump': [],
    'Dishwahsing': ['dishwasher'],
    'WashingMachine': ['washingmachine', 'washer_dryer'],
    'Freezer': ['freezer'],
    'Other': ['fixed_phone',  'iron', 'kettle', 'vacuum_cleaner',],
}

x_axis = np.arange(len(categories_names))
# Finds the consumptions based on the types
cat_consumption_bot = [
    np.sum(cons_bot_total[np.isin(app_dic['type'], cat_types)])
    for category, cat_types in categories_names.items()
]
cat_consumption_crest = [
    np.sum(cons_crest_total[np.isin(sim_CREST.appliances['type'], cat_types)])
    for category, cat_types in categories_names.items()
]
offset = 0.3
indices_reordered = np.array([
    list(app_dic['name']).index(name)
    for name in sim_CREST.appliances['name']
])
plt.bar(x_axis, height=cat_consumption_bot/sum(cat_consumption_bot), width=offset, label='Activity based')
plt.bar(x_axis + offset, height=cat_consumption_crest/sum(cat_consumption_crest), width=offset, label='Occupancy based')
plt.xticks(x_axis + offset, categories_names.keys(), rotation=30)
plt.legend()
plt.show()

# %%

# Plots the different activitie
plot_stacked_activities(
    {key: arrays for key, arrays in dict_states.items()},
    time_axis=time_axis,
)

# TODO
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Define what to do for appliance that do not depend on activity

# %%

# plt.figure()
# for key, profile in data.load_real_profiles_dict('full')['fridge'].items():
#     plt.plot(profile, label=key)
# plt.legend()
# plt.show()
# %%
