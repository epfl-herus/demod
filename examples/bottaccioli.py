""" Attempt to reproduce Bottacioli paper.

doi 10.1109/ACCESS.2018.2886201
"""
# %%
from importlib import reload
import os
import sys
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.colors import Colormap

import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from demod.utils.plotters import FIGSIZE, plot_household_activities
from demod.utils.appliances import merge_appliance_dict
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
    logger=SimLogger('get_activity_states', 'current_time', aggregated=False),
    subsimulator=SemiMarkovSimulator,
    data=data,
    use_week_ends_days=True
)

sim_app = ActivityApplianceSimulator(
    n_households, initial_activities_dict=sim.get_activity_states(),
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
    n_households, initial_activities_dict=sim.get_activity_states(),
    data=data,
    equipped_sampling_algo="subgroup",
    subgroup_list=hh_subgroups,
    n_households_list=n_hh_list,
    logger=SimLogger('current_time', 'get_current_power_consumptions', aggregated=False)
)

# %%

print(sim_app.appliances['use_variable_loads'])

for i in range(7*144):

    for i in range(10):
        sim_app.step(sim.get_activity_states())
        sim_prob_app.step(sim.get_activity_states())
        sim_CREST.step(sim.get_active_occupancy())
    sim.step()


# %%
dict_states = sim.logger.get('get_activity_states')
time_axis = sim.logger.get('current_time')

power_consumptions = sim_app.logger.get('get_current_power_consumptions')
power_consumptions_prob = sim_prob_app.logger.get('get_current_power_consumptions')
time_axis_app = sim_app.logger.get('current_time')


sim_CREST.logger.plot()
# %%

from demod.utils.plotters import FIGSIZE, plot_household_activities, plot_appliance_consumptions
# Could make functions out of these plots
for n_ieth_hh in range(n_households):

    # Plot the appliances
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)

    app_dic = merge_appliance_dict(sim_app.appliances, sim_prob_app.appliances)
    available_app = np.append(  # Merg the two sims results
        sim_app.available_appliances[n_ieth_hh, :],
        sim_prob_app.available_appliances[n_ieth_hh, :]
    )
    plot_appliance_consumptions(
        np.c_[  # Merge power consumption of the two sims
            power_consumptions[:, n_ieth_hh, :],
            power_consumptions_prob[:, n_ieth_hh, :]
        ],
        app_dic,
        time_axis=time_axis_app,
        differentiative_factor='name',
        labels_list=app_dic['name'][available_app],
        ax=axes[0]
    )

    # Plots the different activities

    plot_household_activities(
        { key: arrays[:, n_ieth_hh] for key, arrays in dict_states.items()},
        time_axis=time_axis,
        ax=axes[1]
    )
    plt.show()

# %%

# TODO
# Control the activities of the appliances and the ones of the TOU with the ones of Bottaccioli
# Define what to do for appliance that do not depend on activity
