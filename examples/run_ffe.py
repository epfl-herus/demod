"""Households with an average electric consumption were selected as
examples for major household types from a simulated settlement of 940
households representative for Germany. Care was also taken to ensure
that thermal consumption and annual kilometrage were as average as
possible. The period is one year and the temporal resolution is one
minute. Corresponding datasets for thermal load profiles and mobility
profiles are available.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from matplotlib.lines import Line2D
from demod.datasets.GermanTOU.loader import GTOU
from demod.simulators.base_simulators import SimLogger
from demod.datasets.FfE_Open_Data import FfE_DataLoader
from demod.simulators.sparse_simulators import SubgroupsActivitySimulator
from demod.simulators.appliance_simulators import OccupancyApplianceSimulator
import numpy as np
import matplotlib.pyplot as plt


data = FfE_DataLoader(allow_pickle=True)
ffe_profiles, ffe_labels = data.load_electric_load_profiles(return_labels=True)
#
# profiles, labels = data.load_thermal_load_profiles(return_labels=True)
# print(profiles)
# print(data.load_labels_name([lab['hh_type'] for lab in labels]))
# print(data.get_heating_load_profiles())



corresponding_gtou_subgroups = [
    {'n_residents': 1, 'hh_work_type': '1_fulltime'},  # 'One full-time working person'
    {'n_residents': 1, 'hh_work_type': '1_retired'},  # 'One pensioneer'
    {'n_residents': 2, 'hh_work_type': '2_fulltime'},  # 'Two full-time working persons'
    {'n_residents': 2, 'hh_work_type': '2_retired'},  # 'Two pensioneers'
    {'n_residents': 2, 'hh_work_type': '1_fulltime_1_halftime'},  # 'One full-time and one part-time working person'
    {'n_residents': 3, 'household_type': 4, 'hh_work_type': '2_fulltime'},  # 'Two full-time working persons, one child'
    {'n_residents': 3, 'household_type': 4, 'hh_work_type': '1_fulltime_1_halftime'},  # 'One full-time and one part-time working person, one child'
    {'n_residents': 4, 'household_type': 4, 'hh_work_type': '2_fulltime'},  # 'Two full-time working persons, two children'
    {'n_residents': 4, 'household_type': 4, 'hh_work_type': '1_fulltime_1_halftime'},  # 'One full-time and one part-time working person, two children'
    {'n_residents': 5, 'household_type': 4, 'hh_work_type': '2_fulltime'},  # ' Two full-time working persons, three children'
    {'n_residents': 5, 'household_type': 4, 'hh_work_type': '1_fulltime_1_halftime'},  # ' One full-time and one part-time working person, three children'
]

n_hh_sim = 50
sim_act = SubgroupsActivitySimulator(
    corresponding_gtou_subgroups,
    [n_hh_sim for i in corresponding_gtou_subgroups],
)

sim_app = OccupancyApplianceSimulator(
    corresponding_gtou_subgroups,
    [n_hh_sim for i in corresponding_gtou_subgroups],
    initial_active_occupancy=sim_act.get_active_occupancy(),
    logger=SimLogger('get_energy_consumption', aggregated=False)
)

n_days = 3
for i in range(n_days*24*6):
    sim_act.step()
    for i in range(10):
        sim_app.step(sim_act.get_active_occupancy())

sim_app.logger.plot(aggregate=True)


demod_profiles = sim_app.logger.get('get_energy_consumption')


median_profiles = []
for i in range(11):
    subgroup_profiles = demod_profiles[:, i*50:(i+1)*50]
    med_ind = np.argsort(np.sum(subgroup_profiles,axis=0))[25]
    median_profiles.append(subgroup_profiles[:, med_ind])



plt.title('Total yearly consumption')
for i, (demod_profile, ffe_profile, label) in enumerate(zip(median_profiles, ffe_profiles, ffe_labels)):
    print(np.sum(demod_profile), np.sum(ffe_profile))
    # convert w*min to kwh
    dmd_line = plt.bar(i-0.2, np.sum(demod_profile)/ 6000., 0.4, color='orange')
    ffe_line = plt.bar(i+0.2, np.sum(ffe_profile)/ 6000., 0.4, color='blue')
plt.legend(
    handles=[
        Line2D([0], [0], color='orange', label='Demod'),
        Line2D([0], [0], color='blue', label='FFE'),]
)
plt.ylabel('Total yearly consumption [kWh]')

plt.xlabel('Household types')
plt.show()


minutes_axis = np.linspace(0, 24, 1440)
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(8,8) )

plt.subplots_adjust(hspace=0)
fig.suptitle('Daily consumptions, Top: Demod, Down: FFE')
# computes the means of the days
dmd_daily_averages = [np.mean(prof.reshape((n_days, -1)), axis=0) for prof in median_profiles]
axes[0].stackplot(minutes_axis, dmd_daily_averages)

ffe_daily_averages = [np.mean(prof.reshape((365, -1)), axis=0) for prof in ffe_profiles]
axes[1].stackplot(minutes_axis, np.roll(ffe_daily_averages, - 4*60, axis=1))
axes[1].set_ylabel('Consumption [W]')

[ax.grid() for ax in axes]

plt.show()