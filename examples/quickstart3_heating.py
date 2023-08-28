"""
HEATING DEMAND EXAMPLE
"""


import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# comment it if you dowloaded demod from GitHub
import demod

from demod.simulators.base_simulators import SimLogger
from demod.datasets.GermanTOU.loader import GTOU
from demod.datasets.OpenPowerSystems.loader import OpenPowerSystemClimate
from demod.datasets.Germany.loader import GermanDataHerus

from demod.simulators.sparse_simulators import SparseTransitStatesSimulator
from demod.simulators.weather_simulators import RealClimate
from demod.simulators.lighting_simulators import CrestLightingSimulator
from demod.simulators.heating_simulators import  FiveModulesHeatingSimulator
from demod.simulators.appliance_simulators import OccupancyApplianceSimulator
from demod.simulators.weather_simulators import RealInterpolatedClimate


#%% INITIALIZATION

# Inputs
data = GermanDataHerus(version='v0.1')
time_ = datetime.datetime(2018, 1, 1, 0, 0, 0) # starting time of simulation
days = 3 # number of days to be simulated
n_households = 1 # number of households to be simulated
subgroup = {
    'n_residents': 5,
    'household_type': 4, 
} # n° residents should be consisten with the household type
# Here the available household types:
# 1 = One person household
# 2 = Couple without kids
# 3 = Single Parent with at least one kid under 18 and the other under 27
# 4 = Couple with at least one kid under 18 and the other under 27
# 5 = Others

# Occupancy
occ_sim = SparseTransitStatesSimulator(
    n_households, subgroup, data,
    logger=SimLogger('get_active_occupancy', 'get_occupancy'),
    start_datetime=time_
)


# Climate
climate_sim = RealInterpolatedClimate(
    data,
    start_datetime=time_,
    logger=SimLogger(
        'get_irradiance', 'get_outside_temperature'
    ),
    # choose the one minute step size
    step_size = datetime.timedelta(minutes=1),
)


# Heating system 
# It integrates (1) heating demand, (2) thermostat, (3) building thermal 
# dynamics, (4) heating system control and (5) operation
heating_sim = FiveModulesHeatingSimulator(
    n_households=n_households,
    initial_outside_temperature=climate_sim.get_outside_temperature(),
    # The algo that computes the heat demand like in CREST
    heatdemand_algo='heat_max_emmiters',
    logger=SimLogger(
        'get_temperatures',
        'get_dhw_heat_demand',
        'get_sh_heat_demand',
        'get_heat_outputs',
        'get_power_demand',
    )
)


# Appliances, water fixtures 
app_sim = OccupancyApplianceSimulator(
    [subgroup], [n_households], data=data,
    initial_active_occupancy=occ_sim.get_active_occupancy(),
    start_datetime=time_,
    logger=SimLogger(
        'get_power_demand', 'get_dhw_heating_demand',
        'get_current_power_consumptions', aggregated=False )
)


# Lighting
lighting_sim = CrestLightingSimulator(
    n_households=n_households,
    data=data,
    logger=SimLogger('get_power_demand'),
    bulbs_sampling_algo='randn'
)




#%% SIMULATION

appliance_usage = []

for _ in range(24*days*6):
    # every 10 minutes
    occ_sim.step()
    
    for __ in range(10):
        # every 1 minute
        climate_sim.step()
        
        app_sim.step(
            occ_sim.get_active_occupancy()
        )
        lighting_sim.step(
            occ_sim.get_active_occupancy(),
            climate_sim.get_irradiance()
        )
        heating_sim.step(
            climate_sim.get_outside_temperature(),
            climate_sim.get_irradiance(),
            app_sim.get_dhw_heating_demand(),
            occ_sim.get_thermal_gains(),
            lighting_sim.get_thermal_gains(),
            app_sim.get_thermal_gains(),
            external_target_temperature={'space_heating':20},
        )
        # you can pass different target temperature profiles
        # here it is considered a constant temperature of 20°C 
        
        # store appliance usage
        appliance_usage.append(app_sim.get_current_usage())
        



# %% PLOTTING

# load the data for plotting
act_occ = occ_sim.logger.get('get_active_occupancy')
light_cons = lighting_sim.logger.get()
app_cons = app_sim.logger.get('get_current_power_consumptions')
dhw_cons = app_sim.logger.get('get_dhw_heating_demand')
temperatures = heating_sim.logger.get('get_temperatures')
hp_power_cons = heating_sim.logger.get('get_power_demand')
usage = np.asarray(appliance_usage)


# helpers for the plots - time axis
minutes_axis = np.array(
    [time_ + datetime.timedelta(minutes=i) for i, _ in enumerate(app_cons)]
)
ten_minutes_axis = np.array(
    [time_ + datetime.timedelta(minutes=10*i) for i, _ in enumerate(act_occ)]
)


# style 
font = {'family' : 'Arial',
        'weight': 'normal'
        }

plt.rc('font', **font)

SMALLER_SIZE = 23
SMALL_SIZE = 25
MEDIUM_SIZE = 27
BIGGER_SIZE = 28
SMALLER_SIZE = 23

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

framealpha = 0.8

# plot
fig, axs = plt.subplots(2, sharex=True,  gridspec_kw={'hspace': 0.05, 'height_ratios':[1,1]}, figsize=(20,13))

linewidth = 3

# APPLIANCE USAGE -------------------------------------------------------------
# plot occupancy
n = 0
axs[n].step(ten_minutes_axis, act_occ, color='g', linewidth=4, label='active occ.')
axs[n].set_ylabel('Occupancy [n° active occ.]')
# axs[n].set_yticks()
# axs[n].set_yticklabels([0,1,2,3,4])
axs[n].legend(loc='upper left', framealpha=1)

new_ax0 = axs[n].twinx()
times, households, appliance = np.where(usage)
wattage = app_sim.appliances['mean_elec_consumption'][appliance]
appliances_newnames = {
    'FRIDGE1':'FRI1',
    'FRIDGE2':'FRI2',
    'FREEZER1':'FRE1', 
    'FREEZER2':'FRE2', 
    'PHONE':'PHON',  
    'IRON':'IRON',
    'VACUUM':'VACU', 
    'PC1':'PC1', 
    'PC2':'PC2', 
    'LAPTOP1':'LAP1', 
    'LAPTOP2':'LAP2', 
    'TABLET':'TABL', 
    'PRINTER':'PRIN',
    'TV1':'TV1', 
    'TV2':'TV2', 
    'VCR_DVD':'DVD', 
    'RECEIVER':'REC', 
    'CONSOLE':'CONS', 
    'HOB_ELEC':'HOB1',
    'HOB_GAZ':'HOB2', 
    'OVEN':'OVEN', 
    'MICROWAVE':'MICR', 
    'KETTLE':'KETT', 
    'SMALL_COOKING':'COOK',
    'DISH_WASHER':'DW', 
    'TUMBLE_DRYER':'TD', 
    'WASHING_MACHINE':'WM', 
    'WASHER_DRYER':'WD',
    'BASIN':'BASI', 
    'SINK':'SINK', 
    'SHOWER':'SHOW', 
    'BATH':'BATH',
    }

mask = [i not in ['BASIN', 'SINK', 'SHOWER', 'BATH'] for i in app_sim.appliances['name'][appliance]]
inv_mask = [i in ['BASIN', 'SINK', 'SHOWER', 'BATH'] for i in app_sim.appliances['name'][appliance]]


# plot water fixtures   
watts_scatter = new_ax0.scatter(
    minutes_axis[times][inv_mask],
    [' ' for i in app_sim.appliances['name'][appliance][inv_mask]],
    s=80, c='blue', alpha = 0
)

watts_scatter = new_ax0.scatter(
    minutes_axis[times][inv_mask],
    [appliances_newnames[i] for i in app_sim.appliances['name'][appliance][inv_mask]],
    s=80, c='blue', label ='water fixtures' 
)


new_ax0.yaxis.set_tick_params(labelsize=SMALLER_SIZE)

lines_1, labels_1 = axs[n].get_legend_handles_labels()
lines_2, labels_2 = new_ax0.get_legend_handles_labels()

lines = lines_1 + lines_2
labels = labels_1 + labels_2

new_ax0.legend(lines, labels, loc='upper left', framealpha=framealpha)


# plot heat pump power demand
n += 1
axs[n].fill_between(minutes_axis, hp_power_cons, color='r', alpha=0.2, label='$P_{heat\:pump}$')

label = {
    'interior':'$T_{indoor\:air}$',
    'emitter':'$T_{emitters}$',
    'cylinder':'$T_{cylinder}$',
    'cold_water':'$T_{cold\:water}$',
    'building':'$T_{building}$',
    }

color = {
    'interior':'green',
    'emitter':'orange',
    'cylinder':'tomato',
    'cold_water':'deepskyblue',
    'building':'sienna',
    }

# Plot temperature profiles
new_ax2 = axs[n].twinx()
for key, serie in temperatures.items():
    new_ax2.step(minutes_axis, serie, 
                     c=color[key], 
                     linewidth=linewidth,
                     label=label[key]
                     )

new_ax2.step(
    minutes_axis, climate_sim.logger.get('get_outside_temperature'),
    c='blue', linewidth=linewidth, label='$T_{outdoor\:air}$', 
)

axs[n].set_yticks(range(0,1501,500))
axs[n].set_yticklabels([0.0,0.5,1.0,1.5])

axs[n].set_ylabel('Power demand [kW]')
new_ax2.set_ylabel('Temperature [°C]')

lines_1, labels_1 = axs[n].get_legend_handles_labels()
lines_2, labels_2 = new_ax2.get_legend_handles_labels()

lines = lines_1 + lines_2
labels = labels_1 + labels_2

new_ax2.legend(lines, labels, loc='upper left', framealpha=framealpha)


for ax in axs:
    ax.xaxis.grid()

axs[n].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


plt.show()


