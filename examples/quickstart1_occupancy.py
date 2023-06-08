"""
OCCUPANCY SIMULATOR
"""

import matplotlib
import matplotlib.pyplot as plt
import datetime

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from demod.simulators.base_simulators import SimLogger
from demod.datasets.GermanTOU.loader import GTOU
from demod.datasets.OpenPowerSystems.loader import OpenPowerSystemClimate

from demod.simulators.crest_simulators import Crest4StatesModel
from demod.simulators.weather_simulators import RealClimate
from demod.simulators.lighting_simulators import FisherLightingSimulator

#%% Initialization

# number of households
n_households = 100
# Start of the simulation
start_datetime = datetime.datetime(2014, 3, 1, 0, 0, 0)

occupancy_sim = Crest4StatesModel(
    n_households,
    data=GTOU('4_States'),  # Time of use survey for germany
    start_datetime=start_datetime,  # Specifiy the start of the simulaiton
    logger=SimLogger('get_active_occupancy')
)

#%%  Simulation

n_days = 2
for _ in range(n_days*24*6):
    # step size of 10 minutes
    occupancy_sim.step()

#%% Plotting

fig, ax = plt.subplots()

ten_minute_axis = [
    start_datetime + datetime.timedelta(minutes=10*i)
    for i in range(n_days * 24 * 6)
]


ax.step(
    ten_minute_axis, occupancy_sim.logger.get('get_active_occupancy'),
    where='post', label='Active occupants', color='green'
)

time_xaxis = [
    start_datetime + datetime.timedelta(minutes=i)
    for i in range(0, n_days * 24 * 60 + 1, 60 * 6)
]

ax.legend()
ax.set_xticks(time_xaxis)
ax.set_xticklabels([f'{t.hour}:{t.minute}' for t in time_xaxis])

plt.show()

