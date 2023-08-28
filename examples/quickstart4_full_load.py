"""
Script for testing the fully integrated load simulator v2
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import datetime 
import time

original_path = sys.path

if os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') not in sys.path:
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from demod.simulators.base_simulators import SimLogger
from demod.simulators.load_simulators import LoadSimulator
from demod.datasets.Germany.loader import GermanDataHerus


# %% INPUTS

# dataset for the parametrization
data = GermanDataHerus(version='v0.1')

# number of households
n_households = 100

# number of days to be simulated
n_days = 7

# starting time
start_datetime = datetime.datetime(2014, 1, 1, 4, 0, 0)


# %% INITIALIZATION

# initialize the simulator
sim = LoadSimulator(
    n_households=n_households,
    start_datetime=start_datetime,
    data = data
)


# %% SIMULATION

time_axis = []
time_axis_act = []
appliance_consumption = []
heating_consumption = []
temperatures = {}

# start time simulation
start = time.time()

for i in range(60*24*n_days):
    # step every minute
    sim.step()
    
    # store the results
    appliance_consumption.append(sum(sim.get_appliance_power_demand()))
    heating_consumption.append(sum(sim.get_heating_power_demand()))
    
    for key in sim.heating.get_temperatures().keys():
        if key not in temperatures.keys():
            temperatures[key] = []
        temperatures[key].append(np.mean(sim.heating.get_temperatures()[key]))
        
    time_axis.append(sim.current_time)
    
    # keep track of 
    if i % (6*60) == 0:
        print(f'day {i//(60*24)} - hour {(i//60)%24}')
    
# end time simulation
end = time.time()
print(f'required time for simulating {n_households} households for {n_days} days: {end - start} sec')        


# %% PLOTTING 

time_axis = [start_datetime + i * datetime.timedelta(minutes=1) 
             for i in range(60*24*n_days)]

# load profiles
fig, axs = plt.subplots(figsize=(12,4))
axs.plot(time_axis, appliance_consumption, label='appliances')
axs.plot(time_axis, heating_consumption, label='heating')
plt.legend()
plt.tight_layout()
plt.show()

# temperature profiles
fig, axs = plt.subplots(figsize=(12,4))
for k, v in temperatures.items():
    axs.plot(time_axis, v, label=k)
plt.legend()
plt.tight_layout()
plt.show()


#%% RESTORE THE PATH

# clean sys path
if os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') in sys.path:
    sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    
final_path = sys.path
path_check = original_path == final_path
print(f'the path has not changed: {path_check}')

