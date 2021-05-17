import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import demod

from demod.simulators.load_simulators import LoadSimulator
# Import the DatasetLoader
from demod.datasets.CREST.loader import Crest

from demod.simulators.base_simulators import SimLogger

sim = LoadSimulator(n_households=100, data=Crest())

sim = LoadSimulator(
    n_households=1, data=Crest(),
    logger=SimLogger('current_time', 'get_power_demand', 'get_temperatures')
)

for i in range(24 * 60):
    sim.step()

# Plots all the logged data one by one
sim.logger.plot()
# plots all the data in column
sim.logger.plot_column()
# Gets array of the data, can be used for your own purpose
elec_cons = sim.logger.get('get_power_demand')