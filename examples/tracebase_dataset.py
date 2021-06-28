# %%
from datetime import timedelta
import matplotlib.pyplot as plt
from time import sleep
import os
import sys

import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from demod.datasets.tracebase.loader import Tracebase

data = Tracebase(step_size=timedelta(seconds=10))
# %%
load_dict = data.load_real_profiles_dict()
# %% part of the code used for analysing the loads

for key, load in load_dict['dryer'].items():
    mask= np.where(load > 6)[0]
    if len(mask)>2 and key:
        a,b = mask[[0,-1]]
        plt.plot(load)
        plt.vlines([a,b], 0, 2000, 'red')
        plt.title(key)
        print(key)
        plt.show()


# %%

load_dict = data.load_real_profiles_dict('switchedON')
for key in load_dict.keys():
    device_names = []  # stores plotted device names
    for i, (dev_full_name, val) in enumerate(load_dict[key].items()):
        splitted = dev_full_name.split('_')
        # print(len(val))
        dev_split_name = (
            splitted[0] + splitted[1] if len(splitted) == 3 else splitted[0]
        )
        if dev_split_name not in device_names:
            device_names.append(dev_split_name)
            plt.plot(val, label=dev_full_name)

    plt.title(key)
    plt.legend()
    plt.show()
# %%
