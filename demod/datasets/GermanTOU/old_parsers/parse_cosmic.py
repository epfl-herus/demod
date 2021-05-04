#%%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from..helpers import count
#%%
df_cossmic = pd.read_csv(os.path.join( 'daten', 'household_data_1min_singleindex.csv'))
#%%

for to_plot in df_cossmic.columns[31:-1]:
    mask = pd.notna(df_cossmic[to_plot])
    n_minutes = np.sum(mask)
    n_years = n_minutes/60/24/365
    arr = np.array(df_cossmic[to_plot][mask])
    total_kwh = arr[-1] - arr[0]
    arr[1:] -= arr[:-1].copy() # gets the instantatneous consumption from the cumulative
    arr *= 60 * 1000 # kwh/min to watts
    # plt.plot(arr[5500:6000])
    print(to_plot, ' kwh/y ', total_kwh/n_years)


# %%
to_daily = 'DE_KN_residential6_grid_import'
df_cossmic[to_daily]
mask = pd.notna(df_cossmic[to_daily])
arr = np.array(df_cossmic[to_daily][mask])
arr[1:] -= arr[:-1].copy() # gets the instantatneous consumption from the cumulative
if to_daily == 'DE_KN_residential3_grid_import':
    arr[-2:] = arr[-3] # outlier values corrected
if to_daily == 'DE_KN_residential6_grid_import':
    arr[0] = arr[1] # outlier value corrected
arr *= 60 * 1000 # kwh/min to watts
plt.plot(arr)
# %%
