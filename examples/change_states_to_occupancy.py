# %%
import os
import sys


import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from demod.utils.plotters import plot_stack_states
from demod.datasets.GermanTOU.parser import *
from demod.utils.parse_helpers import states_to_transitions
# %%

def states_to_occupancy_new(primary_states, secondary_states, initial_occupancy, final_occupancy, dic_home):
    """Convert the states form the German TOU to occupancy. As the occupancy is not given in the german TOU,
    this uses an algorithm that estimates the occupancy.

    Args:
        primary_states (ndarray): The raw states form the GTOU
        secondary_states (ndarray): The raw secondary states form the GTOU
        initial_occupancy (ndarray): The initial occupancy form the GTOU
        final_occupancy (ndarray): The final occupancy form the GTOU
        dic_home (dict): a dictionary with a factor stating the suceptibility of being home doing that activity

    Returns:
        ndarray: The occupancy states

    Note:
        There is no guarantee on the validity of this, but results seems ok
    """
    # convert the states to their home indice rating
    main_rating, main_lab     = convert_states(primary_states, dic_home)
    secondary_rating, sec_lab = convert_states(secondary_states, dic_home)
    home_ratings = main_lab[main_rating] + sec_lab[secondary_rating]

    #initialize the occupancy
    current_occupancy = np.zeros_like(initial_occupancy, dtype=bool)
    # assign occupants at home
    current_occupancy[initial_occupancy == 1] = True
    # missing must check the current states
    current_occupancy[initial_occupancy == -1] = home_ratings[:, 0][initial_occupancy == -1] >= 0  # favor being home with the =0, as we start at 4:00
    # current_occupancy[initial_occupancy == 2] = False # from inital vector

    # find out where the last travel occurs so that the state can easily be determined
    # (take the last index where ther was a transportation (>=900)
    # or  if there was no travel)
    mask_transportation = (
        is_transportation(primary_states)
        | is_transportation(secondary_states)
    )
    last_travel_indexes = np.array([
        -10 if  # -10(unreachable later)
        len(indices_transp := np.where(mask_t)[0]) == 0
        else indices_transp[-1]  # The last time index of transportation
        for mask_t in mask_transportation
    ], dtype=int)

    occupancies = []
    # Will record who was travelling the step before
    mask_was_travelling = np.zeros_like(current_occupancy)
    mask_after_last_travel = np.zeros_like(current_occupancy)
    mask_after_last_travel[last_travel_indexes < 0] = True
    # Will record the location before the last travel occured
    mask_at_home_before_last_travel = np.zeros_like(current_occupancy)
    previous_occupancy = np.array(current_occupancy)

    # Iterates through time, to record who leaves and who stays
    for i, (home_rating, mask_travel) in enumerate(zip(
        home_ratings.T, mask_transportation.T
    )):

        # people who start changing location
        mask_start_moving = (~mask_was_travelling) & mask_travel
        # save state of before last travel
        mask_at_home_before_last_travel[mask_start_moving] = (
            previous_occupancy[mask_start_moving]
        )
        current_occupancy[mask_start_moving] = False  # If moving, cant be home
        current_occupancy[mask_travel] = False  # If moving, cannot be home

        # people who finish a travel can be at home or not
        mask_finish_travel = mask_was_travelling & (~mask_travel)
        # check if activity can be perform at home,
        # also take into account the previous occupancy and imagine
        # it should change (home -> trans -> outofhome act -> trans -> home)
        previous_occ_bias = (
            -2 * mask_at_home_before_last_travel[mask_finish_travel] + 1
        )  # true->-1, false->1
        # Comes back home if rating of being home is good enough
        current_occupancy[mask_finish_travel] = (
            home_rating[mask_finish_travel] + previous_occ_bias) >= 0

        # if it is the last travel, set to the last occupancy
        mask_after_last_travel[last_travel_indexes == (i-1)] = True
        current_occupancy[mask_after_last_travel & (final_occupancy == 1)] = True
        current_occupancy[mask_after_last_travel & (final_occupancy == 2)] = False

        mask_was_travelling = np.array(mask_travel)
        previous_occupancy = np.array(current_occupancy)
        occupancies.append(np.array(current_occupancy))

    return np.asarray(occupancies).T


occ_new = states_to_occupancy_new(primary_states, secondary_states, initial_location, final_location, dic_home )
occ_old = occ

# %%
prof_old = occ_old + 2* act
prof_new = occ_new + 2* act
lab = ['not at home & not active', 'at home, not active', 'active not at home', 'at home and active']
plot_stack_states(prof_old, lab)
plot_stack_states(prof_new, lab)

#%%
print('-----')
print(np.bincount(np.bincount(states_to_transitions(occ_old, ignore_end_start_transitions=True)['persons'])))
print(np.bincount(np.bincount(states_to_transitions(occ_new, ignore_end_start_transitions=True)['persons'])))
print(np.bincount(np.bincount(states_to_transitions((
        is_transportation(primary_states)
        | is_transportation(secondary_states)
    ), ignore_end_start_transitions=True)['persons'])))
# %%
m = (initial_location != -1) & (final_location != -1)
print('-----Number of persons that end up in a different place')
print(sum(initial_location[m] != final_location[m]))
print(sum(occ_new[:,0] != occ_new[:, -1]))
print('-----')

# %%
