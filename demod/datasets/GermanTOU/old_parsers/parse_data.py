
# %%
import os, sys
import warnings

import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demod.simulators.simulators import OccupancySimulator, TimedStatesSimulator, rescale_pdf
from..helpers import get_states_durations, RMSE, CREST_get_24h_occupancy, count


# %%
df_hh = pd.read_csv('daten' + os.sep + 'zve13_puf_hh.csv', sep=';')
df_akt = pd.read_csv('daten' + os.sep + 'zve13_puf_takt.csv', sep=';')
df_pers = pd.read_csv('daten' + os.sep + 'zve13_puf_pers.csv', sep=';')



# gets the number of persons recorded in the household

# %% remove the unusable data

# missing data, fehl_tag means there is one day not in the records
#df_akt['fehl_tag'] > 0


# read the states from file
primary_states      = df_akt[['tb1_' + str(i) for i in range(1,145)]].to_numpy()
# get the secondary states
secondary_states    = df_akt[['tb2_' + str(i) for i in range(1,145)]].to_numpy()
# transportation mean
transportation_mean = df_akt[['tb3_' + str(i) for i in range(1,145)]].to_numpy()
# if the person was travelling that day
travelling          = np.array(df_akt['tc5'])
# wheter the participant started at home
initial_location    = np.array(df_akt['tc7'])
final_location      = np.array(df_akt['tc8'])
# 'trifft nicht zu' = -2 and means that the start location of the next day should be used
final_location[final_location==-2] = np.roll(initial_location, 1)[final_location==-2]

#%% get a link between df_akt and df_pers
df_akt_index_person = [np.where(df_pers['id_persx'] == i)[0][0] for i in df_akt['id_persx']]
df_hh_max_recorded_persons = [np.max(df_akt['persx'][df_akt['id_hhx']==i]) for i in df_hh['id_hhx']]


# %% convert states to transitions
def states_to_transitions(states):
    states = np.array(states).T

    old_states = states[0]


    transition_times = []
    transition_person = []
    transition_new_state = []
    transition_old_state = []


    for i, s in enumerate(states):
        # check that the state has really disappeared, and not that it changed
        mask_transition = old_states != s
        transition_times.append(np.full( np.sum(mask_transition), i))
        transition_person.append(np.where(mask_transition)[0])
        transition_new_state.append(np.array(s[mask_transition]))
        transition_old_state.append(np.array(old_states[mask_transition]))

        old_states = s

    transitions_dict = {}
    transitions_dict['times']        = np.concatenate(transition_times)
    transitions_dict['persons']       = np.concatenate(transition_person)
    transitions_dict['new_states']    = np.concatenate(transition_new_state)
    transitions_dict['old_states']    = np.concatenate(transition_old_state)

    return transitions_dict

def states_to_transitions_secondary(primary_states, secondary_states):
    p_states = np.array(primary_states).T
    s_states = np.array(secondary_states).T


    old_p_states = p_states[0]
    old_s_states = s_states[0]

    transition_times = []
    transition_person = []
    transition_new_state = []
    transition_old_state = []


    for i, (p, s) in enumerate(zip(p_states, s_states)):
        # check that the state has really disappeared, and not that it changed
        mask_transition_p = (old_p_states != p) & (old_p_states != s)
        mask_transition_s = (old_s_states != p) & (old_s_states != s)
        transition_times.append(np.full(np.sum(mask_transition_p) + np.sum(mask_transition_s), i))
        transition_person.append(np.where(mask_transition_s)[0])
        transition_person.append(np.where(mask_transition_p)[0])
        transition_new_state.append(np.array(p[mask_transition_p]))
        transition_new_state.append(np.array(s[mask_transition_s]))
        transition_old_state.append(np.array(old_p_states[mask_transition_p]))
        transition_old_state.append(np.array(old_s_states[mask_transition_s]))

        old_p_states = p
        old_s_states = s

    transitions_dict = {}
    transitions_dict['times']        = np.concatenate(transition_times)
    transitions_dict['persons']       = np.concatenate(transition_person)
    transitions_dict['new_states']    = np.concatenate(transition_new_state)
    transitions_dict['old_states']    = np.concatenate(transition_old_state)

    return transitions_dict



def group_hh_transitions(primary_states, secondary_states=None, household_indexes=np.array(df_akt['id_hhx']), days_indexes=np.array(df_akt['tagnr'])):

    if secondary_states is not None:
        transitions = states_to_transitions_secondary(primary_states, secondary_states)
    else:
        transitions = states_to_transitions(primary_states)

    # find and counts the transitions in the housholds

    (hh_times, hh_new_states, hh_old_states, hh_nrs, day_nrs), hh_counts = np.unique(np.array(
        [transitions['times'],
        transitions['new_states'],
        transitions['old_states'],
        household_indexes[transitions['persons']],
        days_indexes[transitions['persons']]]),
            axis=1, return_counts=True)

    hh_transitions = {}
    hh_transitions['times']         = hh_times
    hh_transitions['households']    = hh_nrs
    hh_transitions['new_states']    = hh_new_states
    hh_transitions['old_states']    = hh_old_states
    hh_transitions['counts']        = hh_counts
    hh_transitions['day_nrs']       = day_nrs
    return hh_transitions




#hh_transitions = group_in_household_transitions(primary_states, secondary_states)
hh_transitions = group_hh_transitions(primary_states, secondary_states=secondary_states)

# %% analysis of the data representation

fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,8))

ages_female = df_pers['alterx'][df_pers['ha3']==2]
ages_male   = df_pers['alterx'][df_pers['ha3']==1]



ax[0].yaxis.tick_right()
ax[0].hist(ages_male, bins=76, orientation='horizontal', color='royalblue')
ax[0].invert_xaxis()
ax[0].set_title('Men')
ax[0].set_xlim(150,0)
ax[0].set_frame_on(False)

ax[1].hist(ages_female, bins=76, orientation='horizontal', color='royalblue')
ax[1].set_title('Women')
ax[1].set_xlim(0,150)
ax[1].set_frame_on(False)

ax[0].grid(color='white')
ax[1].grid(color='white')

fig.text(0.5, 0.07, 'Number of samples in dataset', va='center', ha='center')
fig.subplots_adjust(wspace=0.123)
fig.show()

#%%  select only a subgroup from the dataset
def get_mask_subgroup(
    only_fully_recorded_household=False, remove_missing_days=False, only_household_identical_days=False,
    quarter=None, weekday=None,
    n_residents=None, household_type=None,
    life_situation=None, age=None, geburtsland=None, gender=None, household_position=None,
    is_travelling=None):

    _mask = np.ones(len(df_akt), dtype=bool)

    if only_fully_recorded_household:
        # check if the diary is in the list of household that have the number of records=number of people living their
        _mask &= np.isin(df_akt['id_hhx'],df_hh['id_hhx'][df_hh['ha1x']-df_hh_max_recorded_persons == 0])

    if remove_missing_days:
        hh_with_missing_day = np.array(df_akt['id_hhx'][df_akt['fehl_tag']!=0])[::2] # work-around trick as there are three days in general and only two if there is a missing day
        missing_day = np.array(df_akt['fehl_tag'][df_akt['fehl_tag']!=0])[::2]
        # remove households where there is a survey missing on a particular day
        _mask &= ~np.logical_or.reduce([ (df_akt['id_hhx'] == hh) & (df_akt['tagnr'] == day) for hh, day in zip(hh_with_missing_day,missing_day)])

    if only_household_identical_days:
        # only keep if the diary has been recorded on the same day for all occupants
        _mask &= (np.array(df_akt['selbtag']) == 1) | (np.array(df_hh['ha1x'][df_akt['id_hhx']-1]) == 1)

    if quarter:
        assert quarter <= 4 and quarter > 0, 'quarter value must be 1,2,3 or 4'
        _mask &= np.array(df_akt['quartal']) == quarter

    if weekday:
        if isinstance(weekday, list):
            _mask &= np.logical_or.reduce([get_mask_subgroup(weekday=i) for i in weekday])
        else:
            assert weekday <= 7 and weekday > 0, 'weekday value must be 1,2,3,4,5,6 or 7'
            _mask &= np.array(df_akt['wtagfei']) == weekday

    if n_residents:
        if isinstance(n_residents, list):
            _mask &= np.logical_or.reduce([get_mask_subgroup(n_residents=i) for i in n_residents])
        else:
            assert n_residents <= 6 and n_residents > 0, 'n_residents value must be 1,2,3,4,5 or 6'
            _mask &= np.array(df_hh['ha1x'][df_akt['id_hhx']-1]) == n_residents

    if household_type:
        assert household_type <= 6 and household_type > 0, 'household_type value must be 1,2,3,4,5 or 6'
        _mask &= np.array(df_hh['hhtyp'][df_akt['id_hhx']-1]) == household_type

    if life_situation:
        assert life_situation <= 11 and life_situation > 0, 'life_situation value must be 1,2,3,4,5 or 6'
        _mask &= np.array(df_pers['pb3'][df_akt_index_person]) == life_situation

    if household_position:
        assert household_position <= 9 and household_position > 0, 'household_position value must be 1,2,3,4,5,6,7,8 or 9'
        _mask &= np.array(df_pers['ha6x'][df_akt_index_person]) == household_position

    if age:

        assert isinstance(age, tuple), 'age arg must be a tuple'
        assert age[0] <= age[1], 'age first value must be less or equal than the second'
        ages = np.array(df_pers['alterx'][df_akt_index_person])
        _mask &= (ages >= age[0]) & (ages < age[1])

    if geburtsland:
        assert geburtsland <= 2 and geburtsland > 0, 'geburtsland value must be 1 or 2'
        _mask &= np.array(df_pers['ha8x'][df_akt_index_person]) == geburtsland

    if gender:
        assert gender <= 2 and gender > 0, 'gender value must be 1 or 2'
        _mask &= np.array(df_pers['ha3'][df_akt_index_person]) == gender


    # check not empty
    if np.sum(_mask) == 0:
        warnings.warn('Empty subgroup detected in select subgroups.', RuntimeWarning)

    return  _mask






# %%  various functions for transforming the data
# convert states to indices using a rule
# -2 is sure not home, 2 is sure home, 1,-1 are probably and 0 is unsure
dic_home = {
    0 : 0,
    110 : 1,
    120 : 1,
    131 : 1,
    132 : 1,
    139 : 2,
    210 : -1,
    220 : -1,
    230 : -1,
    241 : -1,
    242 : 0,
    243 : -1,
    244 : -1,
    245 : -1,
    249 : -1,
    311 : -2,
    312 : -2,
    313 : -2,
    314 : -2,
    315 : -2,
    317 : -2,
    319 : -2,
    321 : -2,
    329 : -2,
    330 : -2,
    341 : -2,
    349 : -2,
    353 : 1,
    354 : 1,
    361 : 1,
    362 : -2,
    363 : -2,
    364 : -2,
    369 : 0,
    411 : 2,
    413 : 2,
    412 : 2,
    414 : 2,
    419 : 2,
    421 : 2,
    422 : 2,
    423 : 2,
    429 : 2,
    431 : 2,
    432 : 2,
    433 : 2,
    434 : 2,
    439 : 2,
    441 : 2,
    442 : 2,
    443 : 2,
    444 : 2,
    445 : 2,
    446 : 2,
    449 : 2,
    451 : 2,
    452 : 2,
    453 : 2,
    454 : 2,
    455 : 2,
    459 : 2,
    461 : -2,
    464 : -1,
    465 : -1,
    466 : -1,
    469 : -1,
    471 : 2,
    472 : 2,
    473 : 2,
    474 : 2,
    475 : 2,
    476 : 2,
    479 : 2,
    480 : 0,
    491 : 1,
    492 : 1,
    499 : 1,
    510 : -1,
    520 : -1,
    531 : -2,
    532 : -1,
    539 : -1,
    611 : 0,
    612 : 0,
    621 : -2,
    622 : -2,
    623 : -2,
    624 : -2,
    625 : -2,
    626 : -2,
    627 : -2,
    629 : 0,
    630 : 0,
    641 : 0,
    642 : 0,
    649 : 0,
    711 : -2,
    712 : -2,
    713 : -2,
    715 : -2,
    716 : -2,
    717 : -2,
    719 : -2,
    730 : 0,
    740 : 1,
    752 : 0,
    759 : 0,
    761 : 0,
    762 : 0,
    763 : 1,
    769 : 0,
    790 : -1,
    811 : 0,
    812 : 0,
    813 : 1,
    814 : 1,
    815 : 1,
    819 : 1,
    820 : 2,
    830 : 1,
    841 : 1,
    842 : 1,
    843 : 1,
    844 : 1,
    849 : 1,
    921 : -2,
    922 : -2,
    923 : -2,
    929 : -2,
    931 : -2,
    934 : -2,
    939 : -2,
    941 : -2,
    945 : -2,
    946 : -2,
    947 : -2,
    948 : -2,
    949 : -2,
    951 : -2,
    952 : -2,
    953 : -2,
    959 : -2,
    961 : -2,
    962 : -2,
    969 : -2,
    970 : -2,
    980 : -2,
    991 : -2,
    992 : -2,
    997 : 1,
    998 : 0,
    999 : 0
}

# convert to activity names
GTOU_label_to_activity = {
    0 : 'only main activity',
    110 : 'sleep',
    120 : 'eat',
    131 : 'wash self',
    132 : 'sleep',
    139 : 'personal',
    210 : 'job',
    220 : 'job',
    230 : 'job',
    241 : 'job',
    242 : 'job',
    243 : 'job',
    244 : 'job',
    245 : 'job',
    249 : 'job',
    311 : 'school',
    312 : 'school',
    313 : 'school',
    314 : 'school',
    315 : 'school',
    317 : 'school',
    319 : 'school',
    321 : 'school',
    329 : 'school',
    330 : 'school',
    341 : 'school',
    349 : 'school',
    353 : 'school homework',
    354 : 'school homework',
    361 : 'school',
    362 : 'school',
    363 : 'school',
    364 : 'school',
    369 : 'school',
    411 : 'cook',
    413 : 'cook',
    412 : 'cook',
    414 : 'cook',
    419 : 'cook',
    421 : 'cleaning',
    422 : 'cleaning',
    423 : 'cleaning',
    429 : 'cleaning',
    431 : 'laundry',
    432 : 'laundry',
    433 : 'laundry',
    434 : 'laundry',
    439 : 'laundry',
    441 : 'house work',
    442 : 'house work',
    443 : 'house work',
    444 : 'house work',
    445 : 'house work',
    446 : 'house work',
    449 : 'house work',
    451 : 'house work',
    452 : 'house work',
    453 : 'house work',
    454 : 'house work',
    455 : 'house work',
    459 : 'house work',
    461 : 'shopping',
    464 : 'shopping',
    465 : 'shopping',
    466 : 'shopping',
    469 : 'shopping',
    471 : 'family care',
    472 : 'family care',
    473 : 'family care',
    474 : 'family care',
    475 : 'family care',
    476 : 'family care',
    479 : 'family care',
    480 : 'family care',
    491 : 'family care',
    492 : 'family care',
    499 : 'family care',
    510 : 'socio-political',
    520 : 'socio-political',
    531 : 'socio-political',
    532 : 'socio-political',
    539 : 'socio-political',
    611 : 'leisure',
    612 : 'telphone',
    621 : 'leisure',
    622 : 'leisure',
    623 : 'leisure',
    624 : 'leisure',
    625 : 'leisure',
    626 : 'leisure',
    627 : 'leisure',
    629 : 'leisure',
    630 : 'leisure',
    641 : 'leisure',
    642 : 'leisure',
    649 : 'leisure',
    711 : 'leisure',
    712 : 'leisure',
    713 : 'leisure',
    715 : 'leisure',
    716 : 'leisure',
    717 : 'leisure',
    719 : 'leisure',
    730 : 'leisure',
    740 : 'leisure',
    752 : 'leisure',
    759 : 'leisure',
    761 : 'leisure',
    762 : 'leisure',
    763 : 'computer',
    769 : 'leisure',
    790 : 'leisure',
    811 : 'leisure',
    812 : 'leisure',
    813 : 'leisure',
    814 : 'leisure',
    815 : 'leisure',
    819 : 'leisure',
    820 : 'TV',
    830 : 'music',
    841 : 'computer/smartphone',
    842 : 'computer/smartphone',
    843 : 'computer/smartphone',
    844 : 'computer/smartphone',
    849 : 'computer/smartphone',
    921 : 'transportation',
    922 : 'transportation',
    923 : 'transportation',
    929 : 'transportation',
    931 : 'transportation',
    934 : 'transportation',
    939 : 'transportation',
    941 : 'transportation',
    945 : 'transportation',
    946 : 'transportation',
    947 : 'transportation',
    948 : 'transportation',
    949 : 'transportation',
    951 : 'transportation',
    952 : 'transportation',
    953 : 'transportation',
    959 : 'transportation',
    961 : 'transportation',
    962 : 'transportation',
    969 : 'transportation',
    970 : 'transportation',
    980 : 'transportation',
    991 : 'transportation',
    992 : 'transportation',
    997 : 'transportation',
    998 : 'transportation',
    999 : 'transportation'
}

# convert to activity names to the CREST consuming activities
GTOU_label_to_CREST_act = {
    0 : '-',
    110 : '-',
    120 : '-',
    131 : 'Act_WashDress',
    132 : '-',
    139 : '-',
    210 : '-',
    220 : '-',
    230 : '-',
    241 : '-',
    242 : '-',
    243 : '-',
    244 : '-',
    245 : '-',
    249 : '-',
    311 : '-',
    312 : '-',
    313 : '-',
    314 : '-',
    315 : '-',
    317 : '-',
    319 : '-',
    321 : '-',
    329 : '-',
    330 : '-',
    341 : '-',
    349 : '-',
    353 : '-',
    354 : '-',
    361 : '-',
    362 : '-',
    363 : '-',
    364 : '-',
    369 : '-',
    411 : 'Act_Cooking',
    413 : 'Act_Cooking',
    412 : 'Act_Cooking',
    414 : 'Act_Cooking',
    419 : 'Act_Cooking',
    421 : 'Act_HouseClean',
    422 : 'Act_HouseClean',
    423 : '-',
    429 : '-',
    431 : 'Act_Laundry',
    432 : 'Act_Iron',
    433 : '-',
    434 : '-',
    439 : '-',
    441 : '-',
    442 : '-',
    443 : '-',
    444 : '-',
    445 : '-',
    446 : '-',
    449 : '-',
    451 : '-',
    452 : '-',
    453 : '-',
    454 : '-',
    455 : '-',
    459 : '-',
    461 : 'shopping',
    464 : 'shopping',
    465 : 'shopping',
    466 : 'shopping',
    469 : 'shopping',
    471 : 'family care',
    472 : 'family care',
    473 : 'family care',
    474 : 'family care',
    475 : 'family care',
    476 : 'family care',
    479 : 'family care',
    480 : 'family care',
    491 : 'family care',
    492 : 'family care',
    499 : 'family care',
    510 : 'socio-political',
    520 : 'socio-political',
    531 : 'socio-political',
    532 : 'socio-political',
    539 : 'socio-political',
    611 : 'leisure',
    612 : 'telphone',
    621 : 'leisure',
    622 : 'leisure',
    623 : 'leisure',
    624 : 'leisure',
    625 : 'leisure',
    626 : 'leisure',
    627 : 'leisure',
    629 : 'leisure',
    630 : 'leisure',
    641 : 'leisure',
    642 : 'leisure',
    649 : 'leisure',
    711 : 'leisure',
    712 : 'leisure',
    713 : 'leisure',
    715 : 'leisure',
    716 : 'leisure',
    717 : 'leisure',
    719 : 'leisure',
    730 : 'leisure',
    740 : 'leisure',
    752 : 'leisure',
    759 : 'leisure',
    761 : 'leisure',
    762 : 'leisure',
    763 : 'computer',
    769 : 'leisure',
    790 : 'leisure',
    811 : 'leisure',
    812 : 'leisure',
    813 : 'leisure',
    814 : 'leisure',
    815 : 'leisure',
    819 : 'leisure',
    820 : 'Act_TV',
    830 : 'music',
    841 : 'computer/smartphone',
    842 : 'computer/smartphone',
    843 : 'computer/smartphone',
    844 : 'computer/smartphone',
    849 : 'computer/smartphone',
    921 : 'transportation',
    922 : 'transportation',
    923 : 'transportation',
    929 : 'transportation',
    931 : 'transportation',
    934 : 'transportation',
    939 : 'transportation',
    941 : 'transportation',
    945 : 'transportation',
    946 : 'transportation',
    947 : 'transportation',
    948 : 'transportation',
    949 : 'transportation',
    951 : 'transportation',
    952 : 'transportation',
    953 : 'transportation',
    959 : 'transportation',
    961 : 'transportation',
    962 : 'transportation',
    969 : 'transportation',
    970 : 'transportation',
    980 : 'transportation',
    991 : 'transportation',
    992 : 'transportation',
    997 : 'transportation',
    998 : 'transportation',
    999 : 'transportation'
}

GTOU_label_to_CREST_act_v2 = {
    0 : '-',
    110 : '-',
    120 : '-',
    131 : 'Act_WashDress',
    132 : '-',
    139 : '-',
    210 : '-',
    220 : '-',
    230 : '-',
    241 : '-',
    242 : '-',
    243 : '-',
    244 : '-',
    245 : '-',
    249 : '-',
    311 : '-',
    312 : '-',
    313 : '-',
    314 : '-',
    315 : '-',
    317 : '-',
    319 : '-',
    321 : '-',
    329 : '-',
    330 : '-',
    341 : '-',
    349 : '-',
    353 : '-',
    354 : '-',
    361 : '-',
    362 : '-',
    363 : '-',
    364 : '-',
    369 : '-',
    411 : 'Act_Cooking',
    413 : 'Act_Dishwashing',
    412 : 'Act_Cooking',
    414 : 'Act_Cooking',
    419 : 'Act_Cooking',
    421 : 'Act_HouseClean',
    422 : 'Act_HouseClean',
    423 : '-',
    429 : '-',
    431 : 'Act_Laundry',
    432 : 'Act_Iron',
    433 : '-',
    434 : '-',
    439 : '-',
    441 : '-',
    442 : '-',
    443 : '-',
    444 : '-',
    445 : '-',
    446 : '-',
    449 : '-',
    451 : '-',
    452 : '-',
    453 : '-',
    454 : '-',
    455 : '-',
    459 : '-',
    461 : 'shopping',
    464 : 'shopping',
    465 : 'shopping',
    466 : 'shopping',
    469 : 'shopping',
    471 : 'family care',
    472 : 'family care',
    473 : 'family care',
    474 : 'family care',
    475 : 'family care',
    476 : 'family care',
    479 : 'family care',
    480 : 'family care',
    491 : 'family care',
    492 : 'family care',
    499 : 'family care',
    510 : 'socio-political',
    520 : 'socio-political',
    531 : 'socio-political',
    532 : 'socio-political',
    539 : 'socio-political',
    611 : 'leisure',
    612 : 'telphone',
    621 : 'leisure',
    622 : 'leisure',
    623 : 'leisure',
    624 : 'leisure',
    625 : 'leisure',
    626 : 'leisure',
    627 : 'leisure',
    629 : 'leisure',
    630 : 'leisure',
    641 : 'leisure',
    642 : 'leisure',
    649 : 'leisure',
    711 : 'leisure',
    712 : 'leisure',
    713 : 'leisure',
    715 : 'leisure',
    716 : 'leisure',
    717 : 'leisure',
    719 : 'leisure',
    730 : 'leisure',
    740 : 'leisure',
    752 : 'leisure',
    759 : 'leisure',
    761 : 'leisure',
    762 : 'leisure',
    763 : 'computer',
    769 : 'leisure',
    790 : 'leisure',
    811 : 'leisure',
    812 : 'leisure',
    813 : 'leisure',
    814 : 'leisure',
    815 : 'leisure',
    819 : 'leisure',
    820 : 'Act_TV',
    830 : 'music',
    841 : 'Act_Elec',
    842 : 'Act_Elec',
    843 : 'Act_Elec',
    844 : 'Act_Elec',
    849 : 'Act_Elec',
    921 : '-',
    922 : '-',
    923 : '-',
    929 : '-',
    931 : '-',
    934 : '-',
    939 : '-',
    941 : '-',
    945 : '-',
    946 : '-',
    947 : '-',
    948 : '-',
    949 : '-',
    951 : '-',
    952 : '-',
    953 : '-',
    959 : '-',
    961 : '-',
    962 : '-',
    969 : '-',
    970 : '-',
    980 : '-',
    991 : '-',
    992 : '-',
    997 : '-',
    998 : '-',
    999 : '-'
}
#raw_states = secondary_states
# convert the raw states to integer states
def convert_states(raw_states, merge_dic = None):
    """
    convert the states following the rule inscibed in the merging dictionary.
    Very useful to merge different states into a single one.

    Args:
        raw_states (np.array): an array containing the possible states
        merge_dic (dict): dictionary that matches each state to a new state

    Returns:
        merged states: the new states
        u_lab: the label of those new states
    """
    u,  inv = np.unique(raw_states,  return_inverse=True)
    # get the integer states
    states = inv.reshape(raw_states.shape)

    # merge the activities
    if merge_dic is not None:
        assert isinstance(merge_dic, dict) | isinstance(merge_dic, np.ndarray), 'merge_dic must be a dictionary'
        to_labels = np.vectorize(lambda x: merge_dic[x])
    else:
        to_labels = np.vectorize(lambda x: x)
    states_labels = to_labels(u)
    # merge the labels together
    u_lab, inv_lab = np.unique(states_labels, return_inverse=True)
    # get the states once merged by category
    merged_states = inv_lab[states]
    return merged_states, u_lab

#  convert the states for a 4 states model
def states_to_activity(states):

    dic = {
        110 : 0,
        120 : 1,
        131 : 1,
        132 : 0,
        139 : 1,
        210 : 1,
        220 : 1,
        230 : 1,
        241 : 1,
        242 : 1,
        243 : 1,
        244 : 1,
        245 : 1,
        249 : 1,
        311 : 1,
        312 : 1,
        313 : 1,
        314 : 1,
        315 : 1,
        317 : 1,
        319 : 1,
        321 : 1,
        329 : 1,
        330 : 1,
        341 : 1,
        349 : 1,
        353 : 1,
        354 : 1,
        361 : 1,
        362 : 1,
        363 : 1,
        364 : 1,
        369 : 1,
        411 : 1,
        412 : 1,
        413 : 1,
        414 : 1,
        419 : 1,
        421 : 1,
        422 : 1,
        423 : 1,
        429 : 1,
        431 : 1,
        432 : 1,
        433 : 1,
        434 : 1,
        439 : 1,
        441 : 1,
        442 : 1,
        443 : 1,
        444 : 1,
        445 : 1,
        446 : 1,
        449 : 1,
        451 : 1,
        452 : 1,
        453 : 1,
        454 : 1,
        455 : 1,
        459 : 1,
        461 : 1,
        464 : 1,
        465 : 1,
        466 : 1,
        469 : 1,
        471 : 1,
        472 : 1,
        473 : 1,
        474 : 1,
        475 : 1,
        476 : 1,
        479 : 1,
        480 : 1,
        491 : 1,
        492 : 1,
        499 : 1,
        510 : 1,
        520 : 1,
        531 : 1,
        532 : 1,
        539 : 1,
        611 : 1,
        612 : 1,
        621 : 1,
        622 : 1,
        623 : 1,
        624 : 1,
        625 : 1,
        626 : 1,
        627 : 1,
        629 : 1,
        630 : 1,
        641 : 1,
        642 : 1,
        649 : 1,
        711 : 1,
        712 : 1,
        713 : 1,
        715 : 1,
        716 : 1,
        717 : 1,
        719 : 1,
        730 : 1,
        740 : 1,
        752 : 1,
        759 : 1,
        761 : 1,
        762 : 1,
        763 : 1,
        769 : 1,
        790 : 1,
        811 : 1,
        812 : 1,
        813 : 1,
        814 : 1,
        815 : 1,
        819 : 1,
        820 : 1,
        830 : 1,
        841 : 1,
        842 : 1,
        843 : 1,
        844 : 1,
        849 : 1,
        921 : 1,
        922 : 1,
        923 : 1,
        929 : 1,
        931 : 1,
        934 : 1,
        939 : 1,
        941 : 1,
        945 : 1,
        946 : 1,
        947 : 1,
        948 : 1,
        949 : 1,
        951 : 1,
        952 : 1,
        953 : 1,
        959 : 1,
        961 : 1,
        962 : 1,
        969 : 1,
        970 : 1,
        980 : 1,
        991 : 1,
        992 : 1,
        997 : 1,
        998 : 1,
        999 : 1
    }
    s,l = convert_states(states, dic)
    return l[s]

def is_transportation(states):
    return (states >=900) & (states!= 997) & (states!= 999) & (states!= 998)
    #return states >=900

def is_for_work_or_school(states):
    return ((states>=200) & (states<400)) | ((states>=900) &(states<=940))


def states_to_occupancy(primary_states, secondary_states, initial_occupancy, final_occupancy, dic_home):

    # convert the states to their home indice rating
    main_rating, _         = convert_states(primary_states, dic_home)
    secondary_rating, _    = convert_states(secondary_states, dic_home)
    home_ratings = main_rating + secondary_rating

    #initialize the occupancy
    current_occupancy= np.zeros_like(initial_occupancy, dtype=bool)
    # assign occupants at home
    current_occupancy[initial_occupancy == 1] = True
    # missing must check the current states
    current_occupancy[initial_occupancy == -1] = home_ratings[:, 0][initial_occupancy == -1] >= 0  # favor being home with the =0, as we start at 4:00
    # current_occupancy[initial_occupancy == 2] = False # from inital vector

    # find out where the last travel occurs so that the state can easily be determined
    # (take the last index where ther was a transportation (>=900) or -42(unreachable later) if there was no travel)
    last_travel_indexes = np.array([-42 if len(inds := np.where(is_transportation(ps) | is_transportation(ss))[0]) == 0 else inds[-1] for ps, ss in zip(primary_states, secondary_states)], dtype=int)


    occupancies = []
    mask_was_travelling = np.zeros_like(current_occupancy)
    mask_at_home_before_last_travel = np.zeros_like(current_occupancy)

    for i, (prim_state, sec_state, home_rating) in enumerate(zip(primary_states.T, secondary_states.T, home_ratings.T)):



        # only update the new states after a travel has been done
        mask_travelling = is_transportation(prim_state) | is_transportation(sec_state) # travelling is occurring




        # people who leave are not there anymore
        mask_leave = ~mask_was_travelling & mask_travelling
        mask_at_home_before_last_travel[mask_leave] = current_occupancy[mask_leave] # save state of before last travel
        current_occupancy[mask_leave] = False

        # people who finish a travel can be at home or not
        mask_finish_travel = mask_was_travelling & ~mask_travelling
        # check if activity can be perform at home,  also take into account the previous occupancy and imagine it should change
        previous_occ_bias = -2 * mask_at_home_before_last_travel[mask_finish_travel] + 1 # true->-1, false->1
        current_occupancy[mask_finish_travel] = (home_rating[mask_finish_travel] + previous_occ_bias)>= 0
        # if it is the last travel, set to the last occupancy
        mask_after_last_travel = last_travel_indexes == i-1 # check if the end of the travel
        current_occupancy[mask_after_last_travel & (final_occupancy == 1)] = True
        current_occupancy[mask_after_last_travel & (final_occupancy == 2)] = False
        # keine angabe gets the last value probability
        current_occupancy[mask_after_last_travel & (final_occupancy == -1)] =  home_ratings.T[-1][mask_after_last_travel & (final_occupancy == -1)]

        #
        mask_travelling = np.array(mask_was_travelling)
        occupancies.append(np.array(current_occupancy))

    return np.asarray(occupancies).T

def states_to_out_for_what_model(occupancy, primary_states, secondary_states):
    out = np.array(primary_states)
    mask_HWH = is_for_work_or_school(primary_states) | is_for_work_or_school(secondary_states)
    out[occupancy] = 0
    out[(~occupancy) & mask_HWH] = 1
    out[(~occupancy) & (~mask_HWH)] = 2
    return out, np.array(['In-house', 'HWH', 'HOH'])




def group_in_household_4states(states, household_indexes=np.array(df_akt['id_hhx']), days_indexes=np.array(df_akt['tagnr'])):
    """Group the states given as single resident observation to a household occupnacy. Merges the occupants
    along the given days and households as optional arguments. (will use German ATUS as default)

    Args:
        states (ndarray): the states of the single residents
        household_indexes (ndarray, optional): the indexes of the housholds. Defaults to np.array(df_akt['id_hhx']).
        days_indexes (ndarray, optional): the indexes of the days. Defaults to np.array(df_akt['tagnr']).

    Returns:
        merged_states: the household states after having merged residents together
    """
    assert (states.shape[0] == len(household_indexes)) & (states.shape[0] == len(days_indexes)), 'length must correspond to the number of diaries'
    assert np.all(np.logical_or.reduce((
        states == 0, states == 1, states == 10, states == 11)
    )), 'states given as input must correspond to the 4 state model with single resident states'
    merged_states = []
    for day_nr, household_id in zip(*np.unique([days_indexes, household_indexes], axis=1)):
        merged_states.append(np.sum(states[(household_indexes==household_id) & (days_indexes==day_nr)], axis=0))
    return np.asarray(merged_states)


def group_in_household_activity(mask_activity, household_indexes=np.array(df_akt['id_hhx']), days_indexes=np.array(df_akt['tagnr'])):
    """
    Group the activity mask given as single resident survey, to the number of occupant of the household.
    Doing this activity at that moment. Merges the occupants
    along the given days and households as optional arguments. (will use German ATUS as default)

    Args:
        mask_activity (ndarray): a mask of the states array that is true when an occupant is doing the activity at a certain time
        household_indexes (ndarray, optional): the indexes of the housholds. Defaults to np.array(df_akt['id_hhx']).
        days_indexes (ndarray, optional): the indexes of the days. Defaults to np.array(df_akt['tagnr']).

    Returns:
        merged_activities: the number of occupant doing activity at a given time after having merged residents together
    """
    assert (mask_activity.shape[0] == len(household_indexes)) & (mask_activity.shape[0] == len(days_indexes)), 'length must correspond to the number of diaries'

    merged_states = []
    for day_nr, household_id in zip(*np.unique([days_indexes, household_indexes], axis=1)):
        merged_states.append(np.sum(mask_activity[(household_indexes==household_id) & (days_indexes==day_nr)], axis=0))
    return np.asarray(merged_states)


# a plot function for stacked states
def plot_stack_states(states, labels):
    """plot the stack states

    Args:
        states (ndarray): the states to be stacked in the plot
        labels (iterable): the labels of the different states values in the states array

    Returns:
        fig, ax: a matplotlib figure and ax objects
    """
    # stack plot with an hourhly axis
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1, 1, 1)
    # stack the states for the stack plot
    stacked_states = np.apply_along_axis(np.bincount, 0, np.array(states, int), minlength=np.max(states)+1)
    ax.stackplot(np.arange(4,28,24/states.shape[1]), stacked_states, labels=labels)
    handles, lab = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lab[::-1], loc='right')
    return fig, ax

# %% see the result, analyse subgroups

subgroup_kwargs = {
    'only_fully_recorded_household':    False,
    'remove_missing_days':              False,
    'quarter':                          None,
    'weekday':                          0,
    'n_residents':                      0,
    'household_type':                   0,
    'is_travelling':                    None,
    'life_situation':                   0,
    'age':                              0,
    'gender':                           0,
    'household_position':               0,
    'geburtsland':                      0
}



merged_states, merged_labels = convert_states(
    primary_states, GTOU_label_to_activity)


plot_stack_states(merged_states[get_mask_subgroup(**subgroup_kwargs)], merged_labels)


# %% do a 4 states occupancy model
occ = states_to_occupancy(primary_states, secondary_states, initial_location, final_location, dic_home )
act = states_to_activity(primary_states)

# merged_states, merged_labels = convert_states(occ)
# plot_stack_states(merged_states, merged_labels)

merged_states, merged_labels = convert_states(10*occ + act)
#plot_stack_states(merged_states, merged_labels)
subgroup_kwargs = {
    'only_fully_recorded_household':    True,
    'remove_missing_days':              True,
    'only_household_identical_days':    True,
    'quarter':                          0,
    #'weekday':                          0,
    #'weekday':                          [1,2,3,4,5],
    'weekday':                          [6,7],
    'n_residents':                      0,
    'household_type':                   0,
    'is_travelling':                    None,
    'life_situation':                   0,
    'age':                              0,
    'gender':                           0,
    'household_position':               0,
    'geburtsland':                      0
}


mask_subgroup = get_mask_subgroup( **subgroup_kwargs)
# group the houshold in the states from the 4 states occupancy model
household_states = group_in_household_4states(
    np.array(merged_labels[merged_states])[mask_subgroup],
    household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
    days_indexes=np.array(df_akt['tagnr'])[mask_subgroup] )

plot_stack_states(*convert_states(household_states))

states_out_for_what , labels_out_for_what = states_to_out_for_what_model(
    occ[mask_subgroup],
    primary_states[mask_subgroup],
    secondary_states[mask_subgroup])
plot_stack_states(*convert_states(
    states_out_for_what , labels_out_for_what
))
transitions_out_for_what = group_hh_transitions(
    states_out_for_what,
    household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
    days_indexes=np.array(df_akt['tagnr'])[mask_subgroup])

# gets the number of observations and how well they have been merged
u,c = np.unique([np.array(df_akt['id_hhx'])[mask_subgroup],np.array(df_akt['tagnr'])[mask_subgroup]], axis=1, return_counts=True)
print('count of number of people diaries in each houshold diary', count(c))


transitions = group_hh_transitions(
    np.array(merged_labels[merged_states])[mask_subgroup],
    household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
    days_indexes=np.array(df_akt['tagnr'])[mask_subgroup]
)
count(transitions['counts'])
# %% generate the transition matrix

def states_to_tpm(states, first_matrix_strategy='last', undefined_states_value=1.):
    # this matrix will store the transitions probabilites
    tpms = []
    # initialize the first state with the previous state
    old_states = np.array(states[-1])
    n_states = int(np.max(states) + 1)  # get the number of states from the input
    for this_states in states:

        # define and counts the transitions
        states_indices, states_counts = np.unique(np.asarray((old_states, this_states)), axis=1, return_counts=True)
        states_indices = [(i) for i in states_indices] # converts the indexes for accessing the matrix later

        # compute the sum of the transitions for each states
        transition_matrice = np.full((n_states, n_states), 0)
        transition_matrice[states_indices] = states_counts
        tpms.append(transition_matrice)

        # save the state for old state
        old_states = np.array(this_states)

    # define what we should do with the first matrix that has false transitions
    if first_matrix_strategy == 'last':
        tpms[0] = np.array(tpms[-1])
    elif first_matrix_strategy == 'nothing':
        pass
    else:
        raise TypeError('Unknown first matrix stragtegy kwarg')

    # converts to probs
    tpms = tpms / np.sum(tpms, axis=2)[:,:,None]
    tpms[np.isnan(tpms)]=0. # set to the same state when there are nan values
    # for the cdfs that have no values, we set unchanging states
    times, rows = np.where(tpms.sum(axis=2)==0)
    tpms[times, rows, rows] = undefined_states_value # make the stay at the same states

    return np.asarray(tpms)

states, states_label = convert_states(household_states)
tpm = states_to_tpm(states.T, undefined_states_value=0.)


def prints_pdfs_absorbing_states(pdfs):
    """Check if the pdf has some states that are absorbing
    Prints out which states are absorbing and the time that they are

    Args:
        pdfs (ndarray): a ndarray of shape (times, states, states)
    """
    # check where the pdfs are not 0
    _, old_states = np.where(pdfs[-1])
    old_states = np.unique(old_states)
    current_absorbing_states = np.array([])
    current_absorbing_times = np.array([])

    returns = []
    i=0
    for i, pdf in enumerate(pdfs):
        # check the available transitions
        this_states, next_states = np.where(pdf)
        this_states = np.unique(this_states)
        next_states = np.unique(next_states)

        # check if an absorbing state stops absorbing
        mask_stop_absorbing = np.isin(current_absorbing_states, this_states)
        if np.any(mask_stop_absorbing):
            print('absorbing states and n_steps ', current_absorbing_states[mask_stop_absorbing], current_absorbing_times[mask_stop_absorbing])

            current_absorbing_states = current_absorbing_states[~mask_stop_absorbing]
            current_absorbing_times = current_absorbing_times[~mask_stop_absorbing]

        # check if they are absorbing states
        if len(this_states) == len(old_states) and np.all(np.unique(old_states) == np.unique(this_states)):
            pass
        else: # the states have a jump issue
            print("Problem with state", i)
            print(np.unique(old_states), np.unique(this_states))

            # gets the elements that start beeing in an absorbing state (they are available from the old state but they have no pdf in this states)
            absorbing_states = old_states[np.isin(old_states, this_states, invert=True)]
            # adds the absorbing states
            returns.append((i, absorbing_states))
            current_absorbing_states = np.concatenate((current_absorbing_states, absorbing_states))
            current_absorbing_times = np.concatenate((current_absorbing_times, np.zeros_like(absorbing_states)))

        old_states = next_states
        current_absorbing_times += 1
    print('ended, absorbing states and n_steps ', current_absorbing_states, current_absorbing_times)
    returns.append((i, current_absorbing_states))
    return returns

prints_pdfs_absorbing_states(tpm)

# used to save the tpms like crest
#np.save('compiled_data'+os.sep+'4states_tpms'+os.sep+'tpm_wd_'+str(subgroup_kwargs['n_residents'])+'res', tpm)
## used to save the starting states and the labels
#start_counts = np.bincount(states[:,0], minlength=np.max(states)+1)
#np.save('compiled_data'+os.sep+'4states_tpms'+os.sep+'startstates_wd_'+str(subgroup_kwargs['n_residents'])+'res', start_counts/np.sum(start_counts))
#np.save('compiled_data'+os.sep+'4states_tpms'+os.sep+'labels_wd_'+str(subgroup_kwargs['n_residents'])+'res', states_label)
#assert tpm.shape[-1] == len(start_counts), 'shape mismatch'
# %% generate the activity profiles

mask_subgroup = get_mask_subgroup( **subgroup_kwargs)

main_activity_states, main_activity_labels = convert_states(
    primary_states[mask_subgroup], GTOU_label_to_CREST_act_v2)
sec_activity_states, sec_activity_labels = convert_states(
    secondary_states[mask_subgroup], GTOU_label_to_CREST_act_v2)

for desired_act in main_activity_labels:

    if desired_act not in ['Act_TV','Act_Cooking','Act_Laundry','Act_WashDress','Act_Iron','Act_HouseClean', 'Act_Dishwashing', 'Act_Elec']:
        continue
    main_act_ind = np.where(main_activity_labels==desired_act)[0]
    sec_act_ind  = np.where(sec_activity_labels==desired_act)[0]
    # find in each household how many people are doing the activity desired
    n_occ_performing_act = group_in_household_activity(
        (main_activity_states == main_act_ind) | (sec_activity_states == sec_act_ind),
        household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
        days_indexes=np.array(df_akt['tagnr'])[mask_subgroup]
        )
    # get the active occupancy of the housholds at that moment
    active_occ = np.minimum(household_states%10, household_states//10)
    for n_occ in range(1,6):
        # the probability that at least n_occ is doing an activity is the probability that
        # a houshold is doing this activity divided by the number of household with that number of
        # active occupant
        n_hh_with_n_act_occ = np.sum(active_occ == n_occ, axis=0)

        # probability that at least n_occ are doing the activity
        # prob_n_occ_performing_act = np.mean(n_occ_performing_act>=n_occ, axis=0)

        prob_at_least_one_occ_perfoming_act = np.sum(
            (n_occ_performing_act>0) & (active_occ == n_occ), axis=0 ) / n_hh_with_n_act_occ
        # set to zero the nan values, as there was no active occupant
        prob_at_least_one_occ_perfoming_act[np.isnan(prob_at_least_one_occ_perfoming_act)] = 0.0
        #used to save the crest model like
        #np.save('compiled_data'+os.sep+'activity_profiles'+os.sep+'like_crest_we_'+desired_act+'_nactocc_'+str(n_occ), prob_at_least_one_occ_perfoming_act)
        plt.plot(prob_at_least_one_occ_perfoming_act, label=str(n_occ))
        # print the average activity probability, for the appliance usage
        if n_occ == 1:
            print(np.mean(prob_at_least_one_occ_perfoming_act))
    plt.title(desired_act)
    plt.ylabel('prob for dwelling that at least one occ is perfoming activity')
    plt.legend()
    plt.show()

# %% simulate
n_households_simulated = 1000
sim = OccupancySimulator(n_households_simulated, tpm.shape[-1], tpm)
initial_pdf = np.full(tpm.shape[-1],0)
initial_pdf[0] = 1
sim.initialize_starting_state(initial_pdf, checkcdf=False)
#%%
for i in range(7*24*6): # let one week as an initialisation
    sim.step()


states_list = np.array([], dtype=int)
for i in range(24*6):
    sim.step()
    states_list = np.append(states_list, np.array(sim.current_states))

states_list.reshape(-1, n_households_simulated)



plot_stack_states(*convert_states(states_label[states_list.T]))



# %%
def graph_metrics(states):
    """compute and return 4 graph metrics for a given array of states


    Args:
        states (ndarray(times, states)): An array containing the states for which we want to compute the graph metrics

    Returns:
        tuple(int, float, float, float): A tuple containing network_size, network_density, centrality, homophily.

    Notes:
        These metrics are an attempt to implement graph metrics as proposed by McKenna et al (2020),
        https://doi.org/10.1016/j.erss.2020.101572
    """
    # gets the max states size to store the edges
    max_size = np.max(states) + 1
    # initialize the first edges with the last matrix
    old_s = np.array(states[-1])
    directed_edges = []

    for s in states:
        indices, counts = np.unique(np.c_[old_s, s], axis=0, return_counts=True)
        indices = tuple(indices.T)
        edges = np.zeros((max_size, max_size))
        edges[indices] = counts
        directed_edges.append(edges)

        old_s = s

    times, source_nodes, end_nodes = np.where(directed_edges)

    source_nodes_id = np.char.add(np.char.add(np.array(times, dtype=str), ['_' for i in range(len(times))]), np.array(source_nodes, dtype=str))
    end_nodes_id = np.char.add(np.char.add(np.array(times, dtype=str), ['_' for i in range(len(times))]), np.array(end_nodes, dtype=str))
    edge_weights = np.array(directed_edges)[times, source_nodes, end_nodes]

    network_size = len(np.unique(np.c_[source_nodes_id, end_nodes_id]))
    network_density = len(source_nodes_id) / ((len(directed_edges)-1) * len(np.unique(source_nodes))**2) # the max number of nodes possible
    centrality = np.mean(edge_weights)
    homophily = np.sum((end_nodes == source_nodes) * edge_weights)/ np.sum(edge_weights)

    return network_size, network_density, centrality, homophily

subgroup_size = 100


(
    graph_metrics(states_list[:, np.random.choice(states_list.shape[1], subgroup_size)]),
    graph_metrics(household_states[np.random.choice(states_list.shape[0], subgroup_size)].T)
    )

# %%
def build_timed_based_transition_matrices(
    initial_states,
    transitions_times, transitions_persons, transitions_new_states,
    n_times=None, n_persons=None, n_states=None,
    first_matrix_strategy='last'):

    # deduces the values of the inputs, or check legitimity of  the values if they were given
    if n_times is None:
        n_times = max(transitions_times) + 1
    else:
        assert n_times > max(transitions_times), 'n_times is smaller than the ones given in transitions times'
    if n_persons is None:
        n_persons = max(transitions_persons) + 1
    else:
        assert n_persons > max(transitions_persons), 'n_persons is smaller than the ones given in transitions persons'
    if n_states is None:
        n_states = max(transitions_new_states) + 1
    else:
        assert n_states > max(transitions_new_states), 'n_states is smaller than the ones given in transitions states'
        assert n_states > max(initial_states), 'n_states is smaller than the ones given in initial states'


    assert len(initial_states) == n_persons, ' The length of the initial states should match the number of persons'
    this_states = np.array(initial_states)

    assert len(transitions_times) == len(transitions_persons), 'length of transitions arrays must be the same'
    assert len(transitions_times) == len(transitions_new_states), 'length of transitions arrays must be the same'



    transitions_times = np.array(transitions_times, dtype=int)
    transitions_persons = np.array(transitions_persons, dtype=int)
    transitions_new_states = np.array(transitions_new_states, dtype=int)

    # checks that the initial transitions are the initial states
    assert np.all(transitions_new_states[transitions_times==0] ==  \
        this_states[transitions_persons[transitions_times == 0]]), 'initial states must correspond to the transitions_times'

    # the array must be sorted, having first the persons in the right order and then the times
    # sorted in persons
    persons_to_be_compared_to = np.roll(transitions_persons, -1)
    persons_to_be_compared_to[-1] = transitions_persons[-1]
    assert np.all(transitions_persons <= persons_to_be_compared_to), 'the input person array must be sorted'
    # sub-sorted in times
    times_to_be_compared_to = np.roll(transitions_times, -1)
    # when there is a change in the persons don't check
    mask_last_transition_of_persons = transitions_persons != persons_to_be_compared_to
    times_to_be_compared_to[mask_last_transition_of_persons] = transitions_times[mask_last_transition_of_persons] + 1
    times_to_be_compared_to[-1] = transitions_times[-1] + 1

    assert np.all(transitions_times < times_to_be_compared_to), 'the times must be subsorted for calculation the durations'


    # gets the first state of each persons
    _, first_times_indices, persons_counts = np.unique(transitions_persons, return_index=True, return_counts=True)
    first_states = transitions_new_states[first_times_indices]

    # gets the persons with no transitions
    mask_no_transition_persons = persons_counts == 1

    # gets the last state of each persons
    last_times_indices = np.roll(first_times_indices - 1, -1)
    last_times_indices[mask_no_transition_persons] = first_times_indices[mask_no_transition_persons] # if there is only one index, start = end
    last_states = transitions_new_states[last_times_indices]



    # gets the states that should be merged
    mask_merge_persons = first_states == last_states
    print(np.average(mask_merge_persons)*100,r' [\%] have been merged')

    # store the transitions that should be ignored
    mask_ignore_durations = np.zeros_like(transitions_times, dtype=bool)
    #ignore the last transitions if they must be merged
    mask_ignore_durations[last_times_indices] = np.invert(mask_merge_persons)


    # gets the old states of all the transitions
    transitions_old_states = np.roll(transitions_new_states, 1)
    transitions_old_states[first_times_indices] = last_states # first transitions has old states the last one


    # change the values of the durations aftern ignoring the fakes

    # initialize the arrays for the duration
    transitions_end_times = np.roll(transitions_times, -1)
    end_of_next_state_times = np.zeros_like(last_times_indices)
    # if there is a transition, the end of state is just the 2nd element of transitions for a person
    end_of_next_state_times[~mask_no_transition_persons] = transitions_times[first_times_indices[~mask_no_transition_persons]+1] # the +1 is for accessing the 2nd element (as the first one is the next day trnasition)
    transitions_end_times[last_times_indices]  = n_times + end_of_next_state_times # the end time for the last element is the start time of the second transition
    # gets the durations of the transitions
    transitions_durations = transitions_end_times - transitions_times



    # this matrix will store the transitions probabilites
    transition_matrices = np.zeros((n_times, n_states, n_states))

    # gets the indices for the transition matrix
    transitions_indices, transitions_counts = np.unique(np.asarray((
        transitions_times,
        transitions_old_states,
        transitions_new_states
    )), axis=1, return_counts=True)
    transitions_indices = [(i) for i in transitions_indices]
    # build the transition matrix
    transition_matrices[transitions_indices] = transitions_counts


    # this matrix will store the transitions probabilites
    durations_matrices = np.zeros((n_times, n_states, n_times+1))

    # gets the indices for the duration matrix
    durations_indices, durations_counts = np.unique(np.asarray((
        transitions_times[~mask_ignore_durations],
        transitions_new_states[~mask_ignore_durations],
        transitions_durations[~mask_ignore_durations]
    )), axis=1, return_counts=True)
    durations_indices = [(i) for i in durations_indices]
    print(durations_indices)
    # build the duration matrix
    durations_matrices[durations_indices] = durations_counts

    # define what we should do with the first matrix that has false transitions
    if first_matrix_strategy == 'last':
        transition_matrices[0] = transition_matrices[-1]
        durations_matrices[0] = durations_matrices[-1]
    elif first_matrix_strategy == 'nothing':
        pass
    else:
        raise TypeError('Unknown first matrix stragtegy kwarg')

    # converts to probs
    transition_matrices = transition_matrices / np.sum(transition_matrices, axis=2)[:,:,None]
    durations_matrices = durations_matrices / np.sum(durations_matrices, axis=2)[:,:,None]
    return transition_matrices, durations_matrices




#%%

##################
# DURATION BASED
##################


transition_matrices, duration_matrices = build_timed_based_transition_matrices(
    initial_states= activities_inverse[df['start index'] == 0],
    transitions_times=transition_times_10min[mask_indices_used],
    transitions_persons=persons_indices[mask_indices_used],
    transitions_new_states=activities_inverse[mask_indices_used])



sim = TimedStatesSimulator(n_households_simulated, 3, transition_matrices, duration_matrices)
sim.initialize_starting_state([0,1,0], checkcdf=False)
# %%


for i in range(7*24*6): # let one week as an initialisation
    sim.step()


states_list = []
for i in range(24*6):
    sim.step()
    states_list.append(np.array(sim.current_states))

states_list = np.asarray(states_list)
# %%
stacked_states = np.apply_along_axis(np.bincount, 1, states_list, minlength=3)

plt.stackplot(np.arange(len(states_list)), stacked_states.T)
plt.show()


# %% compare the durations
states_indice = 0

# real data
durations, corresponding_states = get_states_durations(np.asarray(true_data_states).T)
mask_state = np.where(corresponding_states == states_indice)[0]
pdf_true, labels = np.histogram(durations[mask_state],bins=144, range=(0,144), density=True)
pdf_true_half1, labels = np.histogram(durations[mask_state][::2],bins=144, range=(0,144), density=True)
pdf_true_half2, labels = np.histogram(durations[mask_state][1::2],bins=144, range=(0,144), density=True)


plt.plot(pdf_true, label='real')

# time based simulation
durations, corresponding_states = get_states_durations(np.asarray(states_list).T)
mask_state = np.where(corresponding_states == states_indice)[0]
pdf_timed, labels = np.histogram(durations[mask_state],bins=144, range=(0,144), density=True)
plt.plot(pdf_timed, label='time-based')

# markov 1rst order simulation
durations, corresponding_states = get_states_durations(np.asarray(current_states).T)
mask_state = np.where(corresponding_states == states_indice)[0]
pdf_markov, labels = np.histogram(durations[mask_state],bins=144, range=(0,144), density=True)
plt.plot(pdf_markov, label='markov 1rst order')



plt.legend()

print('Duration Error')
print('RMSE marvok', RMSE(pdf_markov, pdf_true))
print('RMSE timed',  RMSE(pdf_timed, pdf_true))
print('RMSE half/half data',  RMSE(pdf_true_half1, pdf_true_half2))
# %% aggregated values
def stack_states(states, n_states=None):
    if n_states == None:
        n_states = np.max(states)+1
    counts = np.apply_along_axis(np.bincount, 1, states, minlength=n_states)
    # returns probs
    return counts / np.sum(counts[0])

print('State transitions Error')
print('RMSE marvok', RMSE(stack_states(current_states), stack_states(true_data_states)))
print('RMSE timed', RMSE(stack_states(states_list), stack_states(true_data_states)))
print('RMSE half/half data',  RMSE(stack_states(true_data_states[:,0::2]), stack_states(true_data_states[:,1::2])))

# %% 24 hour occupancy values


print(
    '24 hours occupancies \n'
    'True ',
    np.average(CREST_get_24h_occupancy(true_data_states.T,state_labels=activities)),
    '\n Markov ',
    np.average(CREST_get_24h_occupancy(current_states.T,state_labels=activities)),
    '\n Timed ',
    np.average(CREST_get_24h_occupancy(states_list.T,state_labels=activities)))
# %% get again the full activities for some fun in clustering
activities, activities_indices, activities_inverse = np.unique(
    df['activity'], return_index=True, return_inverse=True)
# %% clustering the patterns from the data set

states = transitions_to_states(
    transitions_times=transition_times_10min[mask_indices_used],
    transitions_persons=persons_indices[mask_indices_used],
    transitions_new_states=activities_inverse[mask_indices_used]
)
# invert the states
#states[states==1] = 4
#states[states==2] = 1
#states[states==4] = 2

states

from sklearn.cluster import KMeans, AgglomerativeClustering



n = 2
cluster_id = KMeans(n_clusters=n,).fit_predict(states)
for i in range(n):
    plt.figure(figsize=(16,9))
    stacked_states = np.apply_along_axis(np.bincount, 0, states[cluster_id==i], minlength=np.max(states)+1)
    plt.stackplot(np.arange(states.shape[1])/6, np.roll(stacked_states, 24, axis=1), labels=activities)
    plt.title('number of patterns in cluster :' + str(np.sum(cluster_id==i)))

    plt.xlim(0,24)
    plt.legend(loc= 'upper left')
    plt.show()

# %%


