
# %%

from .parse_helpers import *


import os, sys
import datetime
import json

import numpy as np
from demod.simulators.sparse_simulators import *
from..helpers import  subgroup_file




def create_data_sparse9states(subgroup_kwargs, compiled_data_path = os.path.join('GermanTOU', 'compiled_data')):

    # transportation model
    states_out_for_what , labels_out_for_what = states_to_out_for_what_model(
        occ,
        primary_states,
        secondary_states)

    # convert states
    primary_activities, activity_labels = convert_states(primary_states, GTOU_label_to_energy_activity)
    secondary_activities, sec_activity_labels = convert_states(secondary_states, GTOU_label_to_energy_activity)
    #states = states_out_for_what
    #labels = labels_out_for_what
    active_states = (activity_labels[primary_activities] != 'not active') & (sec_activity_labels[secondary_activities] != 'not active')
    labels_9states = ['active']

    # puts it to a N-states power model
    personas_states = np.array(active_states, dtype=np.uint64)



    # replace states by the two commuting states
    activity_offset = len(labels_9states)
    labels_9states = np.append(labels_9states, labels_out_for_what[1:]) # don't override the in house activites in the labels
    personas_states = np.where(
        states_out_for_what != 0, # 0 is the 'In-house' state
        MAX_PEOPLE_HOUSEHOLD**np.array(activity_offset + states_out_for_what - 1, dtype=np.uint64),
        personas_states
    )


    # gets the concerned households
    mask_subgroup = get_mask_subgroup( **subgroup_kwargs)


    # group activities by households
    hh_states = group_in_household_activity(
        personas_states[mask_subgroup],
        household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
        days_indexes=np.array(df_akt['tagnr'])[mask_subgroup])


    # convert states to a sparse TPM
    states, states_label = convert_states(hh_states)
    tpm = states_to_sparse_tpm(states, first_matrix_strategy='nothing')

    # get the path where the data will be saved
    path = subgroup_file(subgroup_kwargs, folder_path=os.path.join(compiled_data_path, 'sparse_9states') + os.sep)

    # saves what needs to be saved
    try:
        # save the transition prob matrix
        tpm.save(path)
        # get the pdf of the initial distribution and save it
        initial_counts = np.bincount( states[:,0], minlength=tpm.n_states)
        initial_pdf = initial_counts/ np.sum(initial_counts)
        np.save(path+'_initialpdf', initial_pdf )
        # save the labels
        np.save(path+'_labels', states_label)
        np.save(path+'_activity_labels', labels_9states )
        dict_legend = {}
        dict_legend['number of households diaries'] = len(hh_states)
        dict_legend['number of persons diaries'] = int(np.sum(mask_subgroup))
        dict_legend['subgroup_kwargs'] = subgroup_kwargs
        dict_legend['date'] = str(datetime.datetime.now())
        with open(path+'_dict_legend.json', 'w') as fp:
            json.dump(dict_legend, fp, indent=4)
    except FileNotFoundError as e:
        raise ValueError('invalid folder path, must contain a folder called "sparse_9states"')



# %%
