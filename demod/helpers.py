""" This file should be removed when possible. """
import os
from .simulators.base_simulators import Simulator
from .simulators.util import OLD_DATASET_PATH

import numpy as np
import pandas as pd
import datetime
import warnings

import pytz


def subgroup_file(subgroup_kwargs, folder_path = os.path.join(OLD_DATASET_PATH + os.sep +'GermanTOU', 'compiled_data', 'sparse_activities') + os.sep):
    """Create a file name with the access path included for the specified subgroup kwargs

    Args:
        subgroup_kwargs (dict): The keyword arguments for the desired subgroup
        folder_path (string, optional): The path of the folder where the file is located. Defaults to os.path.join('GermanTOU', 'compiled_data', 'sparse_activities')+os.sep.

    Returns:
        string: contains the path location and the name of the file for the subgroup
    """
    return ''.join([folder_path] + [str(i)+'_' +str(j)+'_' if j and j is not True else '' for i, j in zip(subgroup_kwargs.keys(), subgroup_kwargs.values())])



def CREST_get_24h_occupancy(states, state_labels):
    """return the households that have a 24 h occupancy given the states

    Args:
        states (ndarray(int, size=(n_households, n_times))):
            The matrix of the states for the housholds at all the times
        states_labels (ndarray, size(n_states)):
            The labels for the states given as 1rst argument

    Returns:
        ndaray(bool, size=n_households): True if the household has a 24 hour occupancy else false
    """
    assert states.shape[1] == 144, "A wrong number of times for CREST was given in the states array, \
         must be 144 not " + str(states.shape[1])

    # the first value in the labels mean the number of residents that are at home
    occupancy24 = np.all(state_labels[states]// 10 > 0, axis=1)

    return occupancy24


def CREST_read_activity_pdf(day_type, add_levels=True):
    """
    return the
    1. the pdf for all the activities from crest
    2. the label of the activities"""
    path_name = OLD_DATASET_PATH + os.sep +'CREST_data' + os.sep + 'CREST_Demand_Model_v2.3.3.xlsm - ActivityStats.csv'

    if day_type == 'd':
        df = pd.read_csv(path_name, header=28, nrows=36, usecols=np.arange(144)+4)
    elif day_type == 'e':
        df = pd.read_csv(path_name, header=28+36, nrows=36, usecols=np.arange(144)+4)
    else:
        raise ValueError("incorrect day type was given  : " + day_type)
    pdf = np.array(df).T.reshape(144, 6, 6)

    labels = pd.read_csv(path_name, header=28, nrows=6, usecols=[3])
    labels = [lab.upper() for lab in labels['Unnamed: 3']]  # converts to list

    if add_levels:  # adds an activity that can always happen (pdf) = 1
        labels.append('LEVEL')
        pdf = np.concatenate((pdf, np.ones((144,6,1))), axis=2)



    return pdf, labels


def GTOU_read_activity_pdf(day_type, add_levels=True):
    """
    return the
    1. the pdf for all the activities from GTOU
    2. the label of the activities"""

    assert day_type == 'd' or day_type == 'e', "incorrect day type was given  : "

    path_name = OLD_DATASET_PATH + os.sep +'GermanTOU'+os.sep+'compiled_data'+os.sep+'activity_profiles'+os.sep
    pdf = np.array([]).reshape((-1,144))
    labels = ['Act_TV','Act_Cooking','Act_Laundry','Act_WashDress','Act_Iron','Act_HouseClean', 'Act_Dishwashing', 'Act_Elec']
    for desired_act in labels:

        for n_occ in range(0,6):

            if n_occ == 0:
                profile = np.zeros(144)
            else:
                profile = np.load(path_name + 'like_crest_w' + day_type + '_'+ desired_act +'_nactocc_'+str(n_occ) + '.npy')

            pdf = np.r_[pdf, profile.reshape((-1, 144))]
    pdf = pdf.reshape(len(labels), 6, 144)
    pdf = np.moveaxis(pdf, 2, 0)
    pdf = np.moveaxis(pdf, 1, 2)


    labels = [lab.upper() for lab in labels]  # converts to list

    if add_levels:  # adds an activity that can always happen (pdf) = 1
        labels.append('LEVEL')
        pdf = np.concatenate((pdf, np.ones((144,6,1))), axis=2)



    return pdf, np.array(labels)




def get_activity_probs_from_occupancy(occupancy, activities_pdf):
    """ At one step, get the active and at home persons
    args
    the active occupancy at the current time
    the probablity for each activity to be performed depending on the occupancy
    returns
        a boolean array for each houshold the new set of activites
        dim 0 is the households, dim 1 is the activites performed"""
    pdfs = activities_pdf[occupancy]  # get the pdfs for each type of occupancies we have


    return pdfs  # the pdf for each activities for each household

def get_switchon_probs(household_activity_pdfs, household_activity_labels, appliances, occupancies):
    # TODO check if the appliances switch on probabilites would not be better computed in
    # initialization and then accessed in memory


    # set up the array for the switch on pdf
    n_households = len(household_activity_pdfs)
    out = np.ones((n_households, appliances['number']))

    # multiply by the switch on probabilities of each appliances
    out *= appliances['switch-on probs']

    # access each appliance pdf in the activity pdfs
    for i, activity in enumerate(household_activity_labels):
        # gets which appliances are this type of activity
        mask = appliances['activation type'] == activity
        # multiply by the activity probablilty

        out[:, mask] *= household_activity_pdfs[:, i][:, None]



    # handles the appliances activities that are not in the activity labels
    mask = appliances['activation type'] == 'ACTIVE_OCC'
    is_occupied = occupancies > 0  # gets the occupied households
    out[:, mask] *= is_occupied[:, None]


    return out




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
