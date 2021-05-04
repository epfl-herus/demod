
# %%
from .parse_helpers import *
from..helpers import subgroup_file



dir_name = os.path.dirname(os.path.realpath(__file__))

def create_data_activity_profile(
    subgroup_kwargs,
    compiled_data_path = os.path.join(dir_name, 'compiled_data')):
    """Create the activity profiles of a day

    Args:
        subgroup_kwargs (dict): dictonary of the subgroup to generate the activity profile
        compiled_data_path (string, optional): The access path where the data should be saved.
            Defaults to os.path.join(__this_dir__, 'compiled_data').
    """



    #generate the activity profiles

    mask_subgroup = get_mask_subgroup( **subgroup_kwargs)

    # gets the activities we want
    main_activity_states, main_activity_labels = convert_states(
        primary_states[mask_subgroup], GTOU_label_to_CREST_act_v2)
    sec_activity_states, sec_activity_labels = convert_states(
        secondary_states[mask_subgroup], GTOU_label_to_CREST_act_v2)

    # convert to a 4 states model, and group the 4 states to households states to get the active occupancy
    merged_states, merged_labels = convert_states(10*occ + act)

    household_states = group_in_household_4states(
        np.array(merged_labels[merged_states])[mask_subgroup],
        household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
        days_indexes=np.array(df_akt['tagnr'])[mask_subgroup] )

    active_occ = np.minimum(household_states%10, household_states//10)

    # loop over the activities to be parsed
    all_activity_labels = ['Act_TV','Act_Cooking','Act_Laundry','Act_WashDress','Act_Iron','Act_HouseClean', 'Act_Dishwashing', 'Act_Elec']
    for desired_act in all_activity_labels:
        # gets the indice of the activity in the states
        main_act_ind = np.where(main_activity_labels==desired_act)[0]
        sec_act_ind  = np.where(sec_activity_labels==desired_act)[0]
        # if there is no record of the desired activity, choose an impossible index
        if len(main_act_ind) == 0:
            main_act_ind = -42
        if len(sec_act_ind) == 0:
            sec_act_ind = -42

        # find in each household how many people are doing the activity desired
        n_occ_performing_act = group_in_household_activity(
            (main_activity_states == main_act_ind) | (sec_activity_states == sec_act_ind),
            household_indexes=np.array(df_akt['id_hhx'])[mask_subgroup],
            days_indexes=np.array(df_akt['tagnr'])[mask_subgroup]
            )

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

            # get the path where the data will be saved

            path = subgroup_file(subgroup_kwargs, folder_path=os.path.join(compiled_data_path, 'activity_profiles') + os.sep)

            np.save(path+ '_' + desired_act + '_nactocc_' + str(n_occ), prob_at_least_one_occ_perfoming_act)





