
#%%
import os
import sys

import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import demod

from importlib import reload
from demod.utils.monte_carlo import monte_carlo_from_1d_pdf
from demod.simulators.base_simulators import SimLogger
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator
from demod.utils.parse_helpers import convert_states
from demod.datasets.GermanTOU.parser import (
    get_mask_subgroup, get_tpms_activity,
    GTOU_label_to_Bottaccioli_act, primary_states, occ,
    df_pers
)
from demod.metrics.states import count
from demod.utils.subgroup_handling import subgroup_households_to_persons
# %%
reload(demod.simulators.activity_simulators)
from demod.simulators.activity_simulators import SubgroupsIndividualsActivitySimulator
sim = SubgroupsIndividualsActivitySimulator(
    [
        {'household_type': 2},
        {'household_type': 3, 'n_residents': 3},
        {'household_type': 4, 'n_residents': 5}
    ], [3, 2, 1],
)
sim.get_n_doing_activity(1)
sim.step()


# %%
reload(demod.utils.subgroup_handling)
from demod.utils.subgroup_handling import subgroup_households_to_persons, Subgroup
# A list of lists of subgrouups
persons_subgroups = subgroup_households_to_persons(
    [
        {'household_type': 2},
        {'household_type': 3, 'n_residents': 3},
        {'household_type': 4, 'n_residents': 5}
    ])
n_hh_list = [3, 2, 1]

subgroups_persons_types, inverse, person_numbers = np.unique(
    np.concatenate(persons_subgroups),
    return_counts=True, return_inverse=True
)


# Adds an empty list of each person type
hh_of_person = [[] for _ in person_numbers]

# Tracks how many persons of each sim where counted
persons_counted = np.array(person_numbers)
_past_hh_counts = 0  # Tracks how many hh have been visited
for i, (hh_persons, n_hh) in enumerate(zip(persons_subgroups, n_hh_list)):
    print(i)
    for pers_subgroup in hh_persons:
        print(pers_subgroup)
        # Find which will be the subsim that simulates this person
        _ind_subsim = list(subgroups_persons_types).index(pers_subgroup)
        print(_ind_subsim)
        # This persone suubgrouup has been counted n_hh times
        persons_counted[_ind_subsim] += n_hh
        # Appends the household number
        for hh in range(n_hh):
            hh_of_person[_ind_subsim].append(_past_hh_counts + hh)

    _past_hh_counts += n_hh



# %%
reload(demod.utils.parse_helpers)
subgroup = {
    'weekday': [6, 7],
    'only_fully_recorded_household': False,
    'remove_missing_days': False,
    'only_household_identical_days': False,
}

(
    tpm, duration, duration_with_previous, states_label,
    initial_pdf, intial_durations_pdf, dict_legend
) = get_tpms_activity(
    subgroup,
    activity_dict=GTOU_label_to_Bottaccioli_act,
    add_durations=True)

count(tpm)
# %%
from demod.simulators.activity_simulators import SemiMarkovSimulator
# %%

reload(demod.simulators.activity_simulators)
from demod.simulators.activity_simulators import SemiMarkovSimulator
sim = SemiMarkovSimulator(
    1000, len(states_label), tpm, duration, states_label,
    logger=SimLogger('current_states', aggregated=False)
)
# %%
sim.initialize_starting_state(initial_pdf, intial_durations_pdf)
# %%
for i in range(4*144):
    sim.step()

#%%

from demod.utils.plotters import plot_stack_states
import matplotlib.pyplot as plt
plot_stack_states(sim.logger.get('current_states').T, states_label)
plt.show()
# %% compare to the real states

mask_subgroup = get_mask_subgroup(**subgroup)
raw_states = primary_states[mask_subgroup]
# Add the away states
raw_states[~occ[mask_subgroup]] = 0
activity_dict = GTOU_label_to_Bottaccioli_act.copy()
activity_dict[0] = 'away'


states, states_label = convert_states(raw_states, activity_dict)

states_label[states_label == '-'] = 'other'


plot_stack_states(states, states_label)
plt.show()
# %%
monte_carlo_from_1d_pdf

states
