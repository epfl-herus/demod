# %%
import os, sys
import warnings

import numpy as np
import pandas as pd



# %%

df = pd.read_excel(os.path.join('daten','appliances_penetration.ods'), header=[0,1,2])

# %% defautl names and probabilities for the appliances 
appliances_names = [
    'FRIDGE1',
    'FRIDGE2',
    'FREEZER1',
    'FREEZER2',
    'PHONE',
    'IRON',
    'VACUUM',
    'PC1',
    'PC2',
    'LAPTOP1',
    'LAPTOP2',
    'TABLET',
    'PRINTER',
    'TV1',
    'TV2',
    'VCR_DVD',
    'RECEIVER',
    'CONSOLE',
    'HOB_ELEC',
    'HOB_GAZ',
    'OVEN',
    'MICROWAVE',
    'KETTLE',
    'SMALL_COOKING',
    'DISH_WASHER',
    'TUMBLE_DRYER',
    'WASHING_MACHINE',
    'WASHER_DRYER',
    'DESWH',
    'E_INST',
    'ELEC_SHOWER',
    'BASIN',
    'SINK',
    'SHOWER',
    'BATH']

appliances_probs_default = [
    0.997,
    0.233,
    0.482,
    0.057,
    0.849,
    0.900,
    0.937,
    0.442,
    0.102,
    0.739,
    0.332,
    0.475,
    0.752,
    0.943,
    0.618,
    0.607,
    0.927,
    0.297,
    0.94,
    0.061,
    0.616,
    0.713,
    0.975,
    1.000,
    0.719,
    0.423,
    0.95,
    0.153,
    0.000,
    0.000,
    0.000,
    0.994,
    1.000,
    0.997,
    0.916]


# %%
offset_nb = 4
offset_percent = 55
#appliances_labels = df['Unnamed: 1_level_0'].to_numpy().reshape(-1)

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

def get_column_appliance_subgroup(
    n_residents=None, household_revenue=None, year=None, household_type=None,
     life_situation=None, age=None, geburtsland=None, gender=None, household_position=None,
    is_travelling=None, **kwargs):
    """Returns the column in the file that correspond to german appliances.
    Basic implementation cannot combine lots of attributes.
    It will first check if both household_type and n__residents are given
    The it will check n_residents, revenue and year

    Args:
        n_residents (int, optional): The number of residents in the household. Defaults to None.
        household_revenue ([type], optional): [description]. Defaults to None.
        year ([type], optional): [description]. Defaults to None.
        household_type (int, optional): The type of the household (1 = Einpersonenhaushalt, 2 = Paare ohne Kinder, 3 = Alleinerziehende mit mindestens einem Kind unter18 Jahren und ledigen Kindern unter 27 Jahren, 4= Paare mit mindestens einem Kind unter 18 Jahren und ledigen Kindernledigen Kindern unter 27 Jahren, 5 = Sonstige Haushalte). Defaults to None.
        gender (int, optional): 1 = man, 2 = woman. Defaults to None.
        age (int or tuple, optional): The age of the participants, if tuple returns all inside interval. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    columns = []

    # first check the family types
    offset_hh_type = 47
    offset_hh_with_children = 57
    if household_type == 1: # leaving alone,
        if gender is None:
            return offset_hh_type + 0
        elif gender == 1: #man
            return offset_hh_type + 1
        elif gender == 2: #woman
            return offset_hh_type + 2
        else:
            raise ValueError('Gender must be 1, 2 or None')
    elif household_type == 2: # couple withouth children
        return offset_hh_type + 5
    elif household_type == 3 and n_residents == 2: # lone parents 1 children 
        return offset_hh_with_children + 1
    elif household_type == 3 and n_residents == 3: # lone parents 2 children 
        return offset_hh_with_children + 2
    elif household_type == 3 and n_residents == 4: # lone parents 3 children 
        return offset_hh_with_children + 2
    elif household_type == 4 and n_residents == 3: # couple with 1 children 
        return offset_hh_with_children + 4
    elif household_type == 4 and n_residents == 4: # lone parents 2 children 
        return offset_hh_with_children + 5
    elif household_type == 4 and n_residents == 5: # lone parents 3 children 
        return offset_hh_with_children + 6


    # finds the columns corresponding to the desired socio economical types
    elif n_residents:
        assert n_residents > 0 and n_residents < 6, 'n_residents must be from 1 to 5'
        offset_n_residents = 12
        return offset_n_residents + n_residents
    
    elif household_revenue:
        offset_revenue = 27
        assert household_revenue >= 0, ' household_revenue cannot be negative '
        if household_revenue < 900:
            col = 0
        elif household_revenue < 1300:
            col = 1
        elif household_revenue < 1500:
            col = 2
        elif household_revenue < 2000:
            col = 3
        elif household_revenue < 2600:
            col = 4
        elif household_revenue < 3600:
            col = 5
        elif household_revenue < 5000:
            col = 6
        else:
            col = 7
        
        return offset_revenue + col
    
    elif year:
        if year == 2008:
            col = 2
        elif year == 2013:
            col = 3
        elif year == 2018:
            col = 4
        else:
            raise ValueError('Year not valid, only 2008, 2013 and 2018')

        return col


    

    
    else:
        raise ValueError('Could not find any appliance set for  subgroup')




def load_appliances_penetration_germany(subgroup_kwargs, data_path=os.path.join('daten','appliances_penetration.ods')):
    """Loads the appliances equippement probabilities for the desired subgroup

    Args:
        subgroup_kwargs ([type]): [description]
    
    Return:
        np.ndarray : the probabilities of equppement of each appliance.

    Note:
        The generation algorithm is fixed and everything will fail if we modify appliances
        from any of the excell datafiles. Must be improved in the future.
    """
    col_n = get_column_appliance_subgroup(**subgroup_kwargs)

    df = pd.read_excel(data_path, header=[0,1,2])

    column = df[df.columns[col_n]].to_numpy()
    column[column=='-'] = 0 # replace the strig char
    column = np.asfarray(column) / 100.
    appliances_probs = np.array(appliances_probs_default)

    # replace the appliances that have the 

    # fridge :
    offset_app = 39
    appliances_probs[0] = column[offset_percent + offset_app] 
    # for the secondary, the prob of having is the average number - prob having one, but min 1. 
    appliances_probs[1] = min(column[offset_nb + offset_app]- column[offset_percent + offset_app], 1.)
    # freezer:
    offset_app = 40
    appliances_probs[2] = column[offset_percent + offset_app] 
    appliances_probs[3] = min(column[offset_nb + offset_app]- column[offset_percent + offset_app], 1.)
    # fixed phone:
    offset_app = 34
    appliances_probs[4] = column[offset_percent + offset_app] 
    # iron : None
    # Vaccum : None
    # fixed computer
    offset_app = 25
    appliances_probs[7] = column[offset_percent + offset_app] 
    appliances_probs[8] = min(column[offset_nb + offset_app]- column[offset_percent + offset_app], 1.)
    # laptop
    offset_app = 27
    appliances_probs[9] = column[offset_percent + offset_app] 
    appliances_probs[10] = min(column[offset_nb + offset_app]- column[offset_percent + offset_app], 1.)
    # tablet:
    offset_app = 28
    appliances_probs[11] = column[offset_percent + offset_app] 
    # printer:
    offset_app = 29
    appliances_probs[12] = column[offset_percent + offset_app] 
    # television
    offset_app = 8
    appliances_probs[13] = column[offset_percent + offset_app] 
    appliances_probs[14] = min(column[offset_nb + offset_app]- column[offset_percent + offset_app], 1.)
    # blueray dvd reader:
    offset_app = 13
    appliances_probs[15] = column[offset_percent + offset_app] 
    # tv box: None
    # game console
    offset_app = 22
    appliances_probs[17] = column[offset_percent + offset_app] 
    # cooking plates elec
    offset_app = 45
    appliances_probs[18] = column[offset_percent + offset_app] 
    # cooking plates gaz
    offset_app = 46
    appliances_probs[19] = column[offset_percent + offset_app]
    # oven : none
    # micro wave
    offset_app = 42
    appliances_probs[21] = column[offset_percent + offset_app] 
    # kettle : none 
    # small cooking : none 
    # dish washer
    offset_app = 41
    appliances_probs[24] = column[offset_percent + offset_app] 
    # dryer
    offset_app = 44
    appliances_probs[25] = column[offset_percent + offset_app] 
    # washing machine
    offset_app = 43
    appliances_probs[26] = column[offset_percent + offset_app] 
    # water appliances : none


    return np.array(appliances_probs)
# %%
# get the appliances that matter
appliances_indexes = np.array([
    0, #  Cars
    #1, #  new
    #2, #  occasion
    #3, #  leasing
    4, #  Motorcyles
    5, #  Bicycles
    6, #  of which electrics
    #7, #  nan
    8, #  TV
    #9, #  of which flatscreen
    #10, #  Reciever sat
    #11, #  Reciever cable
    #12, #  Reciever antenna
    13, #  DVD & Blu-ray reader
    #14, #  of which blue-ray
    #15, #  Camcorder
    #16, #  Camcorder analog
    #17, #  Camcorder digital
    #18, #  Photo camera
    #19, #  of which analog
    #20, #  of which digital
    #21, #  MP3 player
    22, #  Game console
    #23, #  nan
    #24, #  Computer
    25, #  Fixed computer
    #26, #  mobile
    27, #  Laptop/Notebook
    28, #  Tablet
    29, #  Printer
    30, #  Internet reciever
    #31, #  of which cable
    #32, #  of which mobile
    #33, #  Telephone
    34, #  fixed telephone
    35, #  Mobile phone
    #36, #  of which smart phone
    #37, #  GPS
    #38, #  nan
    39, #  Fridge
    40, #  Freezer
    41, #  Dish washer
    42, #  Micro-wave
    43, #  Washing machine
    44, #  Dryer
    45, #  Cooking plates electric
    46, #  Cooking plates gaz
    47, #  Fitness machine
])
# %%
# build a pdf for the number of appliance available

col_n = get_column_appliance_subgroup(year=2018)
print(df.columns[col_n])
prob_equipped = df[df.columns[col_n]].to_numpy()[appliances_indexes+offset_percent].reshape(-1)
prob_equipped[prob_equipped=='-'] = np.nan
prob_equipped /= 100
prob_equipped  = np.array(prob_equipped , dtype=float)
n_equipped = df[df.columns[col_n]].to_numpy()[appliances_indexes+offset_nb].reshape(-1)
n_equipped[n_equipped=='-'] = 100
n_equipped /= 100
n_equipped = np.array(n_equipped, dtype=float)


pdf = np.c_[[
    1 - prob_equipped,
    p1 := 2. * prob_equipped - n_equipped,
    (n_equipped - p1) / 2.
]].T


prob_having = np.c_[[
    prob_equipped, # prob having 1 time the appliance
    p1 := np.where(n_equipped - prob_equipped < 1 , n_equipped - prob_equipped, 1), # prob having a second time the appliance
    n_equipped - prob_equipped - p1 # prob having a third time the appliance
]].T



# %%
pdf = np.c_[[
    1 - prob_equipped,
    np.where((p1 := 2. * prob_equipped - n_equipped) > 0, p1, 0),
    np.where(p1 > 0, (n_equipped - p1) / 2., np.where(
        (p2:= 3*prob_equipped - n_equipped) > 0, p2, 0)),
    np.where(p1 > 0, 0, np.where(
        p2 > 0, n_equipped - 2 * prob_equipped, np.where(
            (p3:=  4*prob_equipped - n_equipped) > 0, p3, 0 ))),
    np.where(p1 > 0, 0, np.where(
        p2 > 0, 0, np.where(
            p3 > 0, n_equipped - 3 * prob_equipped, np.where(
                (p4:=  5*prob_equipped - n_equipped) > 0, p4, 0 )))), 
    np.where(p1 > 0, 0, np.where(
        p2 > 0, 0, np.where(
            p3 > 0, 0, np.where(
                p4 > 0, n_equipped - 4 * prob_equipped, np.where(
                    (p5:=  6*prob_equipped - n_equipped) > 0, p5, 0 ))))),   
    np.where(p1 > 0, 0, np.where(
        p2 > 0, 0, np.where(
            p3 > 0, 0, np.where(
                p4 > 0, 0, np.where(
                    p5 > 0, n_equipped - 5 * prob_equipped, np.where(
                        (p6:=  7*prob_equipped - n_equipped) > 0, p6, 0 )))))),   
    np.where(p1 > 0, 0, np.where(
        p2 > 0, 0, np.where(
            p3 > 0, 0, np.where(
                p4 > 0, 0, np.where(
                    p5 > 0, 0, np.where(
                        p6 > 0, n_equipped - 6 * prob_equipped, np.where(
                            (p7:=  8*prob_equipped - n_equipped) > 0, p7, 0 ))))))),   
    np.where(p1 > 0, 0, np.where(
        p2 > 0, 0, np.where(
            p3 > 0, 0, np.where(
                p4 > 0, 0, np.where(
                    p5 > 0, 0, np.where(
                        p6 > 0, n_equipped - 6 * prob_equipped, np.where(
                            (p7:=  8*prob_equipped - n_equipped) > 0, p7, 0 ))))))),   
]].T
# %%
appliances = {}
appliances['names'] = []
appliances['equipped probs'] = []
appliances['type'] = []

# threshold to check if the ownership is relevant
threshold = 0.1

for probs, label in zip(prob_having, appliances_labels[appliances_indexes+offset_percent]):
    print(probs,label)
    for i, p in enumerate(probs):
        if p > threshold:
            appliances['names'].append(label + '_' + str(i+1))
            appliances['equipped probs'].append(p)
            appliances['type'].append(label)

# convert the list to np arrays
for i in appliances:
    appliances[i] = np.array(appliances[i])

# %%

power_dict = {

}
#%%