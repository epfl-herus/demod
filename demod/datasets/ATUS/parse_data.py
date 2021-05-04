
# %%
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demod.simulators.simulators import OccupancySimulator, TimedStatesSimulator
from..helpers import get_states_durations, RMSE, CREST_get_24h_occupancy

# %%
df = pd.read_csv('atusact_2019.dat')
# %% get the limits between the diaries
persons, starts_indicies, persons_indices = np.unique(df['TUCASEID'], return_index=True, return_inverse=True)
stops_indicies = np.roll(starts_indicies, -1) - 1
assert len(np.unique(df['TUSTARTTIM'][starts_indicies])) == 1, 'Start of diary should be at the same time for all reports'
# %%
df.columns
# %%
activities, activities_indices, activities_inverse = np.unique(df['TRCODE'], return_index=True, return_inverse=True)
df['activity index'] = df['TRCODE'].apply(lambda x : int(activities_indices[activities==x]))
# %% converts the time to indices
df['start hour'] = df['TUSTARTTIM'].apply(lambda x : int(str(x)[:2]))
df['start minute'] = df['TUSTARTTIM'].apply(lambda x : int(str(x)[3:5]))
# converts from our to min.indices and makes the day start at 4 am and force indices to be positive
df['start index'] = (df['start hour']*60 + df['start minute'] - 4*60) % (24*60)


mask_old_activity = np.arange(-1, len(df)-1)
mask_old_activity[starts_indicies] = stops_indicies
# get the transitions (the time, the previous state, the next state)
transitions_indexes = np.asarray((
    df['start index'].to_numpy(),
    activities_inverse[mask_old_activity],
    activities_inverse,
    persons_indices))
# %% populates the transition matrices

n_states = len(activities)
n_steps = 60*24
transitions_indexes = [(i) for i in transitions_indexes] # converts the indexes in 3 arrays for accessing the matrix later


# %%
ATUS_label_to_activity = {
    10101 :  'sleeping',
    10102 :  'sleeping',
    10201 :  'washing / dressing',
    10299 :  'washing / dressing',
    10301 :  'care',
    10399 :  'other',
    10401 :  'other',
    10499 :  'other',
    10501 :  'other',
    19999 :  'other',
    20101 :  'house cleaning', # "Interior cleaning";
    20102 :  'laundry', # "Laundry";
    20103 :  'other', # "Sewing, repairing, and maintaining textiles";
    20104 :  'other', # "Storing interior hh items, inc. food";
    20199 :  'house cleaning', # "Housework, n.e.c.*";
    20201 :  'cooking', # "Food and drink preparation";
    20202 :  'cooking', # "Food presentation";
    20203 :  'cooking', # "Kitchen and food clean-up";
    20299 :  'cooking', # "Food and drink prep, presentation, and clean-up, n.e.c.*";
    20301 :  'house cleaning', # "Interior arrangement, decoration, and repairs";
    20302 :  'house cleaning', # "Building and repairing furniture";
    20303 :  'other', # "Heating and cooling";
    20399 :  'house cleaning', # "Interior maintenance, repair, and decoration, n.e.c.*";
    20401 :  'house cleaning', # "Exterior cleaning";
    20402 :  'house cleaning', # "Exterior repair, improvements, and decoration";
    20499 :  'house cleaning', # "Exterior maintenance, repair and decoration, n.e.c.*";
    20501 :  'house cleaning', # "Lawn, garden, and houseplant care";
    20502 :  'house cleaning', # "Ponds, pools, and hot tubs";
    20599 :  'house cleaning', # "Lawn and garden, n.e.c.*";
    20601 :  'other', # "Care for animals and pets (not veterinary care)";
    20602 :  'leisure no appliance', # "Walking / exercising / playing with animals";
    20699 :  'leisure no appliance', # "Pet and animal care, n.e.c.*";
    20701 :  'other', # "Vehicle repair and maintenance (by self)";
    20799 :  'other', # "Vehicles, n.e.c.*";
    20801 :  'other', # "Appliance, tool, and toy set-up, repair, and maintenance (by self)";
    20899 :  'other', # "Appliances and tools, n.e.c.*";
    20901 :  'home office', # "Financial management";
    20902 :  'home office', # "Household and personal organization and planning";
    20903 :  'home office', # "HH and personal mail and messages (except e-mail)";
    20904 :  'home office', # "HH and personal e-mail and messages";
    20905 :  'home office', # "Home security";
    20999 :  'home office', # "Household management, n.e.c.*";
    29999 :  'home office', # "Household activities, n.e.c.*";
    30101 :  'other', # "Physical care for hh children";
    30102 :  'leisure no appliance', # "Reading to/with hh children";
    30103 :  'leisure no appliance', # "Playing with hh children, not sports";
    30104 :  'leisure no appliance', # "Arts and crafts with hh children";
    30105 :  'leisure no appliance', # "Playing sports with hh children";
    30106 :  'leisure no appliance', # "Talking with/listening to hh children";
    30108 :  'leisure no appliance', # "Organization and planning for hh children";
    30109 :  'leisure no appliance', # "Looking after hh children (as a primary activity)";
    30110 :  'leisure no appliance', # "Attending hh children's events";
    30111 :  'leisure no appliance', # "Waiting for/with hh children";
    30112 :  'mobility', # "Picking up/dropping off hh children";
    30199 :  'leisure no appliance', # "Caring for and helping hh children, n.e.c.*";
    30201 :  'home office', # "Homework (hh children)";
    30202 :  'leisure no appliance', # "Meetings and school conferences (hh children)";
    30203 :  'home office', # "Home schooling of hh children";
    30204 :  'home office', # "Waiting associated with hh children's education";
    30299 :  'home office', # "Activities related to hh child's education, n.e.c.*";
    30301 :  'care', # "Providing medical care to hh children";
    30302 :  'care', # "Obtaining medical care for hh children";
    30303 :  'care', # "Waiting associated with hh children's health";
    30399 :  'care', # "Activities related to hh child's health, n.e.c.*";
    30401 :  'care', # "Physical care for hh adults";
    30402 :  'care', # "Looking after hh adult (as a primary activity)";
    30403 :  'care', # "Providing medical care to hh adult";
    30404 :  'care', # "Obtaining medical and care services for hh adult";
    30405 :  'care', # "Waiting associated with caring for household adults";
    30499 :  'care', # "Caring for household adults, n.e.c.*";
    30501 :  'care', # "Helping hh adults";
    30502 :  'home office', # "Organization and planning for hh adults";
    30503 :  'mobility', # "Picking up/dropping off hh adult";
    30504 :  'care', # "Waiting associated with helping hh adults";
    30599 :  'care', # "Helping household adults, n.e.c.*";
    40101 :  'care', # "Physical care for nonhh children";
    40102 :  'leisure no appliance', # "Reading to/with nonhh children";
    40103 :  'leisure no appliance', # "Playing with nonhh children, not sports";
    40104 :  'leisure no appliance', # "Arts and crafts with nonhh children";
    40105 :  'leisure no appliance', # "Playing sports with nonhh children";
    40106 :  'leisure no appliance', # "Talking with/listening to nonhh children";
    40108 :  'leisure no appliance', # "Organization and planning for nonhh children";
    40109 :  'leisure no appliance', # "Looking after nonhh children (as primary activity)";
    40110 :  'leisure no appliance', # "Attending nonhh children's events";
    40111 :  'leisure no appliance', # "Waiting for/with nonhh children";
    40112 :  'mobility', # "Dropping off/picking up nonhh children";
    40199 :  'care', # "Caring for and helping nonhh children, n.e.c.*";
    40201 :  'home office', # "Homework (nonhh children)";
    40202 :  'leisure no appliance', # "Meetings and school conferences (nonhh children)";
    40301 :  'care', # "Providing medical care to nonhh children";
    40302 :  'care', # "Obtaining medical care for nonhh children";
    40399 :  'care', # "Activities related to nonhh child's health, n.e.c.*";
    40401 :  'care', # "Physical care for nonhh adults";
    40402 :  'care', # "Looking after nonhh adult (as a primary activity)";
    40403 :  'care', # "Providing medical care to nonhh adult";
    40404 :  'care', # "Obtaining medical and care services for nonhh adult";
    40405 :  'care', # "Waiting associated with caring for nonhh adults";
    40499 :  'care', # "Caring for nonhh adults, n.e.c.*";
    40501 :  'care', # "Housework, cooking, and shopping assistance for nonhh adults";
    40502 :  'care', # "House and lawn maintenance and repair assistance for nonhh adults";
    40503 :  'care', # "Animal and pet care assistance for nonhh adults";
    40504 :  'care', # "Vehicle and appliance maintenance/repair assistance for nonhh adults";
    40505 :  'care', # "Financial management assistance for nonhh adults";
    40506 :  'care', # "Household management and paperwork assistance for nonhh adults";
    40507 :  'mobility', # "Picking up/dropping off nonhh adult";
    40508 :  'care', # "Waiting associated with helping nonhh adults";
    40599 :  'care', # "Helping nonhh adults, n.e.c.*";
    49999 :  'care', # "Caring for and helping nonhh members, n.e.c.*";
    50101 :  'job', # "Work, main job";
    50102 :  'job', # "Work, other job(s)";
    50103 :  'job', # "Security procedures related to work";
    50104 :  'job', # "Waiting associated with working";
    50199 :  'job', # "Working, n.e.c.*";
    50202 :  'eat', # "Eating and drinking as part of job";
    50205 :  'job', # "Waiting associated with work-related activities";
    50299 :  'job', # "Work-related activities, n.e.c.*";
    50301 :  'job', # "Income-generating hobbies, crafts, and food";
    50302 :  'job', # "Income-generating performances";
    50303 :  'job', # "Income-generating services";
    50304 :  'job', # "Income-generating rental property activities";
    50399 :  'job', # "Other income-generating activities, n.e.c.*";
    50401 :  'job', # "Job search activities";
    50403 :  'job', # "Job interviewing";
    50404 :  'job', # "Waiting associated with job search or interview";
    59999 :  'job', # "Work and work-related activities, n.e.c.*";
    60101 :  'school', # "Taking class for degree, certification, or licensure";
    60102 :  'school', # "Taking class for personal interest";
    60103 :  'school', # "Waiting associated with taking classes";
    60199 :  'school', # "Taking class, n.e.c.*";
    60201 :  'school', # "Extracurricular club activities";
    60202 :  'school', # "Extracurricular music and performance activities";
    60204 :  'school', # "Waiting associated with extracurricular activities";
    60301 :  'school', # "Research/homework for class for degree, certification, or licensure";
    60302 :  'school', # "Research/homework for class for pers. interest";
    60399 :  'school', # "Research/homework n.e.c.*";
    60401 :  'school', # "Administrative activities: class for degree, certification, or licensure";
    60402 :  'school', # "Administrative activities: class for personal interest";
    60499 :  'school', # "Administrative for education, n.e.c.*";
    69999 :  'school', # "Education, n.e.c.*";
    70101 :  'shopping', # "Grocery shopping";
    70102 :  'mobility', # "Purchasing gas";
    70103 :  'shopping', # "Purchasing food (not groceries)";
    70104 :  'shopping', # "Shopping, except groceries, food and gas";
    70105 :  'shopping', # "Waiting associated with shopping";
    70199 :  'shopping', # "Shopping, n.e.c.*";
    70201 :  'shopping', # "Comparison shopping";
    70301 :  'shopping', # "Security procedures rel. to consumer purchases";
    80101 :  'care', # "Using paid childcare services";
    80201 :  'shopping', # "Banking";
    80202 :  'shopping', # "Using other financial services";
    80203 :  'shopping', # "Waiting associated w/banking/financial services";
    80301 :  'shopping', # "Using legal services";
    80399 :  'shopping', # "Using legal services, n.e.c.*";
    80401 :  'shopping', # "Using health and care services outside the home";
    80402 :  'care', # "Using in-home health and care services";
    80403 :  'care', # "Waiting associated with medical services";
    80501 :  'care', # "Using personal care services";
    80502 :  'care', # "Waiting associated w/personal care services";
    80601 :  'shopping', # "Activities rel. to purchasing/selling real estate";
    80701 :  'care', # "Using veterinary services";
    80702 :  'care', # "Waiting associated with veterinary services";
    89999 :  'others', # "Professional and personal services, n.e.c.*";
    90101 :  'house cleaning', # "Using interior cleaning services";
    90103 :  'house cleaning', # "Using clothing repair and cleaning services";
    90104 :  'house cleaning', # "Waiting associated with using household services";
    90199 :  'house cleaning', # "Using household services, n.e.c.*";
    90201 :  'house cleaning', # "Using home maint/repair/d\E9cor/construction svcs";
    90202 :  'house cleaning', # "Waiting associated w/ home main/repair/d\E9cor/constr";
    90299 :  'house cleaning', # "Using home maint/repair/d\E9cor/constr services, n.e.c.*";
    90301 :  'others', # "Using pet services";
    90302 :  'others', # "Waiting associated with pet services";
    90399 :  'others', # "Using pet services, n.e.c.*";
    90401 :  'others', # "Using lawn and garden services";
    90402 :  'others', # "Waiting associated with using lawn and garden services";
    90501 :  'others', # "Using vehicle maintenance or repair services";
    90502 :  'others', # "Waiting associated with vehicle main. or repair svcs";
    90599 :  'others', # "Using vehicle maint. and repair svcs, n.e.c.*";
    99999 :  'others', # "Using household services, n.e.c.*";
    100101 :  'others', # "Using police and fire services";
    100102 :  'others', # "Using social services";
    100103 :  'others', # "Obtaining licenses and paying fines, fees, taxes";
    100199 :  'others', # "Using government services, n.e.c.*";
    100201 :  'others', # "Civic obligations and participation";
    100299 :  'others', # "Civic obligations and participation, n.e.c.*";
    100304 :  'others', # "Waiting associated with using government services";
    110101 :  'eat', # "Eating and drinking";
    110201 :  'eat', # "Waiting associated w/eating and drinking";
    120101 :  'leisure no appliance', # "Socializing and communicating with others";
    120201 :  'leisure no appliance', # "Attending or hosting parties/receptions/ceremonies";
    120202 :  'leisure no appliance', # "Attending meetings for personal interest (not volunteering)";
    120299 :  'leisure no appliance', # "Attending/hosting social events, n.e.c.*";
    120301 :  'leisure no appliance', # "Relaxing, thinking";
    120302 :  'leisure no appliance', # "Tobacco and drug use";
    120303 :  'TV', # "Television and movies (not religious)";
    120304 :  'TV', # "Television (religious)";
    120305 :  'radio', # "Listening to the radio";
    120306 :  'TV', # "Listening to/playing music (not radio)";
    120307 :  'leisure no appliance', # "Playing games";
    120308 :  'computer', # "Computer use for leisure (exc. Games)";
    120309 :  'leisure no appliance', # "Arts and crafts as a hobby";
    120310 :  'leisure no appliance', # "Collecting as a hobby";
    120311 :  'leisure no appliance', # "Hobbies, except arts and crafts and collecting";
    120312 :  'leisure no appliance', # "Reading for personal interest";
    120313 :  'leisure no appliance', # "Writing for personal interest";
    120399 :  'leisure no appliance', # "Relaxing and leisure, n.e.c.*";
    120401 :  'leisure no appliance', # "Attending performing arts";
    120402 :  'leisure no appliance', # "Attending museums";
    120403 :  'leisure no appliance', # "Attending movies/film";
    120404 :  'leisure no appliance', # "Attending gambling establishments";
    120405 :  'leisure no appliance', # "Security procedures rel. to arts and entertainment";
    120499 :  'leisure no appliance', # "Arts and entertainment, n.e.c.*";
    120501 :  'leisure no appliance', # "Waiting assoc. w/socializing and communicating";
    120502 :  'leisure no appliance', # "Waiting assoc. w/attending/hosting social events";
    120503 :  'leisure no appliance', # "Waiting associated with relaxing/leisure";
    120504 :  'leisure no appliance', # "Waiting associated with arts and entertainment";
    129999 :  'leisure no appliance', # "Socializing, relaxing, and leisure, n.e.c.*";
    130101 :  'sports', # "Doing aerobics";
    130102 :  'sports', # "Playing baseball";
    130103 :  'sports', # "Playing basketball";
    130104 :  'sports', # "Biking";
    130105 :  'sports', # "Playing billiards";
    130106 :  'sports', # "Boating";
    130107 :  'sports', # "Bowling";
    130108 :  'sports', # "Climbing, spelunking, caving";
    130109 :  'sports', # "Dancing";
    130110 :  'sports', # "Participating in equestrian sports";
    130112 :  'sports', # "Fishing";
    130113 :  'sports', # "Playing football";
    130114 :  'sports', # "Golfing";
    130116 :  'sports', # "Hiking";
    130117 :  'sports', # "Playing hockey";
    130118 :  'sports', # "Hunting";
    130119 :  'sports', # "Participating in martial arts";
    130120 :  'sports', # "Playing racquet sports";
    130122 :  'sports', # "Rollerblading";
    130124 :  'sports', # "Running";
    130125 :  'sports', # "Skiing, ice skating, snowboarding";
    130126 :  'sports', # "Playing soccer";
    130127 :  'sports', # "Softball";
    130128 :  'sports', # "Using cardiovascular equipment";
    130129 :  'sports', # "Vehicle touring/racing";
    130130 :  'sports', # "Playing volleyball";
    130131 :  'sports', # "Walking";
    130132 :  'sports', # "Participating in water sports";
    130133 :  'sports', # "Weightlifting/strength training";
    130134 :  'sports', # "Working out, unspecified";
    130136 :  'sports', # "Doing yoga";
    130199 :  'sports', # "Playing sports n.e.c.*";
    130202 :  'leisure no appliance', # "Watching baseball";
    130203 :  'leisure no appliance', # "Watching basketball";
    130207 :  'leisure no appliance', # "Watching bowling";
    130209 :  'leisure no appliance', # "Watching dancing";
    130210 :  'leisure no appliance', # "Watching equestrian sports";
    130213 :  'leisure no appliance', # "Watching football";
    130214 :  'leisure no appliance', # "Watching golfing";
    130216 :  'leisure no appliance', # "Watching hockey";
    130218 :  'leisure no appliance', # "Watching racquet sports";
    130219 :  'leisure no appliance', # "Watching rodeo competitions";
    130222 :  'leisure no appliance', # "Watching running";
    130224 :  'leisure no appliance', # "Watching soccer";
    130225 :  'leisure no appliance', # "Watching softball";
    130226 :  'leisure no appliance', # "Watching vehicle touring/racing";
    130227 :  'leisure no appliance', # "Watching volleyball";
    130232 :  'leisure no appliance', # "Watching wrestling";
    130299 :  'leisure no appliance', # "Attending sporting events, n.e.c.*";
    130301 :  'leisure no appliance', # "Waiting related to playing sports or exercising";
    130302 :  'leisure no appliance', # "Waiting related to attending sporting events";
    130399 :  'leisure no appliance', # "Waiting associated with sports, exercise, and recreation, n.e.c.*";
    130402 :  'leisure no appliance', # "Security related to attending sporting events";
    139999 :  'leisure no appliance', # "Sports, exercise, and recreation, n.e.c.*";
    140101 :  'leisure no appliance', # "Attending religious services";
    140102 :  'leisure no appliance', # "Participation in religious practices";
    140103 :  'leisure no appliance', # "Waiting associated w/religious and spiritual activities";
    140105 :  'leisure no appliance', # "Religious education activities";
    149999 :  'leisure no appliance', # "Religious and spiritual activities, n.e.c.*";
    150101 :  'computer', # "Computer use";
    150102 :  'leisure no appliance', # "Organizing and preparing";
    150103 :  'leisure no appliance', # "Reading";
    150104 :  'phone', # "Telephone calls (except hotline counseling)";
    150105 :  'leisure no appliance', # "Writing";
    150106 :  'others', # "Fundraising";
    150199 :  'others', # "Administrative and support activities, n.e.c.*";
    150201 :  'cooking', # "Food preparation, presentation, clean-up";
    150202 :  'others', # "Collecting and delivering clothing and other goods";
    150203 :  'care', # "Providing care";
    150204 :  'care', # "Teaching, leading, counseling, mentoring";
    150299 :  'care', # "Social service and care activities, n.e.c.*";
    150301 :  'others', # "Building houses, wildlife sites, and other structures";
    150302 :  'others', # "Indoor and outdoor maintenance, repair, and clean-up";
    150399 :  'others', # "Indoor and outdoor maintenance, building and clean-up activities, n.e.c.*";
    150401 :  'leisure no appliance', # "Performing";
    150402 :  'leisure no appliance', # "Serving at volunteer events and cultural activities";
    150499 :  'leisure no appliance', # "Participating in performance and cultural activities, n.e.c.*";
    150501 :  'leisure no appliance', # "Attending meetings, conferences, and training";
    150602 :  'leisure no appliance', # "Public safety activities";
    150699 :  'leisure no appliance', # "Public health and safety activities, n.e.c.*";
    150701 :  'leisure no appliance', # "Waiting associated with volunteer activities";
    150799 :  'leisure no appliance', # "Waiting associated with volunteer activities, n.e.c.*";
    150801 :  'leisure no appliance', # "Security procedures related to volunteer activities";
    150899 :  'leisure no appliance', # "Security procedures related to volunteer activities, n.e.c.*";
    159999 :  'leisure no appliance', # "Volunteer activities, n.e.c.*";
    160101 :  'phone', # "Telephone calls to/from family members";
    160102 :  'phone', # "Telephone calls to/from friends, neighbors, or acquaintances";
    160103 :  'phone', # "Telephone calls to/from education services providers";
    160104 :  'phone', # "Telephone calls to/from salespeople";
    160105 :  'phone', # "Telephone calls to/from professional or personal care svcs providers";
    160106 :  'phone', # "Telephone calls to/from household services providers";
    160107 :  'phone', # "Telephone calls to/from paid child or adult care providers";
    160108 :  'phone', # "Telephone calls to/from government officials";
    160199 :  'phone', # "Telephone calls (to or from), n.e.c.*";
    160201 :  'phone', # "Waiting associated with telephone calls";
    169999 :  'phone', # "Telephone calls, n.e.c.*";
    180101 :  'mobility', # "Travel related to personal care";
    180201 :  'mobility', # "Travel related to housework";
    180202 :  'mobility', # "Travel related to food and drink prep., clean-up, and presentation";
    180203 :  'mobility', # "Travel related to interior maintenance, repair, and decoration";
    180204 :  'mobility', # "Travel related to exterior maintenance, repair, and decoration";
    180205 :  'mobility', # "Travel related to lawn, garden, and houseplant care";
    180206 :  'mobility', # "Travel related to care for animals and pets (not vet care)";
    180207 :  'mobility', # "Travel related to vehicle care and maintenance (by self)";
    180208 :  'mobility', # "Travel related to appliance, tool, and toy set-up, repair, and maintenance (by self)";
    180209 :  'mobility', # "Travel related to household management";
    180299 :  'mobility', # "Travel related to household activities, n.e.c.*";
    180301 :  'mobility', # "Travel related to caring for and helping hh children";
    180302 :  'mobility', # "Travel related to hh children's education";
    180303 :  'mobility', # "Travel related to hh children's health";
    180304 :  'mobility', # "Travel related to caring for hh adults";
    180305 :  'mobility', # "Travel related to helping hh adults";
    180401 :  'mobility', # "Travel related to caring for and helping nonhh children";
    180402 :  'mobility', # "Travel related to nonhh children's education";
    180403 :  'mobility', # "Travel related to nonhh children's health";
    180404 :  'mobility', # "Travel related to caring for nonhh adults";
    180405 :  'mobility', # "Travel related to helping nonhh adults";
    180499 :  'mobility', # "Travel rel. to caring for and helping nonhh members, n.e.c.*";
    180501 :  'mobility', # "Travel related to working";
    180502 :  'mobility', # "Travel related to work-related activities";
    180503 :  'mobility', # "Travel related to income-generating activities";
    180504 :  'mobility', # "Travel related to job search and interviewing";
    180599 :  'mobility', # "Travel related to work, n.e.c.*";
    180601 :  'mobility', # "Travel related to taking class";
    180602 :  'mobility', # "Travel related to extracurricular activities (ex. Sports)";
    180603 :  'mobility', # "Travel related to research/homework";
    180604 :  'mobility', # "Travel related to registration/administrative activities";
    180699 :  'mobility', # "Travel related to education, n.e.c.*";
    180701 :  'mobility', # "Travel related to grocery shopping";
    180702 :  'mobility', # "Travel related to purchasing gas";
    180703 :  'mobility', # "Travel related to purchasing food (not groceries)";
    180704 :  'mobility', # "Travel related to shopping, ex groceries, food, and gas";
    180799 :  'mobility', # "Travel related to consumer purchases, n.e.c.*";
    180801 :  'mobility', # "Travel related to using childcare services";
    180802 :  'mobility', # "Travel related to using financial services and banking";
    180803 :  'mobility', # "Travel related to using legal services";
    180804 :  'mobility', # "Travel related to using medical services";
    180805 :  'mobility', # "Travel related to using personal care services";
    180806 :  'mobility', # "Travel related to using real estate services";
    180807 :  'mobility', # "Travel related to using veterinary services";
    180899 :  'mobility', # "Travel rel. to using prof. and personal care services, n.e.c.*";
    180901 :  'mobility', # "Travel related to using household services";
    180902 :  'mobility', # "Travel related to using home main./repair/d\E9cor./construction svcs";
    180903 :  'mobility', # "Travel related to using pet services (not vet)";
    180905 :  'mobility', # "Travel related to using vehicle maintenance and repair services";
    181001 :  'mobility', # "Travel related to using government services";
    181002 :  'mobility', # "Travel related to civic obligations and participation";
    181101 :  'mobility', # "Travel related to eating and drinking";
    181201 :  'mobility', # "Travel related to socializing and communicating";
    181202 :  'mobility', # "Travel related to attending or hosting social events";
    181203 :  'mobility', # "Travel related to relaxing and leisure";
    181204 :  'mobility', # "Travel related to arts and entertainment";
    181205 :  'mobility', # "Travel as a form of entertainment";
    181299 :  'mobility', # "Travel rel. to socializing, relaxing, and leisure, n.e.c.*";
    181301 :  'mobility', # "Travel related to participating in sports/exercise/recreation";
    181302 :  'mobility', # "Travel related to attending sporting/recreational events";
    181399 :  'mobility', # "Travel related to sports, exercise, and recreation, n.e.c.*";
    181401 :  'mobility', # "Travel related to religious/spiritual practices";
    181499 :  'mobility', # "Travel rel. to religious/spiritual activities, n.e.c.*";
    181501 :  'mobility', # "Travel related to volunteering";
    181599 :  'mobility', # "Travel related to volunteer activities, n.e.c.*";
    181601 :  'mobility', # "Travel related to phone calls";
    181801 :  'mobility', # "Security procedures related to traveling";
    181899 :  'mobility', # "Security procedures related to traveling, n.e.c.*";
    189999 :  'mobility', # "Traveling, n.e.c.*";
    500101 :  'unavailable data', # "Insufficient detail in verbatim";
    500103 :  'unavailable data', # "Missing travel or destination";
    500105 :  'unavailable data', # "Respondent refused to provide information/'none of your business'";
    500106 :  'unavailable data', # "Gap/can't remember";
    500107 :  'unavailable data', # "Unable to code activity at 1st tier";
    }

df['activity'] = df['TRCODE'].apply(lambda x: ATUS_label_to_activity[x])
activities, activities_indices, activities_inverse = np.unique(df['activity'], return_index=True, return_inverse=True)


# %%
def transitions_to_states(
    transitions_times, transitions_persons, transitions_new_states,
    initial_states=None,
    n_times=None, n_persons=None, n_states=None):


    transitions_times = np.array(transitions_times, dtype=int)
    transitions_persons = np.array(transitions_persons, dtype=int)
    transitions_new_states = np.array(transitions_new_states, dtype=int)

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


    assert len(transitions_times) == len(transitions_persons), 'length of transitions arrays must be the same'
    assert len(transitions_times) == len(transitions_new_states), 'length of transitions arrays must be the same'


    if initial_states is None: # initalize the state if not given
        initial_states = np.array(transitions_new_states[transitions_times==0])
    assert len(initial_states) == n_persons, ' The length of the initial states should match the number of persons'
    # checks that the initial transitions are the initial states
    assert np.all(transitions_new_states[transitions_times==0] ==  \
        initial_states[transitions_persons[transitions_times == 0]]), 'initial states must correspond to the transitions_times'
    this_states = np.array(initial_states)

    states = []
    for t in range(n_times):
        # search for the transitions happening at this state
        mask_this_transitions = transitions_times == t

        # gets the persons that change now
        transitioning_persons = transitions_persons[mask_this_transitions]

        # gets the next states for households that change
        next_states = np.array(this_states)
        next_states[transitioning_persons] = transitions_new_states[mask_this_transitions]

        # save the new states
        this_states = next_states
        states.append(this_states)

    return np.asarray(states).T

#%%
def build_transition_matrices(
    initial_states,
    transitions_times, transitions_persons, transitions_new_states,
    n_times=None, n_persons=None, n_states=None, first_matrix_strategy='last', return_states=False):

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

    if return_states:
        states = []
    # this matrix will store the transitions probabilites
    transition_matrices = np.zeros((n_times, n_states, n_states))
    for t in range(n_times):
        # search for the transitions happening at this state
        mask_this_transitions = transitions_times == t

        # gets the persons that change now
        transitioning_persons = transitions_persons[mask_this_transitions]

        # gets the next states for households that change
        next_states = np.array(this_states)
        next_states[transitioning_persons] = transitions_new_states[mask_this_transitions]

        # define and counts the transitions
        states_indices, states_counts = np.unique(np.asarray((this_states, next_states)), axis=1, return_counts=True)
        states_indices = [(i) for i in states_indices] # converts the indexes for accessing the matrix later

        # compute the sum of the transitions for each states
        transition_matrices[t][states_indices] = states_counts

        # save the new states
        this_states = next_states
        if return_states:
            states.append(this_states)

    # define what we should do with the first matrix that has false transitions
    if first_matrix_strategy == 'last':
        transition_matrices[0] = transition_matrices[-1]
    elif first_matrix_strategy == 'nothing':
        pass
    else:
        raise TypeError('Unknown first matrix stragtegy kwarg')

    # converts to probs
    transition_matrices = transition_matrices / np.sum(transition_matrices, axis=2)[:,:,None]
    if return_states:
        return transition_matrices, np.array(states)
    else:
        return transition_matrices


# %%
def get_mask_remove_fake_transitions(
    transitions_times, transitions_persons, transitions_new_states,
    n_times=None, n_persons=None, n_states=None):
    """Remove the transitions that happen right after another.
    Ignore the transitions at time 0
    """
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

    assert len(transitions_times) == len(transitions_persons), 'length of transitions arrays must be the same'
    assert len(transitions_times) == len(transitions_new_states), 'length of transitions arrays must be the same'



    transitions_times = np.array(transitions_times, dtype=int)
    transitions_persons = np.array(transitions_persons, dtype=int)
    transitions_new_states = np.array(transitions_new_states, dtype=int)

    # get the initial states
    this_states = transitions_new_states[transitions_times==0]

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

    assert np.all(transitions_times <= times_to_be_compared_to), 'the times must be subsorted for calculation the durations'



    real_indices = []
    previous_state = -1
    previous_person = -1

    for i, ( person, new_state) in enumerate(zip(transitions_persons, transitions_new_states)):
        # check if we don't over pass to the new person
        if previous_person != person:
            real_indices.append(i) # if we overpass accept the index
        elif previous_state == new_state:# don-t accept, ignore
            pass
        else: # transition to a new state so accept
            real_indices.append(i)

        # update the previous for the next
        previous_person = person
        previous_state = new_state

    mask = np.zeros_like(transitions_times, dtype=bool)
    mask[np.array(real_indices, dtype=int)] = True
    return mask

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

# %%
transitions_to_states(
transitions_times=     [0,1,2,3,4],
transitions_persons=   [0,0,0,0,0],
transitions_new_states=[0,1,1,1,0]
)
# %%
transitions_to_states(
transitions_times=     [0,1,2,3,4, 0,1,3, 0,1],
transitions_persons=   [0,0,0,0,0, 1,1,1, 2,2],
transitions_new_states=[0,1,1,1,0, 0,1,0, 1,1]
)
# %%
true_indices = get_mask_remove_fake_transitions(
transitions_times=     [0,1,2,3,4, 0,1,3, 0,1],
transitions_persons=   [0,0,0,0,0, 1,1,1, 2,2],
transitions_new_states=[0,1,1,1,0, 0,1,0, 1,1]
)
true_indices
# %%
build_transition_matrices([0],
transitions_times=     [0,1,2],
transitions_persons=   [0,0,0],
transitions_new_states=[0,1,0], first_matrix_strategy='nothing')
# %%
build_timed_based_transition_matrices([0,1],
transitions_times=     [0,1,2,3,4, 0],
transitions_persons=   [0,0,0,0,0, 1],
transitions_new_states=[0,1,1,1,0, 1], first_matrix_strategy='nothing')
# %%
build_timed_based_transition_matrices([0,1,1],
transitions_times=     [0,1,2,3,4,0,3,4,0,2,3],
transitions_persons=   [0,0,0,0,0,1,1,1,2,2,2],
transitions_new_states=[0,1,1,1,0,1,0,1,1,0,1], first_matrix_strategy='nothing')
# %%
build_transition_matrices([0, 0],
transitions_times=     [0,1,3,0,2,3],
transitions_persons=   [0,0,0,1,1,1],
transitions_new_states=[0,1,0,0,1,0], n_times=6)
# %%
build_transition_matrices([0,2,1,0],
transitions_times=     [0,1,2,0,1,2,3,1,3],
transitions_persons=   [0,0,0,1,1,2,2,3,3],
transitions_new_states=[0,1,0,2,0,1,2,0,1])
# %%
# initial_states = activities_inverse[df['start index'] == 0]



# transition_matrices = build_transition_matrices(
#     initial_states,
#     transitions_times=df['start index'],
#     transitions_persons=persons_indices,
#     transitions_new_states=activities_inverse)


# %% converts the activities to occupancyes
df['at home'] = np.where(df['TEWHERE']==1, 1, 0)
df['active'] = np.where(df['activity']=='sleeping', 0, 1)
# makes people who sleep and did not answer where that they are at home
df['at home'] = np.where(
    np.logical_and(df['activity']=='sleeping', df['TEWHERE']<0),
    1, df['at home'] )
# %% refactor the dataset to a simopler one

# only 4 states model activites
activities, activities_indices, activities_inverse = np.unique(
    df['at home']*10 + df['active'], return_index=True, return_inverse=True)

# 10 min resolution instead of 1, need to ignore the changes that happen in less than 10 min
transition_times_10min = (df['start index'] + 9)//10
mask_ignore = transition_times_10min == np.roll(transition_times_10min, 1)
mask_ignore[transition_times_10min==144] = True


mask_indices_used = np.array(~mask_ignore)
mask_indices_used[mask_indices_used] = get_mask_remove_fake_transitions(
    transitions_times=transition_times_10min[mask_indices_used],
    transitions_persons=persons_indices[mask_indices_used],
    transitions_new_states=activities_inverse[mask_indices_used]
)

# define the number of households we want to simulate
n_households_simulated = 100000
#%% Markovian 1rst order simulator

transition_matrices, true_data_states = build_transition_matrices(
    initial_states= activities_inverse[df['start index'] == 0],
    transitions_times=transition_times_10min[mask_indices_used],
    transitions_persons=persons_indices[mask_indices_used],
    transitions_new_states=activities_inverse[mask_indices_used], return_states=True)

sim = OccupancySimulator(n_households_simulated, 3, transition_matrices)
sim.initialize_starting_state([0,1,0])



# %%

for i in range(7*24*6): # let one week as an initialisation
    sim.step()

current_states = []
for i in range(24*6):

    sim.step()
    current_states.append(np.array(sim.current_states))

current_states = np.asarray(current_states)
# %%
stacked_states = np.apply_along_axis(np.bincount, 1, current_states,minlength=3)
plt.stackplot(np.arange(len(current_states)), stacked_states.T)

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
