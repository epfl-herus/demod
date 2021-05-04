import datetime
import numpy as np
import pandas as pd
import os
import itertools
import warnings

from .base_simulators import  Simulator
from .util import OLD_DATASET_PATH
from ..utils.monte_carlo import  monte_carlo_from_1d_cdf, monte_carlo_from_cdf, monte_carlo_from_1d_pdf
from..helpers import subgroup_file


class ElectricCarSimulator(Simulator):
    MAX_CARS = 3
    STEP_IGNORE_TRANSITIONS = 1
    def read_cars(self, data='GTOU', **kwargs):
        path = OLD_DATASET_PATH + os.sep +'GermanTOU' + os.sep + 'inputs.ods'
        df = pd.read_excel(path, header=1, skiprows=[2,3], sheet_name='Electric Cars')
        dic_cars = {}
        dic_cars['type'] = df['Car Type'].to_numpy()
        dic_cars['proportion'] = df['Proportion of car'].to_numpy()
        dic_cars['capacity kWh'] = df['Battery capacity'].to_numpy()
        dic_cars['consumption kWh/100km'] = df['Consumption'].to_numpy()

        dic_cars['max charge power kW'] = df['Max charging power AC'].to_numpy()

        return dic_cars

    def read_charging_station(self, data='GTOU', **kwargs):
        path = OLD_DATASET_PATH + os.sep +'GermanTOU' + os.sep + 'inputs.ods'
        df = pd.read_excel(path, header=1, skiprows=[2,3], sheet_name='Charging Station')
        dic_station = {}
        dic_station['type'] = df['Station Type'].to_numpy()
        dic_station['proportion'] = df['Proportion of dwellings with this station'].to_numpy()
        dic_station['max charge power kW'] = df['Max charging power'].to_numpy()

        return dic_station

    def sample_n_cars(self, subgroup_kwargs):
        # read the folder
        warnings.warn('number of car pdf is now hard coded, must change implementation')
        pdf = np.array([0.14292968, 0.50291952, 0.29017517, 0.06397563])
        return monte_carlo_from_1d_pdf(pdf, n_samples=self.n_households)

    def attributes_cars(self, subgroup_kwargs):
        total_n_cars = np.sum(self.n_cars)
        self.total_n_cars = total_n_cars
        self.car_types = monte_carlo_from_1d_pdf(self.dic_cars['proportion'], n_samples=total_n_cars)
        self.car_consumptions = np.array(self.dic_cars['consumption kWh/100km'][self.car_types])
        self.battery_capacity = np.array(self.dic_cars['capacity kWh'][self.car_types])
        self.car_hh = np.array([sum([n*[i] for i,n in enumerate(self.n_cars)], [])], dtype=int).reshape(-1) # the household to which the car belongs

    def attributes_stations(self, subgroup_kwargs):
        r = monte_carlo_from_1d_pdf(self.dic_station['proportion'], n_samples=self.n_households)
        self.station_power = self.dic_station['max charge power kW'][r] * 1000 # convert to watts

    def attributes_stategy(self, subgroup_kwargs, charging_attribution_method = 'after use', **kwargs):
        # 0 = direct after use
        # 1 = timer
        self.charging_strategy = np.zeros(self.n_households, dtype=int)
        # creates times between 10pm and 8 am
        self.car_timer = np.random.randint(108, 108+60, size= self.total_n_cars) % 144
        if charging_attribution_method == 'after use':
            return
        elif charging_attribution_method == 'timers':
            self.charging_strategy = 1
        elif charging_attribution_method == 'mixed':
            self.charging_strategy = np.random.randint(0,2,size=len(self.charging_strategy))
        else:
            raise ValueError('Unkonwn chargin strategy')


    def __init__(self, n_households, initial_occupancy, initial_HOH, initial_HWH,  subgroup_kwargs, start_time_step=0, **kwargs):
        # define the parameters

        super().__init__(n_households, **kwargs)

        self.dic_cars = self.read_cars(**kwargs)
        self.dic_station = self.read_charging_station(**kwargs)

        self.n_cars = self.sample_n_cars(subgroup_kwargs)
        self.occupancy = initial_occupancy
        self.n_HOH = initial_HOH
        self.n_HWH = initial_HWH

        self.attributes_cars(subgroup_kwargs)
        self.attributes_stations(subgroup_kwargs)
        self.attributes_stategy(subgroup_kwargs, **kwargs)

        # the probabilities that a car is used, given the number of people leaving
        # from the GTOU
        self.prob_leave_car_HOH = np.array([ 0, 0.62679598, 0.81792453, 0.87903226, 0.93333333,1.])
        self.prob_leave_car_HWH = np.array([0.0, 0.52228261, 0.47798742, 0.5 , 0.5, 0.5])
        self.probs_leave_with_car = np.zeros_like(self.occupancy, dtype=float)
        self.initialize_starting_state(start_time_step)

    def initialize_starting_state(self, start_time_step):
        self.in_use = np.zeros_like(self.car_hh, dtype=bool)
        self.times_left_chargin = np.zeros_like(self.car_hh, dtype=int)
        self.times_used = np.zeros_like(self.car_hh, dtype=int)
        self.is_charging = np.zeros_like(self.car_hh, dtype=bool)
        return super().initialize_starting_state(start_time_step=start_time_step)

    def compute_trip_energy(self, mask_arriving_cars):
        """returns tirip energy in kwh

        Args:
            mask_arriving_cars ([type]): [description]
        """
        # compute the energy consumed by the car from the trip
        ratio_car_used_trip = 0.5
        average_speed = 46. #km/h # commuting speed https://nhts.ornl.gov/2009/pub/stt.pdf
        km_performed = self.times_used[mask_arriving_cars] /6. * average_speed * ratio_car_used_trip # 10min time steps
        kWh_consumed = self.car_consumptions[mask_arriving_cars] * km_performed / 100. # cons in kwh/100km
        return kWh_consumed

    def step(self, occupancy, n_HOH, n_HWH, increment=datetime.timedelta(minutes=10)):
        super().step()
        # compare the values with the old stored


        # increment the values of interest

        self.times_used[self.in_use] += 1


        # check if there is an arrival
        mask_arrival = self.occupancy < occupancy

        n_cars_in_use = np.bincount(self.car_hh[self.in_use], minlength=self.n_households)
        # defines if the arrival was done by car, this samples random probabilities for a car arrival
        r = monte_carlo_from_1d_pdf([0.1, 0.9], n_samples=self.n_households)
        # arrival if car if all occupants left out have a car or if randomly decided
        mask_hh_where_car_arrival = ((mask_arrival & r) | ((n_HOH + n_HWH) < n_cars_in_use)) & (n_cars_in_use>0)
        # get a car of this houshold

        mask_candidate_cars = np.isin(self.car_hh, np.where(mask_hh_where_car_arrival)[0]) & self.in_use

        # choose only one of the candidate
        _, ind = np.unique(self.car_hh[mask_candidate_cars], return_index=True)
        # arrival
        mask_arriving_cars = np.zeros_like(mask_candidate_cars)
        mask_arriving_cars[np.where(mask_candidate_cars)[0][ind]] = True

        if self.current_time_step%144 == self.STEP_IGNORE_TRANSITIONS:
            mask_arriving_cars[:] = False # avoid car arrival at change of day to avoid discontinuities form occ sim

        self.in_use[mask_arriving_cars] = False

        kWh_consumed = self.compute_trip_energy(mask_arriving_cars)

        time_to_charge = (kWh_consumed * 3.6e6) / self.station_power[self.car_hh[mask_arriving_cars]] /10. / 60. # convert to joules and the to 10 min timesteps
        time_to_charge = np.array(time_to_charge, dtype=int) # convert to integers
        self.times_left_chargin[mask_arriving_cars] += time_to_charge
        self.times_used[mask_arriving_cars] = 0

        # print(self.times_left_chargin[mask_arriving_cars] )

        self.start_charging(mask_arriving_cars)

        # update the departures
        mask_departure = self.occupancy > occupancy
        n_leaving = self.occupancy - occupancy
        mask_HWH = (self.n_HWH < n_HWH) & mask_departure
        mask_HOH = (self.n_HOH < n_HOH) & mask_departure



        probs_leave_with_car = np.zeros_like(mask_departure, dtype=float)
        probs_leave_with_car[mask_HWH] = self.prob_leave_car_HWH[n_leaving[mask_HWH]]
        probs_leave_with_car[mask_HOH] = self.prob_leave_car_HOH[n_leaving[mask_HOH]] # hoh will overwrite hwh changes
        self.probs_leave_with_car = probs_leave_with_car
        # draw from random sample


        hh_where_car_leaves = np.where(probs_leave_with_car > np.random.uniform(size=len(probs_leave_with_car)))[0]
        # print(hh_where_car_leaves)
        available_cars_mask = np.isin(self.car_hh, hh_where_car_leaves) & (~self.in_use)

        # makes only one car per household leave
        _, ind = np.unique( self.car_hh[available_cars_mask], return_index=True)
        # print(np.where(available_cars_mask)[0][ind])
        mask_leaving_cars = np.zeros_like(mask_candidate_cars)
        mask_leaving_cars[np.where(available_cars_mask)[0][ind]]= True
        if self.current_time_step%144 == self.STEP_IGNORE_TRANSITIONS:
            mask_leaving_cars[:] = False # avoid car arrival at change of day to avoid discontinuities form occ sim
        # updates cars that leave
        self.in_use[mask_leaving_cars] = True
        self.is_charging[mask_leaving_cars] = False


        # update the occupancy
        self.occupancy = np.array(occupancy)
        self.n_HWH = np.array(n_HWH)
        self.n_HOH = np.array(n_HOH)

        self.update_charging()


    def start_charging(self, mask_car_start_charging):

        # where after use charging stat chargin
        self.is_charging[mask_car_start_charging & (self.charging_strategy[self.car_hh]==0)] = True
        # 6 am timers
        t = self.current_time_step%144
        #assume we start simulation at 4 am
        # for timeres after 4 am
        n_times_till_charged = np.where( self.car_timer - t > 0, self.car_timer - t, self.car_timer + (144-t))
        # for timers before 4 am
        self.is_charging[(self.times_left_chargin == n_times_till_charged)  & (self.charging_strategy[self.car_hh]==1) & (~self.in_use)] = True
        # if the times left chargin is to large, set to charge
        self.is_charging[(self.times_left_chargin>144) & (~self.in_use)]

    def update_charging(self):

        self.times_left_chargin[self.is_charging] -= 1
        self.is_charging[self.times_left_chargin < 1] = False # end of charging

    def get_electric_consumption(self):
        """return the charging consumption in W

        Returns:
            [ndarray(n_households)]: [description]
        """
        unique, counts = np.unique(self.car_hh[self.is_charging], return_counts=True)
        n_charging = np.zeros(self.n_households, dtype=float)
        n_charging[unique] = counts
        return n_charging * self.station_power

    def get_car_electric_consumption(self):
        """retrun the elec consumption avferga by car in W

        Returns:
            ndarray(n_cars): [description]
        """

        ratio_car_used_trip = 0.5
        average_speed = 46. #km/h # commuting speed https://nhts.ornl.gov/2009/pub/stt.pdf
        car_consumption = np.zeros(self.total_n_cars, dtype=float)
        car_consumption[self.in_use] = (self.car_consumptions[self.in_use]*3.6e6) / 100 * ratio_car_used_trip  * average_speed / 3600 # calibrate the conumption with

        return car_consumption

    def get_n_used_cars(self):
        unique, counts = np.unique(self.car_hh[self.in_use], return_counts=True)
        n_used = np.zeros(self.n_households, dtype=int)
        n_used[unique] = counts
        return n_used

