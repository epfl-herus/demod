from typing import List


from ..datasets.base_loader import DatasetLoader
from ..datasets.CREST.loader import Crest
import itertools
import numpy as np
from numpy import random
from numpy.core.fromnumeric import size
from numpy.core.numeric import zeros_like
import pandas as pd

import sys
import os
import datetime
import math


from .base_simulators import (
    Callbacks,
    Simulator,
    MultiSimulator,
    TimeAwareSimulator,
    after_next_day_callback,
    cached_getter,
)
from .activity_simulators import ActivitySimulator, MarkovChain1rstOrder
from .appliance_simulators import AppliancesSimulator
from .util import OLD_DATASET_PATH, inherit_getters_docstring, sample_population
from ..utils.monte_carlo import (
    monte_carlo_from_1d_pdf,
    monte_carlo_from_1d_cdf,
    monte_carlo_from_cdf,
)
from ..utils.distribution_functions import rescale_pdf, check_valid_cdf
from ..utils import data_types


def CREST_data_load_pdf(n_residents, day_type, adjust_pdf_values=True):
    """Loads a pdf from the CREST data base

    Args:
        n_residents (int): the number of residents
        day_type (char): "d' or 'e' for weekday or weekend
        adjust_pdf_values (bool, optional): Whether we want to correct the values. Defaults to True.

    Raises:
        TypeError: if the type of n_residents is wrong

    Returns:
        ndarray: the pdf from CREST corresponding to the args
    """
    # first check the n residents input
    try:
        str_n_residents = str(n_residents)
        assert (
            int(str_n_residents) > 0 and int(str_n_residents) <= 6
        ), "Number of residents must be between 1 and 6"

    except:
        raise TypeError(
            "Number of residents was not correctly understood, must be int between [1, 6]"
        )

    pathpdfs = (
        OLD_DATASET_PATH
        + os.sep
        + "CREST_data"
        + os.sep
        + "CREST_Demand_Model_v2.3.3.xlsm - tpm"
        + str_n_residents
        + "_w"
        + day_type
        + ".csv"
    )
    # read the files
    transition = np.loadtxt(pathpdfs, skiprows=10, delimiter=",")
    transition_matrices = transition[:, 2:].reshape(
        (-1, (n_residents + 1) ** 2, (n_residents + 1) ** 2)
    )
    if adjust_pdf_values:
        # replaces the missing values by 1 where alls are 0 in the pdf
        # note that this is only required to have a valid cdf, but those states should never be reached by any household, if the data was correctly interpreted
        times, rows = np.where(transition_matrices.sum(axis=2) == 0)
        # make the stay at the same states
        transition_matrices[times, rows, rows] = 1.0
        transition_matrices = rescale_pdf(transition_matrices)
    return transition_matrices


@inherit_getters_docstring
class CrestOccupancySimulator(MarkovChain1rstOrder, ActivitySimulator):
    """Occupancy modelling as performed in CREST 4 states model.

    Simulates a single subgroup of given resident number and day type.
    Differentiate the households by number of residents and by weekdays
    vs weekends.

    Attributes:
        n_households: The number of households to be simulated.
        n_residents: The number of residents between 1 and 6
        daytype: :py:obj:`'e'` for weekend or :py:obj:`'d'` for weekday
    """

    n_households: int
    n_residents: int
    day_type: str

    def __init__(
        self,
        n_households: int,
        n_residents: int,
        day_type: str,
        data: data_types.DataInput = Crest(),
        adjustcdfvalues: bool = True,
        **kwargs
    ):
        """
        Initialize a simulator for the CREST model.
        Loads the correct files located in CREST_data to set up the parameters.
        The simulator start by default at the state between 00:00 and 00:10.

        Parameters:
            n_households:
                The number of households to be simulated.
            n_residents :
                The number of residents, can be between [1, 6]
            day_type :
                Can be 'e' for weekend or 'd' for weekday

        Raises:
            AssertionError
                If the parameters have wrong types or shapes
        """
        # check the inputs
        try:
            assert day_type == "e" or day_type == "d", (
                'day_type must be a str "e" or "d", not : ' + day_type
            )
        except:
            raise TypeError('day_type must be a str "e" or "d" ')

        # creates subgroup
        subgroup = {}
        subgroup['n_residents'] = n_residents
        subgroup['weekday'] = [1, 2, 3, 4, 5] if day_type == 'd' else [6, 7]

        transition_matrices, labels, startpdf = data.load_tpm(subgroup)

        self.n_residents = n_residents
        # crest 4 states has a n_states proportional to the n_residents
        super().__init__(
            n_households,
            (n_residents + 1) ** 2,
            transition_matrices,
            labels=labels,
            **kwargs
        )
        super().initialize_starting_state(startpdf, checkcdf=adjustcdfvalues)

    @cached_getter
    def get_occupancy(self) -> np.array:
        return self.state_labels[self.current_states] // 10

    @cached_getter
    def get_active_occupancy(self) -> np.array:
        current_state_labels = self.state_labels[self.current_states]
        return np.min(
            [current_state_labels // 10, current_state_labels % 10], axis=0
        )

    @cached_getter
    def get_thermal_gains(self) -> np.array:
        active_gains = 147
        dormant_gains = 84
        # gets the number of persons sleeping
        dormants = np.max(
            (
                (self.state_labels[self.current_states] // 10)
                - (self.state_labels[self.current_states] % 10),
                np.zeros_like(self.current_states),
            ),
            axis=0,
        )
        return (
            dormant_gains * dormants
            + active_gains * self.get_active_occupancy()
        )


class Crest4StatesModel(TimeAwareSimulator, MultiSimulator):
    """Simulates the whole population as in CREST.

    This simulates households based on the 4 States model of McKenna
    [McKenna2016]_.
    You can see more information at :ref:`overview_4_States`.
    The simulator distinguish only the number of residents from 1 to 5
    and between weekends and weekdays.
    It samples randomly how many households belong to which number of
    residents based on the pdf of the real households.

    :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
    must be implemented for
    :py:obj:`~demod.utils.cards_doc.Loader.population_type` = 'resident_number'


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.start_datetime`
        :py:attr:`~demod.utils.cards_doc.Params.population_sampling_algo`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.utils.cards_doc.Loader.load_tpm`
        :py:meth:`~demod.utils.cards_doc.Loader.load_population_subgroups`
    Step input
        None
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_active_occupancy`
        :py:meth:`~demod.utils.cards_doc.Sim.get_thermal_gains`
    Step size
        10 Minutes


    """

    current_day_type: str  # 'e' or 'd' for weekend\day
    simulators: List[CrestOccupancySimulator]


    def __init__(
        self,
        n_households: int,
        data: Crest = Crest(),
        start_datetime: datetime.datetime = datetime.datetime(2014, 1, 1, 0, 0, 0),
        **kwargs
    ):
        """Initialize a simulator for households of different number of
        residents based on CREST 4 States model.

        Args:
            n_households (int): the total number of households to simulate
            day_type (char): 'd' if it is a weekday, else 'e'
            start_datetime: When the simulation should start
        """

        # get the data of the households numbers

        subgroups, pdf_households, _ = data.load_population_subgroups(
            'resident_number'
        )
        self.data = data

        n_residents_labels = np.array([
            subgroup['n_residents'] for subgroup in subgroups
        ], dtype=int)

        numbers = sample_population(
            n_households, pdf_households, **kwargs
        )
        kwargs.pop('population_sampling_algo', None)

        # counts how many household of each number of residents were sampled
        n_residents = n_residents_labels[numbers > 0]
        n_subhouseholds = numbers[numbers > 0]

        # creates all the sub-simulators for occupancy
        self.current_day_type = "d" if start_datetime.isoweekday() <= 5 else "e"
        occupancy_simulators = [
            CrestOccupancySimulator(
                int(n_hous), n_res, self.current_day_type, data=data
            )
            for n_res, n_hous in zip(n_residents, n_subhouseholds)
        ]

        # attributes a subgroup
        [
            setattr(sim, "subgroup", {"n_residents": sim.n_residents})
            for sim in occupancy_simulators
        ]

        if "step_size" in kwargs:
            raise ValueError(
                "Invalid argument : 'step_size' is not required "
                "as 'Crest4StatesModel' can only use 10 minutes "
                "step_size."
            )

        # initialize the time aware part of the simuulator
        super().__init__(
            occupancy_simulators,
            # CREST uses 10 minutes steps
            step_size=datetime.timedelta(minutes=10),
            start_datetime=start_datetime,
            **kwargs
        )

        self.n_residents = np.concatenate(
            [
                n_res * np.ones((n_hous), dtype=int)
                for n_res, n_hous in zip(n_residents, n_subhouseholds)
            ]
        )

        super().initialize_starting_state(
            initialization_time=self.data.refresh_time,
        )

    def on_after_refresh_time(self) -> None:
        """Update the TPMs for the new daytype.

        Note:
            It was deduced that the change of TPM must be perform after
            midnight, as deduction seem to make think that the first
            TPM is for the transition 00:00 -> 00:10.
        """
        day_type = "d" if self.current_time.isoweekday() <= 5 else "e"
        if day_type != self.current_day_type:
            self.current_day_type = day_type
            # tansfrom day type to subgroup['weekday']
            subgroup = {}
            subgroup['weekday'] = (
                [1, 2, 3, 4, 5] if self.current_time.isoweekday() <= 5
                else [6, 7]
                )
            for sim in self.simulators:
                subgroup['n_residents'] = sim.n_residents
                tpms, new_labels, _ = self.data.load_tpm(subgroup)
                sim._set_tpm(tpms, new_labels=new_labels)

    def get_states_labeled(self):

        return np.concatenate(
            [sim.state_labels[sim.current_states] for sim in self.simulators]
        )

    @Callbacks.after_refresh_time
    def step(self) -> None:
        """Updates the states of the residents."""
        return super().step()


class CRESTAppliancesSimulator(AppliancesSimulator):
    """[summary]

    Args:
        AppliancesSimulator ([type]): [description]
    """

    def CREST_read_appliances(self):
        """return a dictionary with the informations for all the appliances"""
        path_name = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.2.xlsm - AppliancesAndWaterFixtures.csv"
        )
        df = pd.read_csv(
            path_name,
            header=0,
            skiprows=[0, 1, 2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44],
            nrows=35,
        )

        appliances = {}
        appliances["name"] = np.array(df["Unnamed: 3"])
        appliances["equipped dwellings probs"] = np.array(df["Unnamed: 4"])
        appliances["activation type"] = np.array(df["Unnamed: 5"])
        appliances["mean cycle durations"] = np.array(df["(min).1"])
        appliances["delay after cycle"] = np.array(df["(min).2"])
        appliances["switch-on probs"] = np.array(df["Unnamed: 28"])
        appliances["mean cycle power"] = np.array(
            df["(W)"]
        )  # will be flow for water
        appliances["standby power"] = np.array(df["(W).1"])

        mpf = np.array(df["Unnamed: 30"])  # needs to set some nan values to 1
        appliances["mean power factor"] = np.where(np.isnan(mpf), 1.0, mpf)

        appliances["number"] = len(appliances["name"])

        return appliances

    def GTOU_read_appliances(self):
        """return a dictionary with the informations for all the appliances"""
        path_name = (
            OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
        )
        df = pd.read_excel(
            path_name,
            sheet_name="Appliances",
            header=0,
            skiprows=[0, 1, 2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44],
            nrows=41,
        )

        appliances = {}
        appliances["name"] = np.array(df["Unnamed: 3"])
        appliances["equipped dwellings probs"] = np.array(df["Unnamed: 4"])
        appliances["activation type"] = np.array(df["Unnamed: 5"])
        appliances["mean cycle durations"] = np.array(df["(min).1"])
        appliances["delay after cycle"] = np.array(df["(min).2"])
        appliances["switch-on probs"] = np.array(df["Unnamed: 29"])
        appliances["mean cycle power"] = np.array(
            df["(W)"]
        )  # will be flow for water
        appliances["standby power"] = np.array(df["(W).1"])

        mpf = np.array(df["Unnamed: 31"])  # needs to set some nan values to 1
        appliances["mean power factor"] = np.where(np.isnan(mpf), 1.0, mpf)

        appliances["number"] = len(appliances["name"])

        return appliances

    def CREST_data_read_water_usage(self):
        path_name = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.2.xlsm - WaterUsage.csv"
        )
        df = pd.read_csv(path_name, header=2, skiprows=[0, 1, 2], nrows=151)
        water_usage = {}
        water_usage["Litres"] = np.array(df["k (event volume)"])
        water_usage["BASIN"] = np.array(
            np.cumsum(df["Probability of occurrence"])
        )
        water_usage["SINK"] = np.array(
            np.cumsum(df["Probability of occurrence"])
        )
        water_usage["SHOWER"] = np.array(np.cumsum(df["Unnamed: 4"]))
        water_usage["BATH"] = np.array(np.cumsum(df["Unnamed: 5"]))
        water_usage["CDFs"] = np.c_[
            water_usage["BASIN"],
            water_usage["SINK"],
            water_usage["SHOWER"],
            water_usage["BATH"],
        ].T
        water_usage["labels"] = np.array(["BASIN", "SINK", "SHOWER", "BATH"])

        return water_usage

    def __init__(self, n_households, data="CREST", **kwargs):

        if data == "CREST":
            appliances_dict = self.CREST_read_appliances()
            appliances_dict[
                "water usage cdfs"
            ] = self.CREST_data_read_water_usage()
        elif data == "GTOU":
            appliances_dict = self.GTOU_read_appliances()
            appliances_dict[
                "water usage cdfs"
            ] = self.CREST_data_read_water_usage()
        elif isinstance(data, dict):
            appliances_dict = data
        else:
            raise Exception("data type or value was not understood")

        super().__init__(n_households, appliances_dict, **kwargs)

    def switch_on(self, indexes_household, indexes_appliance):
        """Switch on model for CREST. uses a simple mean value for the duration

        Args:
            indexes_household ([type]): [description]
            indexes_appliance ([type]): [description]
        """
        # first switch on
        # add a simple value that is the mean
        self.n_steps_left[
            indexes_household, indexes_appliance
        ] = self.appliances["mean cycle durations"][indexes_appliance]
        # adds the time till refresh is possible
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
            self.appliances["mean cycle durations"][indexes_appliance]
            + self.appliances["delay after cycle"][indexes_appliance]
        )

        # get the water fixtures
        mask_water_fixture = np.isin(
            self.appliances["name"][indexes_appliance],
            self.appliances["water usage cdfs"]["labels"],
        )

        self._switch_on_water_fixtures(
            indexes_household[mask_water_fixture],
            indexes_appliance[mask_water_fixture],
        )

    def _switch_on_water_fixtures(self, indexes_household, indexes_appliance):

        # sample the durations from the cdfs
        indexes_water_appliances = np.zeros_like(indexes_appliance, dtype=int)
        # convert to the water appliance label
        for i, name in enumerate(
            self.appliances["water usage cdfs"]["labels"]
        ):
            indexes_water_appliances[
                self.appliances["name"][indexes_appliance] == name
            ] = i

        # get the cdfs of the appliances that will change
        cdfs = self.appliances["water usage cdfs"]["CDFs"][
            indexes_water_appliances
        ]
        # get the volumes that will be consumed
        volumes = self.appliances["water usage cdfs"]["Litres"][
            monte_carlo_from_cdf(cdfs)
        ]
        # compute the duration from the mean usage
        durations = (
            volumes / self.appliances["mean cycle power"][indexes_appliance]
        )  # power is flow for water appliances
        # water fixtures duration
        self.n_steps_left[indexes_household, indexes_appliance] = durations
        # adds the time till refresh is possible
        self.n_steps_till_refresh[indexes_household, indexes_appliance] = (
            durations + self.appliances["delay after cycle"][indexes_appliance]
        )

    def switch_off_inactive(self, active_occupancy):
        # switch off appliances that need switch off when there is no occupancy

        # first get the mask of the ones that are switched off
        mask_app_need_switchoff = np.logical_and.reduce(
            [
                self.appliances["activation type"] != "ACT_LAUNDRY",
                self.appliances["activation type"] != "ACT_DISHWASHER",
                self.appliances["activation type"] != "LEVEL",
            ]
        )
        mask_household_need_switchoff = active_occupancy == 0
        mask_need_switchoff = (
            mask_household_need_switchoff[:, None]
            * mask_app_need_switchoff[None, :]
        )

        # finish the time remaining for the ones that were being used
        self.n_steps_left[mask_need_switchoff] = 0
        # start or continue a refresh period for them
        self.n_steps_till_refresh[mask_need_switchoff] = np.minimum(
            self.n_steps_till_refresh[mask_household_need_switchoff][
                :, mask_app_need_switchoff
            ],
            self.appliances["delay after cycle"][mask_app_need_switchoff],
        ).reshape(-1)

    def get_current_power_consumptions(self):
        power_consumptions = np.zeros_like(self.n_steps_left, dtype=float)
        # check the appliances that are on or off and determine the values of power for on and off
        power_consumptions += (
            (self.n_steps_left == 0)
            * self.appliances["standby power"]
            * self.available_appliances
        )
        power_consumptions += (self.n_steps_left > 0) * self.appliances[
            "mean cycle power"
        ]

        # handle the special cases (washing machines)
        # get the indicies of the specials
        index = self.appliances["name"] == "WASHING_MACHINE"
        mask_available = self.available_appliances[:, index].reshape(-1)
        power_consumptions[
            mask_available, index
        ] = self._compute_washing_machine_power(
            self.n_steps_left[mask_available, index], "WASHING_MACHINE"
        )

        index = self.appliances["name"] == "WASHER_DRYER"
        mask_available = self.available_appliances[:, index].reshape(-1)
        power_consumptions[
            mask_available, index
        ] = self._compute_washing_machine_power(
            self.n_steps_left[mask_available, index], "WASHER_DRYER"
        )

        # the water appliances do not consume electricity
        index = self.appliances["name"] == "BASIN"
        power_consumptions[:, index] = 0.0
        index = self.appliances["name"] == "SINK"
        power_consumptions[:, index] = 0.0
        index = self.appliances["name"] == "SHOWER"
        power_consumptions[:, index] = 0.0
        index = self.appliances["name"] == "BATH"
        power_consumptions[:, index] = 0.0

        return power_consumptions

    def get_current_water_consumptions(self):
        water_consumptions = np.zeros_like(self.n_steps_left, dtype=float)
        # check the appliances that are on or off and determine the values of power for on and off
        water_consumptions += (
            (self.n_steps_left <= 0)
            * self.appliances["standby power"]
            * self.available_appliances
        )
        # water takes into account short duration events (less than 1 min)
        water_consumptions += (
            self.n_steps_left
            * ((self.n_steps_left > 0) & (self.n_steps_left < 1))
            * self.appliances["mean cycle power"]
        )
        water_consumptions += (self.n_steps_left >= 1) * self.appliances[
            "mean cycle power"
        ]

        # electrical appliances don't consume anytthing, so remove them
        mask_water = np.isin(
            self.appliances["name"],
            self.appliances["water usage cdfs"]["labels"],
        )
        water_consumptions[:, ~mask_water] = 0
        return water_consumptions

    def get_hot_water_thermal_transfer_coefficient(self):
        # sum the hot water demand from all the fixtures
        demands = self.get_current_water_consumptions().sum(axis=1)
        # Then calculate variable thermal resistance representing hot water demand
        # conversion from litres to m^3 and from per minute to per second
        dblV_w = demands / 1000 / 60

        # to convert from m^3 per second to kg per second
        # set the density of water
        dblRho_w = 1000
        dblM_w = dblRho_w * dblV_w

        # convert to a thermal heat transfer coefficient in W/K
        SPECIFIC_HEAT_CAPACITY_WATER = 4200.0
        return SPECIFIC_HEAT_CAPACITY_WATER * dblM_w

    def _compute_washing_machine_power(self, n_steps_left, name):
        if name == "WASHING_MACHINE":
            current_time = 138 - n_steps_left
        elif name == "WASHER_DRYER":
            current_time = 198 - n_steps_left
        else:
            raise ValueError(name + " is not a valid name")

        power = np.zeros_like(n_steps_left, dtype=float)
        # define the values of power depending on n_times left, form CREST
        power[current_time <= 8] = 73  # Start-up and fill
        # Heating
        power[np.logical_and(current_time > 8, current_time <= 29)] = 2056
        power[
            np.logical_and(current_time > 29, current_time <= 81)
        ] = 73  # Wash and drain
        power[
            np.logical_and(current_time > 81, current_time <= 92)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 92, current_time <= 94)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 94, current_time <= 105)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 105, current_time <= 107)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 107, current_time <= 118)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 118, current_time <= 120)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 120, current_time <= 131)
        ] = 73  # Spin
        power[
            np.logical_and(current_time > 131, current_time <= 133)
        ] = 250  # Rinse
        power[
            np.logical_and(current_time > 133, current_time <= 138)
        ] = 568  # Fast spin
        power[
            np.logical_and(current_time > 138, current_time <= 198)
        ] = 2500  # Drying cycle

        # standby
        power[n_steps_left == 0] = self.appliances["standby power"][
            self.appliances["name"] == name
        ]

        return power


def CRESTDATA_get_clearness_TPM(return_labels=False):
    path = (
        OLD_DATASET_PATH
        + os.sep
        + "CREST_data"
        + os.sep
        + "CREST_Demand_Model_v2.3.3.xlsm-ClearnessIndexTPM.csv"
    )
    df = pd.read_csv(path, header=8, nrows=101, usecols=np.arange(2, 2 + 101))
    if return_labels:
        return df.to_numpy(), np.array(df.columns)
    else:
        return df.to_numpy()



def CRESTDATA_get_lighting_bulbs():
    path = (
        OLD_DATASET_PATH
        + os.sep
        + "CREST_data"
        + os.sep
        + "CREST_Demand_Model_v2.3.3.xlsm - bulbs.csv"
    )
    df = pd.read_csv(path, header=9, nrows=100, usecols=np.arange(1, 1 + 39))
    return df.to_numpy()


class CRESTLightingSimulator(Simulator):
    def __init__(
        self,
        n_households,
        data="CREST",
        initialization_method="off",
        logger=None,
    ):
        super().__init__(n_households, logger=logger)
        super().initialize_starting_state()

        self.bulbs_power = self.get_bulbs_configuration(data=data)
        self.bulbs_times_left = self.initialize_bulbs(initialization_method)

        self.irradiance_threshold = self.generate_irradiance_threshold(
            data=data
        )

        self.calibration_scalar = self.get_calibaration_scalar(data=data)
        self.relative_use = self.generate_relative_use(data=data)

        self.effective_occupancy_array = self.get_effective_occupancy_array(
            data=data
        )

        (
            self.durations_cdf,
            self.durations_low,
            self.durations_high,
        ) = self.get_durations_cdf(data=data)

    def GTOU_sample_bulbs_config(self):
        path = OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
        df = pd.read_excel(path, header=75, sheet_name="Lighting")
        bulbs_penetration = df["Penetration"].to_numpy()
        bulbs_consumption = df["Consumption[W]"].to_numpy()
        assert len(bulbs_penetration) == len(
            bulbs_consumption
        ), "penetration and conumption in Lighting input.ods size must match"
        df = pd.read_excel(path, header=68, sheet_name="Lighting")
        mean_n = df["Unnamed: 3"][0]
        std_n = df["Unnamed: 3"][1]
        # sample the number of lights
        n_bulbs = np.array(
            mean_n + std_n * np.random.randn(self.n_households), dtype=int
        )

        bulbs_power = np.zeros((self.n_households, max(n_bulbs)))
        for i, n in enumerate(n_bulbs):
            # ensure the bulb number is at least one
            n = max(n, 1)
            bulbs_power[i, :n] = bulbs_consumption[
                monte_carlo_from_1d_pdf(bulbs_penetration, n_samples=n)
            ]
        return bulbs_power

    def initialize_bulbs(self, method):

        if method == "off":
            # crest data is initialized by having all lights off, times left = 0
            return self.bulbs_power * 0
        else:
            raise ValueError(
                'Unkown initialization method for lighting simulator. Available: "off" '
            )

    def get_calibaration_scalar(self, data):
        if data == "CREST":
            return 0.008153686396677
        elif data == "GTOU":
            path = (
                OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
            )
            df = pd.read_excel(path, header=23, sheet_name="Lighting")
            return df.columns[5]
        else:
            raise ValueError("Unknonw data :" + data)

    def get_bulbs_configuration(self, data="CREST"):
        if data == "CREST":
            bulbs_configurations = CRESTDATA_get_lighting_bulbs()
        elif data == "GTOU":
            return self.GTOU_sample_bulbs_config()

        else:
            raise ValueError("Unknonw data :" + data)
        # get a random bulb configuration for each households
        r = np.random.randint(
            0, len(bulbs_configurations), size=self.n_households
        )
        return bulbs_configurations[r]

    def generate_irradiance_threshold(self, data="CREST"):
        if data == "CREST":
            mean = 60
            std = 10
        elif data == "GTOU":
            path = (
                OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
            )
            df = pd.read_excel(path, header=3, sheet_name="Lighting")
            mean = df.columns[5]
            std = df.columns[6]
        else:
            raise ValueError("Unknonw data :" + data)
        return np.random.normal(loc=mean, scale=std, size=self.n_households)

    def get_effective_occupancy_array(self, data="CREST"):
        if data == "CREST":
            return np.array([0.000, 1.000, 1.528, 1.694, 1.983, 2.094])
        elif data == "GTOU":
            path = (
                OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
            )
            df = pd.read_excel(path, header=35, sheet_name="Lighting")
            return np.array(df["occupancy"][:6], dtype=float)
        else:
            raise ValueError("Unknonw data :" + data)

    def active_to_effective(self, active_occupancy):
        active_occupancy = np.array(active_occupancy, dtype=int)
        return self.effective_occupancy_array[active_occupancy]

    def generate_relative_use(self, data="CREST"):
        if data == "CREST" or data == "GTOU":
            r = np.random.uniform(size=(self.bulbs_power.shape))
            # gets a exponential distribution
            return -np.log(r)
        else:
            raise ValueError("Unknonw data :" + data)

    def get_durations_cdf(self, data="CREST"):
        if data == "CREST":
            cdf = np.array(
                [
                    0.111111111111111,
                    0.222222222222222,
                    0.333333333333333,
                    0.444444444444444,
                    0.555555555555556,
                    0.666666666666667,
                    0.777777777777778,
                    0.888888888888889,
                    1,
                ]
            )
            times_low = np.array([1, 2, 3, 5, 9, 17, 28, 50, 92], dtype=int)
            times_high = np.array([1, 2, 4, 8, 16, 27, 49, 91, 259], dtype=int)

        elif data == "GTOU":
            path = (
                OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
            )
            df = pd.read_excel(path, header=53, sheet_name="Lighting")
            times_low = np.array(df["(minutes)"][:9], dtype=float)
            times_high = np.array(df["(minutes).1"][:9], dtype=float)
            cdf = np.array(df["probability"][:9], dtype=float)
        else:
            raise ValueError("Unknonw data :" + data)
        return cdf, times_low, times_high

    def draw_random_durations(self, n_samples):
        # get the duration interval indexes
        e = monte_carlo_from_1d_cdf(self.durations_cdf, n_samples=n_samples)

        # sample uniformly in the interval
        r = np.random.uniform(size=n_samples)
        return self.durations_low[e] + r * (
            self.durations_high[e] - self.durations_low[e]
        )

    def get_electric_consumptions(self):
        return np.nansum(
            self.bulbs_power * (self.bulbs_times_left > 0), axis=1
        )

    def get_power_demand(self):
        return self.get_electric_consumptions()

    def get_thermal_gains(self):
        return self.get_electric_consumptions()

    def step(self, active_occupancy, irradiance):
        super().step()

        assert (
            len(active_occupancy) == self.n_households
        ), "The active occupancy must be given for each households"
        # update the times left
        self.bulbs_times_left[self.bulbs_times_left > 0] -= 1

        # Get the effective occupancy for this number of active occupants to allow for sharing
        effective_occupancy = self.active_to_effective(active_occupancy)

        bulbs_shape = self.bulbs_power.shape
        # There is a 5% chance of switch on event if the irradiance is above the threshold

        mask_low_irradiance = np.logical_or(
            (irradiance < self.irradiance_threshold)[:, None],
            np.random.uniform(size=bulbs_shape) < 0.05,
        )

        # Get the bulbs that are available at this time to be switched on
        indices_households, indices_bulb = np.where(
            np.logical_and(mask_low_irradiance, self.bulbs_times_left == 0)
        )
        # compute the probabilities of switch on for each bulb
        probs_switch_on = (
            effective_occupancy[indices_households]
            * self.relative_use[indices_households, indices_bulb]
            * self.calibration_scalar
        )

        mask_switch_on = (
            np.random.uniform(size=len(indices_bulb)) < probs_switch_on
        )

        # generate and assigne the times of the switch on events
        self.bulbs_times_left[
            indices_households[mask_switch_on], indices_bulb[mask_switch_on]
        ] = self.draw_random_durations(n_samples=mask_switch_on.sum())

        # If there are no active occupants, turn off the light
        self.bulbs_times_left[active_occupancy == 0, :] = 0



class HeatingControlsSimulator(Simulator):
    def CRESTDATA_read_controls(self):
        path = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.3.3.xlsm - HeatingControls.csv"
        )
        df_home = pd.read_csv(path, header=3, nrows=15)
        df_water = pd.read_csv(path, header=22, nrows=12)

        thermostat = {}
        thermostat["home temperatures cdf"] = np.cumsum(
            np.array(df_home["Percentage of homes"])
        )
        thermostat["home temperatures values"] = np.array(
            df_home["Demand temperature"], dtype=float
        )
        thermostat["water temperatures cdf"] = np.cumsum(
            np.array(df_water["Percentage of homes"])
        )
        thermostat["water temperatures values"] = np.array(
            df_water["Hot water delivery temperature"], dtype=float
        )

        path = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.3.3.xlsm - HeatingControlsTPM.csv"
        )
        df_heating = pd.read_csv(
            path, header=6, nrows=96, usecols=[2, 3, 4, 5]
        )
        thermostat["transitions cdf wd"] = np.cumsum(
            df_heating.to_numpy()[:, :2].reshape((48, 2, 2)), axis=-1
        )
        thermostat["transitions cdf we"] = np.cumsum(
            df_heating.to_numpy()[:, 2:].reshape((48, 2, 2)), axis=-1
        )

        # crest value for the emitters
        thermostat["emitter setpoints"] = 50.0

        # Set thermostat deadbands
        deadbands = {}
        deadbands["space heating"] = 2
        deadbands["SpaceCooling"] = 2
        deadbands["hot water"] = 5
        deadbands["emitter"] = 5
        thermostat["deadband"] = deadbands

        return thermostat

    def GERMANY_read_controls(self):
        raise NotImplementedError("must define how the controls will work")

    def __init__(
        self,
        n_households,
        date_time,
        has_heating_systems,
        is_combi_boiler,
        temperatures,
        data="CREST",
    ):
        super().__init__(n_households)
        self.date_time = date_time

        self.thermostat = self.read_controls(data=data)

        self.assign_controls_to_households(
            has_heating_systems, is_combi_boiler
        )

        self.initialize_heating_controls(temperatures)

    def read_controls(self, data):
        if data == "CREST":
            return self.CRESTDATA_read_controls()
        elif data == "GTOU":
            return self.GERMANY_read_controls()
        else:
            raise ValueError("Unknonw data :" + data)

    def assign_controls_to_households(
        self, has_heating_systems, is_combi_boiler
    ):
        # the array of the household that have heating systems

        # determine space heating thermostat set point
        r = monte_carlo_from_1d_cdf(
            self.thermostat["home temperatures cdf"],
            n_samples=self.n_households,
        )
        self.space_heating_setpoints = self.thermostat[
            "home temperatures values"
        ][r]
        # Only use space heating if there is a heating system
        # Set space heating thermostat to -inf so heating is never used
        self.space_heating_setpoints[~has_heating_systems] = -np.inf

        # determine hot water thermostat set point
        r = monte_carlo_from_1d_cdf(
            self.thermostat["water temperatures cdf"],
            n_samples=self.n_households,
        )
        self.hot_water_setpoints = self.thermostat[
            "water temperatures values"
        ][r]

        self.emitter_setpoints = self.thermostat["emitter setpoints"]

        # assign a random delay to the timers
        self.delays = np.random.randint(-15, 15, size=self.n_households)

        # store the combi boilers
        assert len(is_combi_boiler) == self.n_households
        self.is_combi_boiler = is_combi_boiler

    def change_space_heating_setpoints(self, new_temperatures):

        new_setpoints = np.array(new_temperatures)
        assert len(new_setpoints) == len(self.space_heating_setpoints)
        self.space_heating_setpoints = new_setpoints

    def initialize_heating_controls(self, temperatures):

        thermostat_states = {}
        # Determine initial thermostat states
        thermostat_states["hot water"] = (
            temperatures["cylinder"] < self.hot_water_setpoints
        )
        thermostat_states["space heating"] = (
            temperatures["interior"] < self.space_heating_setpoints
        )
        thermostat_states["emitter"] = (
            temperatures["emitter"] < self.emitter_setpoints
        )

        self.thermostat_states = thermostat_states

        # Determine the heating timer settings for this dwelling based on transition probabilities
        # find which starting probs to use, CREST
        prob_on = 0.09 if self.date_time.isoweekday() <= 5 else 0.10
        cdf = np.array([prob_on, 1.0 - prob_on])
        self.space_heating_timers_states = monte_carlo_from_1d_cdf(
            cdf, n_samples=self.n_households
        )

        # define an iterator for the cdf of the tpm
        self._cdf_iterator = itertools.cycle(
            self.thermostat["transitions cdf wd"]
            if self.date_time.isoweekday() <= 5
            else self.thermostat["transitions cdf we"]
        )

        # NOTE: hot water timer settings will be always on, except for the first half-hour, to introduce some
        # diversity to the initial hot water heating spike
        self.hot_water_timers_states = np.zeros_like(
            self.space_heating_timers_states
        )

    def update_control_states(self):

        # update the states each half hour
        if self.date_time.minute % 30 == 0:
            self.current_cdf = next(self._cdf_iterator)

        # check the households that should be updated on this delay
        mask_updating_households = self.delays == (
            self.date_time.minute % 30 - 15
        )
        # monte carlo markov chain for the timers state
        cdf = self.current_cdf[
            self.space_heating_timers_states[mask_updating_households]
        ]
        self.space_heating_timers_states[
            mask_updating_households
        ] = monte_carlo_from_cdf(cdf)

        # NOTE: hot water timer settings will be always on, except for the first half-hour, to introduce some
        # diversity to the initial hot water heating spike
        self.hot_water_timers_states = np.ones_like(
            self.space_heating_timers_states
        )

    def calculate_control_states(self, temperatures, hot_water_demand):

        # Calculate the states of the thermostats and timers
        # Calculate the thermostat states for current time step, based partly on states in previous time step

        # Hot water thermostat
        self.thermostat_states["hot water"] = np.where(
            np.logical_or(
                self.thermostat_states["hot water"]
                & (
                    temperatures["cylinder"]
                    < (
                        self.hot_water_setpoints
                        + self.thermostat["deadband"]["hot water"]
                    )
                ),
                ~self.thermostat_states["hot water"]
                & (
                    temperatures["cylinder"]
                    <= (
                        self.hot_water_setpoints
                        - self.thermostat["deadband"]["hot water"]
                    )
                ),
            ),
            True,
            False,
        )
        # Room heating thermostat
        self.thermostat_states["space heating"] = np.where(
            np.logical_or(
                self.thermostat_states["space heating"]
                & (
                    temperatures["interior"]
                    < (
                        self.space_heating_setpoints
                        + self.thermostat["deadband"]["space heating"]
                    )
                ),
                ~self.thermostat_states["space heating"]
                & (
                    temperatures["interior"]
                    <= (
                        self.space_heating_setpoints
                        - self.thermostat["deadband"]["space heating"]
                    )
                ),
            ),
            True,
            False,
        )

        # Emitters thermostat
        self.thermostat_states["emitter"] = np.where(
            np.logical_or(
                self.thermostat_states["emitter"]
                & (
                    temperatures["emitter"]
                    < (
                        self.emitter_setpoints
                        + self.thermostat["deadband"]["emitter"]
                    )
                ),
                ~self.thermostat_states["emitter"]
                & (
                    temperatures["emitter"]
                    <= (
                        self.emitter_setpoints
                        - self.thermostat["deadband"]["emitter"]
                    )
                ),
            ),
            True,
            False,
        )

        # Determine whether hot water heating is required
        heat_water_onoff = np.where(
            self.is_combi_boiler,
            hot_water_demand
            > 0,  # If it's a combi system then hot water control signal is determined by hot water demand
            self.hot_water_timers_states
            & self.thermostat_states[
                "hot water"
            ],  # otherwise for regular or system boilers it depends on the timer and thermostat states
        )
        # if combi,  override the timer state so that it is always on
        self.hot_water_timers_states[self.is_combi_boiler] = True

        # space heating requires three conditions to be on
        space_heating_onoff = (
            self.space_heating_timers_states
            * self.thermostat_states["space heating"]
            * self.thermostat_states["emitter"]
        )

        # Determine with the heating system should be switched on if either the hot water is needed
        # or if the space heating is needed
        heater_onoff = heat_water_onoff | space_heating_onoff

        heating_controls = {}
        heating_controls["heater on"] = heater_onoff
        heating_controls["heat water on"] = heat_water_onoff
        heating_controls["space heating on"] = space_heating_onoff
        return heating_controls


class HeatingSystemSimulator(Simulator):
    def CRESTDATA_read_buildings(self):
        path = (
            OLD_DATASET_PATH
            + os.sep
            + "CREST_data"
            + os.sep
            + "CREST_Demand_Model_v2.3.3.xlsm - Buildings.csv"
        )
        df = pd.read_csv(path, header=1, nrows=8, skiprows=[2, 3])

        buildings = {}
        buildings["equipped probs"] = np.array(
            df["Proportion of dwellings of this building type"]
        )
        buildings["name"] = np.array(df["Description"])
        buildings["outside/exterior transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient between outside air and external building thermal capacitance"
            ]
        )
        buildings["outside/interior transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance"
            ]
        )
        buildings["ventilation transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient representing ventilation heat loss between outside air and internal building thermal capacitance"
            ]
        )
        buildings["external capacitance"] = np.array(
            df["External building thermal capacitance"]
        )
        buildings["internal capacitance"] = np.array(
            df["Internal building thermal capacitance"]
        )
        buildings["irradiance multiplier"] = np.array(
            df["Global irradiance multiplier"]
        )
        buildings["ventilation rate"] = np.array(
            df["Ventilation rate, air changes per hour"]
        )
        buildings["floor area"] = np.array(df["Floor area, living space"])
        buildings["height"] = np.array(df["Height, living space"])

        emitters = {}
        emitters["nominal temperature"] = np.array(
            df["Nominal temperature of emitters"]
        )
        emitters["transfer coefficient"] = np.array(
            df["Heat transfer coefficient of heat emitters"]
        )
        emitters["water mass"] = np.array(df["Mass of water in heat emitters"])
        emitters["capacitance"] = np.array(
            df["Thermal capacitance of heat emitters"]
        )

        return buildings, emitters

    def GERMANY_read_buildings(self):
        path = OLD_DATASET_PATH + os.sep + "GermanTOU" + os.sep + "inputs.ods"
        df = pd.read_excel(
            path, header=1, skiprows=[2, 3], sheet_name="Buildings"
        )

        buildings = {}
        buildings["equipped probs"] = np.array(
            df["Proportion of dwellings of this building type"]
        )
        buildings["name"] = np.array(df["Description"])
        buildings["outside/exterior transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient between outside air and external building thermal capacitance"
            ]
        )
        buildings["outside/interior transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance"
            ]
        )
        buildings["ventilation transfer coefficient"] = np.array(
            df[
                "Thermal transfer coefficient representing ventilation heat loss between outside air and internal building thermal capacitance"
            ]
        )
        buildings["external capacitance"] = np.array(
            df["External building thermal capacitance"]
        )
        buildings["internal capacitance"] = np.array(
            df["Internal building thermal capacitance"]
        )
        buildings["irradiance multiplier"] = np.array(
            df["Global irradiance multiplier"]
        )
        buildings["ventilation rate"] = np.array(
            df["Ventilation rate, air changes per hour"]
        )
        buildings["floor area"] = np.array(df["Floor area, living space"])
        buildings["height"] = np.array(df["Height, living space"])

        emitters = {}
        emitters["nominal temperature"] = np.array(
            df["Nominal temperature of emitters"]
        )
        emitters["transfer coefficient"] = np.array(
            df["Heat transfer coefficient of heat emitters"]
        )
        emitters["water mass"] = np.array(df["Mass of water in heat emitters"])
        emitters["capacitance"] = np.array(
            df["Thermal capacitance of heat emitters"]
        )

        return buildings, emitters



    def read_buildings(self, data):
        if data == "CREST":
            return self.CRESTDATA_read_buildings()
        elif data == "GTOU":
            return self.GERMANY_read_buildings()
        else:
            raise ValueError("Unknonw data :" + data)

    def __init__(self, n_households, outside_temperature, data="CREST"):
        super().__init__(n_households)

        self.heating_systems = self.read_heating_system(data=data)
        self.buildings, self.emitters = self.read_buildings(data=data)

        self.assign_heating_systems_to_households()
        self.assign_building_to_households()

        self.initialize_temperatures(outside_temperature)

    def assign_heating_systems_to_households(self):
        cdf = np.cumsum(self.heating_systems["equipped probs"])
        try:
            check_valid_cdf(cdf)
        except Exception as e:
            raise Exception(
                "Incorrect probabilities distribuiton for the heating systems equippememnt probabilities"
            )
        self.heating_types = monte_carlo_from_1d_cdf(
            cdf, n_samples=self.n_households
        )

        self.dblP_pump = self.heating_systems["pump power"][self.heating_types]
        self.dblP_standby = self.heating_systems["standby power"][
            self.heating_types
        ]

        self.dblFuelFlowRate = self.heating_systems["fuel flow rate"][
            self.heating_types
        ]
        self.fuel_types = self.heating_systems["fuel type"][self.heating_types]

        # Heat output of unit
        self.dblPhi_h = self.heating_systems["heat output [W]"][
            self.heating_types
        ]

        self.h_loss = self.heating_systems["thermal resistance loss"][
            self.heating_types
        ]
        SPECIFIC_HEAT_CAPACITY_WATER = 4200.0
        self.C_cyl = (
            self.heating_systems["cylinder volume"][self.heating_types]
            * SPECIFIC_HEAT_CAPACITY_WATER
        )

        # for CREST, defined naively
        self.is_combi_boiler = (
            self.heating_systems["regular/combi"][self.heating_types] == 2
        )
        self.has_heating_system = (
            self.heating_systems["regular/combi"][self.heating_types] != 4
        )

    def assign_building_to_households(self):
        cdf = np.cumsum(self.buildings["equipped probs"])
        try:
            check_valid_cdf(cdf)
        except Exception as e:
            raise Exception(
                "Incorrect probabilities distribuiton for the heating systems equippememnt probabilities"
            )
        self.building_types = monte_carlo_from_1d_cdf(
            cdf, n_samples=self.n_households
        )

        self.emitters_nominal_temperature = self.emitters[
            "nominal temperature"
        ][self.building_types]
        self.dblC_em = self.emitters["capacitance"][self.building_types]
        self.dblH_em = self.emitters["transfer coefficient"][
            self.building_types
        ]

        self.dblC_b = self.buildings["external capacitance"][
            self.building_types
        ]
        self.dblC_i = self.buildings["internal capacitance"][
            self.building_types
        ]
        # Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance (W/K)
        self.dblH_bi = self.buildings["outside/interior transfer coefficient"][
            self.building_types
        ]
        # Thermal transfer coefficient between outside air and external building thermal capacitance (W/K)
        self.dblH_ob = self.buildings["outside/exterior transfer coefficient"][
            self.building_types
        ]
        # Thermal transfer coefficient representing ventilation losses between outside air and internal building thermal capacitance (W/K)
        self.dblH_v = self.buildings["ventilation transfer coefficient"][
            self.building_types
        ]
        self.building_area = self.buildings["floor area"][self.building_types]
        self.irradiance_multiplier = self.buildings["irradiance multiplier"][
            self.building_types
        ]

    def get_target_heat_output_hot_water(
        self, hot_water_heat_demand, hot_water_set_points, increment
    ):
        # Get the hot water thermostat set point for this dwelling

        # Calculate target heat input required from heating system to deliver appropriate
        # temperature of hot water
        cylinder_part = (
            self.C_cyl
            / increment.total_seconds()
            * (hot_water_set_points - self.temperatures["cylinder"])
        )
        hot_water_part = hot_water_heat_demand * (
            self.temperatures["cylinder"] - self.temperatures["cold water"]
        )
        loss_part = self.h_loss * (
            self.temperatures["cylinder"] - self.temperatures["interior"]
        )
        return cylinder_part + hot_water_part + loss_part

    def get_target_heat_output_space_heating(self, increment):
        #  Temperature deadband for emitters
        dblEmitterDeadband = 5
        target_emitter_temperature = (
            self.emitters_nominal_temperature + dblEmitterDeadband
        )
        # calculate the target heat delivery to heat emitters to achieve the set point
        dblPhi_hSpaceTarget = self.dblC_em / increment.total_seconds() * (
            target_emitter_temperature - self.temperatures["emitter"]
        ) + self.dblH_em * (
            self.temperatures["emitter"] - self.temperatures["interior"]
        )
        return dblPhi_hSpaceTarget

    def get_pumps_power_consumptions(self, heating_controls):
        # the heating system pump will operate if space heating timer and thermostat states are on
        # and even if emitter return thermostat state is off
        aP_h = np.where(
            heating_controls["space_thermostat"]
            & heating_controls["space_timer"],
            self.dblP_pump,
            self.dblP_standby,
        )
        # if the heater is on (for hot water of for emitters)
        aP_h = np.where(heating_controls["heater on"], self.dblP_pump, aP_h)
        return aP_h

    def get_fuel_consumptions(self, heating_controls):
        out_dict = {}
        # check where the heating is occuring
        heating_occurs = heating_controls["heater on"] & (self.dblPhi_h > 0)
        for h_type in np.unique(self.fuel_types):
            if h_type == "Mains gas":

                out_dict[h_type] = np.where(
                    (self.fuel_types == h_type) & heating_occurs,
                    self.dblFuelFlowRate * self.dblPhi_hTotal / self.dblPhi_h,
                    0,
                )
            elif h_type == "Electricity":

                out_dict[h_type] = np.where(
                    (self.fuel_types == h_type) & heating_occurs,
                    self.dblFuelFlowRate
                    * 1000
                    * self.dblPhi_hTotal
                    / self.dblPhi_h,
                    0,
                )
            else:
                assert isinstance(h_type, str), "fuel types must be strings"
                raise ValueError("Type of fuel not recognized : " + h_type)
        return out_dict

    def update_heat_output(
        self,
        heating_controls,
        hot_water_heat_demand,
        hot_water_set_points,
        increment=datetime.timedelta(minutes=1),
    ):
        #  Get the control signals from the heating controller

        mask_HeaterOnOff = heating_controls["heater on"]
        mask_HeatWaterOnOff = heating_controls["heat water on"]
        mask_SpaceHeatingOnOff = heating_controls["space heating on"]

        # first compute the temperatures targets

        # get the target heat output for space heating
        dblPhi_hSpaceTarget = self.get_target_heat_output_space_heating(
            increment
        )
        #  get the target heat output required for the hot water
        dblPhi_hWaterTarget = self.get_target_heat_output_hot_water(
            hot_water_heat_demand, hot_water_set_points, increment
        )

        #  assign heat to hot water, bound by max and min values
        dblPhi_hWater = np.clip(dblPhi_hWaterTarget, 0, self.dblPhi_h)
        self.aPhi_hWater = np.where(
            mask_HeaterOnOff & mask_HeatWaterOnOff, dblPhi_hWater, 0
        )
        # assign remaining required capacity to space heating
        dblPhi_hSpace = np.clip(
            dblPhi_hSpaceTarget, 0, self.dblPhi_h - dblPhi_hWater
        )
        self.aPhi_hSpace = np.where(
            mask_HeaterOnOff & mask_HeatWaterOnOff & mask_SpaceHeatingOnOff,
            dblPhi_hSpace,
            0,
        )

        # otherwise heat is required only for space heating
        # assign heat to space heating, bound by max and min values
        dblPhi_hSpace = np.clip(dblPhi_hSpaceTarget, 0, self.dblPhi_h)
        self.aPhi_hSpace = np.where(
            mask_HeaterOnOff & (~mask_HeatWaterOnOff),
            dblPhi_hSpace,
            self.aPhi_hSpace,
        )

        # assign remaining heating system variables
        self.dblPhi_hTotal = self.aPhi_hSpace + self.aPhi_hWater

    def initialize_temperatures(self, outside_temperature):
        self.temperatures = {}
        # Assign initial building temperatures
        # In hot climates or warm weather the initial building temperatures are likely to
        # be increased external temperature
        self.temperatures["building"] = 2 * np.random.uniform(
            size=self.n_households
        ) + max(16, outside_temperature)
        self.temperatures["interior"] = 2 * np.random.uniform(
            size=self.n_households
        ) + np.clip(outside_temperature, 16, 25)
        self.temperatures["emitter"] = np.array(self.temperatures["interior"])
        self.temperatures["cooling"] = np.array(self.temperatures["interior"])

        # Set the initial temperature of the hot water tank
        self.temperatures["cylinder"] = 60.0 + 2 * np.random.uniform(
            size=self.n_households
        )

        # Set cold water inlet temperature
        self.temperatures["cold water"] = 10.0

    def update_temperatures(
        self,
        temperature_outside,
        irradiance,
        hot_water_demand,
        occupancy_thermal_gains,
        lighting_thermal_gains,
        appliances_thermal_gains,
        increment,
    ):
        # Get or calculate the thermal gains

        # ... from primary heating system to space
        dblPhi_hSpace = self.aPhi_hSpace
        # ... from cooling system to space

        # ... from primary heating system to hot water
        dblPhi_hWater = self.aPhi_hWater

        # ... from passive solar gains
        dblPhi_s = irradiance * self.irradiance_multiplier

        # ... from occupants, lighting and appliances
        dblPhi_cOccupancy = occupancy_thermal_gains
        dblPhi_cLighting = lighting_thermal_gains
        dblPhi_cAppliances = appliances_thermal_gains

        dblPhi_c = dblPhi_cOccupancy + dblPhi_cLighting + dblPhi_cAppliances

        # ... from solar thermal collector (if any)
        # dblPhi_collector = aSolarThermal(intRunNumber).GetPhi_s(currentTimeStep)

        # Get the variable hot water demand heat transfer coefficient
        dblH_dhw = hot_water_demand

        # Calculate change in building external thermal node temperatures
        dblDeltaTheta_b = (increment.total_seconds() / self.dblC_b) * (
            -(self.dblH_ob + self.dblH_bi) * self.temperatures["building"]
            + self.dblH_bi * self.temperatures["interior"]
            + self.dblH_ob * temperature_outside
        )

        # Calculate change in building internal thermal node temperatures
        dblDeltaTheta_i = (increment.total_seconds() / self.dblC_i) * (
            self.dblH_bi * self.temperatures["building"]
            - (self.dblH_v + self.dblH_bi + self.dblH_em + self.h_loss)
            * self.temperatures["interior"]
            + self.dblH_v * temperature_outside
            + self.dblH_em * self.temperatures["emitter"]
            + self.h_loss * self.temperatures["cylinder"]
            + dblPhi_s
            + dblPhi_c
        )

        # Calculate change in heat emitter temperatures (heating radiators only)
        dblDeltaTheta_em = (increment.total_seconds() / self.dblC_em) * (
            self.dblH_em * self.temperatures["interior"]
            - self.dblH_em * self.temperatures["emitter"]
            + dblPhi_hSpace
        )

        # Calculate change in temperature of hot water cylinder
        dblDeltaTheta_cyl = (increment.total_seconds() / self.C_cyl) * (
            self.h_loss * self.temperatures["interior"]
            - (self.h_loss + dblH_dhw) * self.temperatures["cylinder"]
            + dblH_dhw * self.temperatures["cold water"]
            + dblPhi_hWater  # + dblPhi_collector
        )

        # Update building thermal node temperatures for this time step
        self.temperatures["building"] += dblDeltaTheta_b
        self.temperatures["interior"] += dblDeltaTheta_i
        self.temperatures["emitter"] += dblDeltaTheta_em
        self.temperatures["cylinder"] += dblDeltaTheta_cyl
