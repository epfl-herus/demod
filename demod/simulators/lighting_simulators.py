
import numpy as np
import pandas as pd
import os
import itertools

from .base_simulators import  Simulator
from ..utils.monte_carlo import monte_carlo_from_1d_cdf, monte_carlo_from_1d_pdf, monte_carlo_from_cdf
from ..utils.error_messages import ALGO_REQUIRES_LOADING_METHOD



class FisherLightingSimulator(Simulator):
    """Lighting simulator as described by [Fisher2015]_.

    Simulates the electric consumption of lighting using a simple
    formula: \n
    :math:`P_{el, l}(t) =n_{active} \cdot P_{el, l, pp}(t) \cdot
    \\frac{I_{g, max }-I_{g}(t)}{I_{g, max }-I_{g, min }}`
    \n
    where :math:`n_{active}` is the
    :py:attr:`~demod.utils.cards_doc.Inputs.active_occupancy`
    , :math:`P_{el, l, pp}(t)` is a constant representing the Power used
    by a single person.
    :math:`I_{g, max}` and :math:`I_{g, min}` are the irradiance
    thresholds for lighting demand and :math:`I_{g}(t)` is the current
    :py:attr:`~demod.utils.cards_doc.Inputs.irradiance`


    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
        :py:attr:`~demod.utils.cards_doc.Params.initial_active_occupancy`
        :py:attr:`~demod.utils.cards_doc.Params.initial_irradiance`
    Data
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_fisher_lighting`
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.active_occupancy`
        :py:attr:`~demod.utils.cards_doc.Inputs.irradiance`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_energy_consumption`
    Step size
        Any.
    """

    def __init__(
            self,
            n_households: int, data='Germany',
            initial_active_occupancy: np.ndarray = None,
            initial_irradiance: float = None,
            **kwargs,
        ) -> None:
        """Initialize a Fisher lighting simulator.

        Args:
            n_households:
            data: Defaults to 'Germany'.
            initial_active_occupancy: Defaults to None.
            initial_irradiance: Defaults to None.
        """
        if data == 'Germany':
            from demod.datasets.Germany.loader import GermanDataHerus
            data = GermanDataHerus()
        super().__init__(n_households, **kwargs)

        lighting_dict = data.load_fisher_lighting()

        self.global_irradiation_min_switch = lighting_dict[
            'irradiation_threshold_min'
        ]
        self.global_irradiation_max_switch = lighting_dict[
            'irradiation_threshold_max'
        ]
        self.individual_light_use = lighting_dict[
            'individual_light_use'
        ]

        self.initialize_starting_state(
            initial_active_occupancy,
            initial_irradiance
        )

    def initialize_starting_state(
        self,
        initial_active_occupancy,
        initial_irradiance
    ) -> None:
        """Initialize the two parameters for the simulator.

        Args:
            initial_active_occupancy: array of initiall active occ
            initial_irradiance: value of initial irradiance
        """
        self.active_occupancy = (
            np.ones_like(self.n_households)
            if initial_active_occupancy is None else initial_active_occupancy
        )
        self.irradiance = initial_irradiance or 0

        return super().initialize_starting_state()

    def calculate_light_usage_value(self, irradiance: float) -> float:
        """Calculate the light usage value as defined in Fisher

        Args:
            irradiance: The current irradiance value

        Returns:
            light_usage: value between 0 and 1
        """
        light_usage = np.clip(
            (self.global_irradiation_max_switch - irradiance)
            / (
                self.global_irradiation_max_switch
                - self.global_irradiation_min_switch
            ), 0, 1)
        return light_usage


    def get_power_consumption(self) -> np.ndarray:
        """Return the power consumption for each households.

        Returns:
            ndarray(float): the power consumption of the households in Watts
        """
        active_occupancy = np.array(self.active_occupancy, dtype=float)  # will ensure that we return a float
        return (
            # fisher calculation
            active_occupancy
            * self.individual_light_use
            * self.calculate_light_usage_value(self.irradiance)
        )

    def step(self, active_occupancy: np.ndarray, irradiance: float) -> None:
        """Fisher simulators step.

        Args:
            active_occupancy:
                The number of active occupants in each household
            irradiance:
                The value of the current irradiance in W/m^2
        """
        self.active_occupancy = active_occupancy
        self.irradiance = irradiance
        super().step()


class CrestLightingSimulator(Simulator):
    """Lighting simulator as proposed by CREST [Richardson2009]_ .

    Params
        :py:attr:`~demod.utils.cards_doc.Params.n_households`
        :py:attr:`~demod.utils.cards_doc.Params.data`
        :py:attr:`~demod.utils.cards_doc.Params.initialization_method`
        :py:attr:`~demod.utils.cards_doc.Params.bulbs_sampling_algo`
        :py:attr:`~demod.utils.cards_doc.Params.logger`
    Data
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_crest_lighting`
        Optionals on :py:attr:`~demod.utils.cards_doc.Params.bulbs_sampling_algo`
        (:py:meth:`~demod.datasets.base_loader.LightingLoader.load_bulbs`
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_installed_bulbs_stats`
        :py:meth:`~demod.datasets.base_loader.LightingLoader.load_bulbs_config`)
    Step input
        :py:attr:`~demod.utils.cards_doc.Inputs.active_occupancy`
        :py:attr:`~demod.utils.cards_doc.Inputs.irradiance`
    Output
        :py:meth:`~demod.utils.cards_doc.Sim.get_energy_consumption`
    Step size
        Any.
    """
    def __init__(
        self,
        n_households,
        data="CREST",
        initialization_method="off",
        bulbs_sampling_algo='config',
        **kwargs
    ):
        """Create a Crest Lighting Simulator.

        Args:
            n_households: number of households simulated.
            data: Data to be used. Defaults to "CREST".
            initialization_method: The method use for initialization.
                See :py:meth:`initialize_bulbs` for
                the possibilities. Defaults to "off".
            bulbs_sampling_algo: The algorithm to sample the bulbs
                installed in each house. See
                :py:meth:`sample_bulbs_configuration`.
                Defaults to 'randn'.
        """
        super().__init__(n_households, **kwargs)

        if data == 'CREST':
            from ..datasets.CREST.loader import Crest
            data = Crest()

        self.data = data

        # Sample some simuulators parameters
        self.bulbs_power = self.sample_bulbs_configuration(bulbs_sampling_algo)
        self.relative_use = self._generate_relative_use()
        self.irradiance_threshold = self.sample_irradiance_threshold()

        # Attribuute some variables
        crest_light_dict = data.load_crest_lighting()

        self.calibration_scalar = crest_light_dict['calibration_scalar']

        self.effective_occupancy_array = crest_light_dict[
            'effective_occupancy'
        ]

        self.durations_cdf = crest_light_dict['durations_cdf']
        self.durations_low = crest_light_dict['durations_minutes_low']
        self.durations_high = crest_light_dict['durations_minutes_high']


        self.initialize_starting_state(initialization_method)

    def initialize_starting_state(self, initialization_method) -> None:
        self.bulbs_times_left = self.initialize_bulbs(initialization_method)
        super().initialize_starting_state()

    def initialize_bulbs(self, initialization_method: str):
        """Initialize the state of the bulbs at the starting state.

        Args:
            initialization_method:
                - 'off': All the bulbs are off at the start.

        Raises:
            ValueError: Unknown initialization_method.

        Returns:
            The value of the bulbs ?
        """

        if initialization_method == "off":
            # crest data is initialized by having all lights off, times left = 0
            return np.array(self.bulbs_power * 0, int)
        else:
            raise ValueError(
                "Unkown 'initialization_method' for lighting simulator."
            )


    def sample_bulbs_configuration(
        self, bulbs_sampling_algo: str
    ) -> np.ndarray:
        """Samples the configuration of the light bulbs in the house.

        The available algorithms are:

        * 'randn':
            samples the bulbs from a random distribution, based on
            values obtained with
            :py:meth:`~demod.datasets.base_loader.LightingLoader.load_bulbs`
            and
            :py:meth:`~demod.datasets.base_loader.LightingLoader.load_installed_bulbs_stats`

        * 'config':
            samples the bulbs randomly based on diffrent configuration
            for the full house, obtained with
            :py:meth:`~demod.datasets.base_loader.LightingLoader.load_bulbs_config`

        Args:
            bulbs_sampling_algo: The algorithm to use for sampling.

        Raises:
            ValueError: If the algo is not known.

        Returns:
            light_configs, The bulbs power consumption for each household.
        """
        if bulbs_sampling_algo == "randn":
            try:
                mean, std = self.data.load_installed_bulbs_stats()
            except (AttributeError, NotImplementedError) as attr_err:
                raise NotImplementedError(
                    ALGO_REQUIRES_LOADING_METHOD.format(
                        algo=bulbs_sampling_algo,
                        simulator=type(self).__name__,
                        loading_method='load_installed_bulbs_stats',
                        dataset=self.data
                    )
                )

            consumptions, penetration = self.data.load_bulbs()

            # sample the number of lights
            n_bulbs = np.clip(
                np.array(
                    mean + std * np.random.randn(self.n_households), dtype=int
                ), 1, None)

            bulbs_power = np.zeros((self.n_households, max(n_bulbs)))
            for i, n in enumerate(n_bulbs):
                # ensure the bulb number is at least one
                n = max(n, 1)
                bulbs_power[i, :n] = consumptions[
                    monte_carlo_from_1d_pdf(penetration, n_samples=n)
                ]
            return bulbs_power

        elif bulbs_sampling_algo == "config":
            try:
                bulbs_configurations = self.data.load_bulbs_config()
            except (AttributeError, NotImplementedError) as attr_err:
                raise NotImplementedError(
                    ALGO_REQUIRES_LOADING_METHOD.format(
                        algo=bulbs_sampling_algo,
                        simulator=type(self).__name__,
                        loading_method='load_bulbs_config',
                        dataset=self.data
                    )
                )
            # get a random bulb configuration for each households
            r = np.random.randint(
                0, len(bulbs_configurations), size=self.n_households
            )
            return bulbs_configurations[r]
        else:
            raise ValueError(
                "Unknonw 'bulbs_sampling_algo': {}.".format(
                    bulbs_sampling_algo
                )
            )


    def sample_irradiance_threshold(self):
        """Sample the irradiance threshold.

        The irradiance threshold is the irradiance  at which it is
        not required to use lighting.
        It is different for all households.
        Samples from a Gaussian distribution with the given parameters
        from :py:meth:`data.load_crest_lighting`.

        Returns:
            thresholds: randomly sampled thresholds
        """
        light_dic = self.data.load_crest_lighting()
        mean = light_dic['irradiance_threshold_mean']
        std = light_dic['irradiance_threshold_std']
        return np.random.normal(loc=mean, scale=std, size=self.n_households)


    def active_to_effective(self, active_occupancy: np.ndarray):
        """Convert the active occupancy to an effective occupancy.

        Effective occupancy accounts for the sharing of light between
        the occupants.

        Args:
            active_occupancy: number of active occupants in each households.

        Returns:
            effective_occupancy: number of effective occupants corresponding
                to the active_occupancy.
        """
        active_occupancy = np.array(active_occupancy, dtype=int)
        return self.effective_occupancy_array[active_occupancy]

    def _generate_relative_use(self):
        """The orginal relative use from CREST.

        The value was infered from the original spreadsheet values.

        Returns:
            A relative use value for each bulb.
        """
        r = np.random.uniform(size=(self.bulbs_power.shape))
        return - 2.30258509299404 * np.log(r)



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

    def step(self, active_occupancy: np.ndarray, irradiance: float):
        """Perform a step of CREST lighting simuulator.

        The light usage depends on the number of active occupants in the
        house and the irradiance from outside.

        Args:
            active_occupancy: number of active occupants in each households.
            irradiance: irradiation from the sun.
        """
        super().step()

        # update the times left
        self.bulbs_times_left[self.bulbs_times_left > 0] -= 1

        # Get the effective occupancy for this number of active occupants
        # to account for sharing
        effective_occupancy = self.active_to_effective(active_occupancy)

        bulbs_shape = self.bulbs_power.shape

        # There is a 5% chance of switch on event if the irradiance is
        # above the threshold
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
        # Sample randomly which bulbs will switch on
        mask_switch_on = (
            np.random.uniform(size=len(indices_bulb)) < probs_switch_on
        )

        # Generate and assign the times of the switch on events
        self.bulbs_times_left[
            indices_households[mask_switch_on], indices_bulb[mask_switch_on]
        ] = self.draw_random_durations(n_samples=mask_switch_on.sum())

        # If there are no active occupants, turn off the light
        self.bulbs_times_left[active_occupancy == 0, :] = 0


