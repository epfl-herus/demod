"""Base simulators module.

This Module contains several base classes for running household
simulations.
The Simulators contained in this module serve as basis for new
implementations of simulations.
"""
from __future__ import annotations

import datetime
from typing import List, NoReturn, Union, Callable

import numpy as np


def cached_getter(getter: GetMethod) -> GetMethod:
    """Decorate getter methods.

    Uses a cache system to store the arrays.
    Cache is cleared at each simulator step().

    Args:
        getter: The getter method to decorate.
    """

    def decorated_getter(self: Simulator, n_ieth_hh: Union[None, int] = None):
        # add getter to the cache if not there
        if getter not in self._cache:
            self._cache[getter] = getter(self)
        # Check if a household is specified
        if n_ieth_hh is None:
            return self._cache[getter]

        return self._cache[getter][n_ieth_hh]

    return decorated_getter


class SimLogger():
    """Specialized logger for any Simulator object.

    Once the parameters are define, it can be set to a
    :py:class:`Simulator`.
    During the simulation, the :py:class:`SimLogger` will automatically
    collect the data.
    Then the :py:meth:`SimLogger.get` method can be used to access the
    collected data.

    TODO: make the logger attributes callable by the logger
    using setattr, and checking that they don't break other methods
    """

    # Some protected names that should never be transformed to numpy format
    _protected: List[str] = ['current_time']
    length: int

    def __init__(
            self,
            attributes_list: Union[str, List[str]], *args,
            aggregated: Union[bool, List[bool]] = True,
        ) -> None:
        """Create a logger.

        Args:
            attributes_list: the name of simulator method-s to be called
                by the logger
            aggregated:
                Whether the attributes should be aggregated over all the
                households.
                Defaults to True.
                If list, must match the len of attributes list.

        """
        # Checks the type of input attributes
        if isinstance(attributes_list, list):
            self.attributes_list = attributes_list.copy()
        elif isinstance(attributes_list, str):
            self.attributes_list = [attributes_list, *args]
        else:
            raise TypeError('attributes_list variable must be list')

        # checks aggregated
        if isinstance(aggregated, list):
            if len(aggregated) != len(self.attributes_list):
                len_err = ValueError(
                    'length of aggregated and attribues must match')
                raise len_err
            self.aggregated = aggregated.copy()
        elif isinstance(aggregated, bool):
            # create an array of bool matchin the attribuites
            self.aggregated = [
                False if attr in self._protected else aggregated
                for attr in self.attributes_list]
        else:
            raise TypeError('aggregated variable must be list or bool')

        dic_results = {}
        for att in self.attributes_list:
            dic_results[att] = []
        self.dic_results = dic_results

        self.length = 0

    def __add__(self, other):
        """Addition of logger merges the value of two loggers together.

        Not secure if the logger have the same attributes names.

        Args:
            other: Other logger to add.

        Raises:
            ValueError: If the logger do not have the same lengths.
            TypeError: If the other object is not a SimLogger.

        Returns:
            logger: Same logger with added the new values.
        """
        if isinstance(other, SimLogger):
            if self.length != other.length:
                raise ValueError(
                    "{} and {} have different lengths: {}, {}.".format(
                        self, other,
                        self.length, other.length
                    )
                )
            self.dic_results.update(other.dic_results)
            self.attributes_list += other.attributes_list
            self.aggregated += other.aggregated
            return self
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self).__name__, type(other).__name__
                ))

    def get(self, attribute: str = None) -> np.ndarray:
        """Get the logged values for the desired attribute.

        Args:
            attribute: The name of an attribute

        Raises:
            ValueError: If the value of the attribute is not in this logger

        Returns:
            Array with the logged values for desired attribute.
        """
        if attribute is None:
            if len(self.attributes_list) == 1:
                attribute = self.attributes_list[0]
            else:
                raise ValueError(
                    "Usage logger.get() is valid only if the logger "
                    "stores one single attributes. Not the case for logger "
                    "{} with attributes: {}.".format(
                        self, self.attributes_list
                    )
                )

        if attribute in self.attributes_list:
            out = self.dic_results[attribute]
            if isinstance(out, dict):
                # for dictionary attributes, convert each value of the
                # dict
                return {
                    # Remove last point to compensate intialization record
                    key: np.asarray(val)[:-1] for key, val in out.items()
                }
            # Remove last point to compensate intialization record
            return np.asarray(out)[:-1]
        else:
            self._raise_unkown_attribute(attribute)

    def clear(self, attribute: Union[List, str] = None) -> None:
        """Clear the requested attribute from the logger.

        If attribute
        not specified, clears all attributes.

        Args:
            attribute: The attribute to clear. Defaults to None.

        Raises:
            ValueError: If the attribute is not recognized.
        """
        if isinstance(attribute, list):
            # recursively clear a list input of attributes
            for att in attribute:
                self.clear(att)
        elif attribute in self.attributes_list:
            # clear ponctually
            self.dic_results[attribute] = []
        elif attribute is None:
            # clear all
            for att in self.attributes_list:
                self.dic_results[att] = []
        else:
            self._raise_unkown_attribute(attribute)

        self.length = 0

    def copy(self) -> SimLogger:
        """Create a copy of the cuurrent logger.

        Only copy the
        attributes, not the collected data.

        Returns:
            An empty logger with the same attributes.
        """
        new_logger = SimLogger(self.attributes_list.copy(), aggregated=self.aggregated)
        new_logger.length = self.length
        return new_logger

    def visit_simulator(self:SimLogger, sim:Simulator) -> None:
        """Visit a simulator object

        Args:
            sim: The simulator that is visited. The SimLogger will
                request and store the attributes registered in the
                logger.

        Note:
            You usually won't need to use this method, as it is called
            in :py:meth:`Simulator.step`
        """
        for attribute, aggregate in zip(self.attributes_list, self.aggregated):
            result = getattr(sim, attribute)
            # if it is a callable, call it
            if callable(result):
                result = result()

            # Aggregates the result over the households
            if aggregate:
                result = (
                    {key: np.sum(val) for key, val in result.items()}
                    if isinstance(result, dict)
                    else np.sum(result)
                )

            # Some attributes are not to be stored as numpy arrays
            if attribute in self._protected:
                self.dic_results[attribute].append(result)

            elif isinstance(result, dict): # dictionary outputs
                if not isinstance(self.dic_results[attribute], dict):
                    # For the first step
                    self.dic_results[attribute] = {
                        key: [] for key in result.keys()
                    }

                for key, value in result.items():
                    self.dic_results[attribute][key].append(
                        np.array(value)
                    )
            else: # standard inputs
                self.dic_results[attribute].append(np.array(result))

        self.length += 1

    def print(self):
        """Print out all the attributes from the logger."""
        for attr in self.attributes_list:
            print(attr)
            print(self.get(attr))

    def plot_column(self, aggregate: bool = False):
        """Plot the logger values in a plot in column.

        If 'current_time' was specified in the attributes,
        it will be used as x axis for all the subplots.

        Args:
            aggregate: Wheter to aggregate the values. Defaults to False.
        """
        import matplotlib.pyplot as plt
        n_plots = len(self.attributes_list)
        if 'current_time' in self.attributes_list:
            n_plots -= 1
            x_ticks = self.get('current_time')
        else:
            x_ticks = np.arange(0, self.length)

        fig, axes = plt.subplots(
            nrows=n_plots,
            sharex=True
        )
        plt.subplots_adjust(hspace=0)


        # Don't touch original logger data, so use a copy.
        attributes_to_plot = self.attributes_list.copy()
        aggregated = self.aggregated.copy()

        try:  # Remove the current time, as used as x axis.
            ind = attributes_to_plot.index('current_time')
            aggregated.pop(ind)
            attributes_to_plot.pop(ind)
        except ValueError as val_err:
            pass

        # Iterates over the attributes and plots
        for ax, attr, aggr in zip(axes, attributes_to_plot, aggregated):
            serie = self.get(attr)

            # Splits the arrays of the dictionary
            if isinstance(serie, dict):
                for key, ser in serie.items():
                    # add the dictionary part of the name
                    lab = attr + "['{}']".format(key)
                    if not aggr and (len(ser.shape) > 1):
                        # Case many households
                        if aggregate:
                            # sum over the households
                            ax.step(
                                x_ticks, np.sum(ser, axis=1), label=lab,
                                where='post'
                            )
                        else:
                            # plots all houuseholds
                            for i, ser in enumerate(np.moveaxis(ser, 1, 0)):
                                ax.step(
                                    x_ticks, ser,
                                    label='{}_hh_{}'.format(lab, i),
                                    where='post'
                                )
                    else:
                        # plot the serie
                        ax.step(x_ticks, ser, label=lab, where='post')

            # Case not aggregated
            elif not aggr and (len(serie.shape) > 1):
                # Case many households
                if aggregate:
                    # sum over the households
                    ax.step(
                        x_ticks, np.sum(serie, axis=1), label=attr,
                        where='post'
                    )
                else:
                    # plots all houuseholds
                    for i, ser in enumerate(np.moveaxis(serie, 1, 0)):
                        ax.step(
                            x_ticks, ser,
                            label='{}_hh_{}'.format(attr, i),
                            where='post'
                        )
            else:
                # plot the serie
                ax.step(x_ticks, serie, label=attr, where='post')

            ax.legend()

        axes[-1].tick_params('x', rotation=25)

        plt.show()

    def _make_x_ticks_in_time_domain(self, ax_x_ticks):
        new_ticks = [int(i) for i in ax_x_ticks if i >= 0 and i <= self.length]
        if new_ticks[-1] != self.length - 1: # add the last data point
            new_ticks[-1] = self.length - 1
        if new_ticks[0] != 0: # adds the first data point
            new_ticks[0] = 0
        return new_ticks


    def plot(self, aggregate: bool = False):
        """Plots the profiles stored by the logger.

        The plots are shown sequentially.

        Args:
            aggregate: whether to force aggregation of the profiles.
        """
        import matplotlib.pyplot as plt
        for attr, aggr in zip(self.attributes_list, self.aggregated):
            if attr == 'current_time': continue
            serie = self.get(attr)

            if not aggr and (len(serie.shape) > 1):
                if aggregate:
                    # sum over the households
                    self._plot_single(np.sum(serie, axis=1), attr)
                else:
                    self._plot_all_households(serie, attr)
            else:
                self._plot_single(serie, attr)

            plt.legend()
            plt.show()

    def _plot_all_households(self, series, attr, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt
        artists = []
        # Iterate over the households
        for i, serie in enumerate(np.moveaxis(series, 1, 0)):
            artists.append(ax.plot(
                serie, label='{}_hh_{}'.format(attr, i)
            ))
        return artists

    def _plot_single(self, serie, attr, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt
        if isinstance(serie, dict):
            return [ax.plot(s, label=attr+'_'+lab) for lab,s in serie.items()]
        return ax.plot(serie, label=attr)


    def _raise_unkown_attribute(self, attribute:str) -> NoReturn:
        """Raise a Value Error for the given attribute.

        Args:
            attribute: The name of the invalid attribute.

        Raises:
            ValueError: Shows the attribute and the missing attrs.
        """
        err_msg = "'{}' not in SimLogger with attributes: '{}'"
        raise ValueError(err_msg.format(attribute, self.attributes_list))




class Simulator():
    """Abstract mother class for all the simulators.

    The 3 methods of this simulator are meant to be called by its
    children.
    See :ref:`create-sim_label` for more informations.

    Attributes:
        n_households: The number of households simulated.
        current_time_step: The current time step of the simulation.
        logger: A logger object, that can log the value of variables
            during the simulation.
    """

    _cache: dict
    n_households: int
    current_time_step: int
    logger: SimLogger

    def __init__(self,
            n_households: int, logger: SimLogger = None) -> None:
        """Initialize the simulator.

        This method is supposed to be called in the :py:func:`__init__`
        of all children of
        :py:class:`Simulator` .

        Args:
            n_households
            logger

        Raises:
            TypeError: If the type of n_housholds is not an integer
            ValueError: If n_households is non positive
        """
        self._cache = {}

        try:
            n_households = int(n_households)
        except Exception as exc:
            raise TypeError(
                "'n_households' must be castable to integer, \
                not:'{}'.".format(type(n_households))) from exc

        if n_households <= 0:
            raise ValueError('n_households must be positive')

        self.n_households = n_households

        # check that logger is None or SimLogger
        if logger is None:
            self.logger = logger
        elif isinstance(logger, SimLogger):
            self.logger = logger.copy()
        else:
            raise TypeError('logger kwargs is not instance of SimLogger')

    def initialize_starting_state(
        self, start_time_step: int = 0, *args, **kwargs
    ) -> None:
        r"""Initialize the values of the variables.

        If a :py:obj:`start_time_step` is given, the initilization will
        call the
        step function to simulate the requested number of time steps.
        Any arguments to be used by the step function can be passed as
        :py:obj:`*args`, :py:obj:`**kwargs`.

        Args:
            start_time_step: The time step at which the
                simulation should start. Defaults to 0.

        Raises:
            TypeError: If the type of start_time_step is not int.
            ValueError: If the start_time_step is negative.
        """
        if not isinstance(start_time_step, int):
            raise TypeError('start_time_step must be an integer')
        if start_time_step < 0:
            raise ValueError('start_time_step must be non-negative')
        self.current_time_step = 0
        for _ in range(start_time_step):
            self.step(*args, **kwargs)

        # clears the logger after initialization
        if self.logger:
            self.logger.clear()
            # visit to store the first step
            self.logger.visit_simulator(self)


    def step(self) -> None:
        """Perform a simulation step.

        Increment the :py:attr:`current_time_step`,
        calls :py:attr:`logger`, and clear cached variables.
        """
        # update the time
        self.current_time_step += 1
        # clear the cache
        self._cache.clear()
        # call the logger
        if self.logger:
            self.logger.visit_simulator(self)


# Support for Typing in simulators.
# TODO: see if we can put them in utils.sim_types.py

StepMethod = Callable[...,None]
GetMethod = Callable[
    [Simulator, Union[None, int]],
    np.ndarray]

InitilizationTime = Union[datetime.datetime, datetime.timedelta]

class MultiSimulator(Simulator):
    """Abstract simulator that can handles multiple sub-simulators.

    It distributes the simulation instructions to the subsimulators and
    handles reconstructing the output variables.
    Subsimulators can be any children of :py:class:`Simulator`, and
    should ideally be of the same kind, to allow getter methods to work
    correctly.

    Attributes:
        simulators: sub-simulators of the multi-simulator.
    """

    simulators: List[Simulator]
    def __init__(
            self, simulators: List[Simulator], logger: SimLogger=None
        ) -> None:
        """Initialize a Multi simulator.

        The daughter class of this should instantiate the simulators and
        pass them as simulators.
        The Multi simulator created will have all the get_ methods from
        the first simulator in simulators.
        The sub-simulators are assumed to all be of the same type.

        Args:
            simulators: List of subsimulator to simulate.
            logger: SimLogger. Defaults to None.
        """
        # computes the total number of households from all subsimulators
        n_households = np.sum([sim.n_households for sim in simulators])
        super().__init__(n_households, logger=logger)

        self.simulators = simulators

        self._assign_getters()

        # TODO: remove this in the children class of this that need it
        if hasattr(simulators[0], 'n_residents'):
            self.n_residents = np.concatenate(
                [s.n_residents * np.ones(s.n_households, dtype=int)
                for s in self.simulators]).reshape(-1)

    def step(self, *args, **kwargs) -> None:
        """Step function for the multi simulator.

        It handles the different :py:obj:`*args` and :py:obj:`**kwargs`
        by redistributing them
        to the inside simulators.

        Note:
            A future implmentation could implement parallelization.
        """
        # Tracks the household offset when iterating over subsimulators
        hh_offset:int = 0
        for sim in self.simulators:
            sub_args = []
            sub_kwargs = {}
            # Gets the sub-args and -kwargs
            for arg in args:
                sub_args.append(arg[hh_offset:hh_offset+sim.n_households])
            for key, values in kwargs.items():
                sub_kwargs[key] = values[hh_offset:hh_offset+sim.n_households]

            sim.step(*sub_args, **sub_kwargs)
            hh_offset += sim.n_households

        super().step()

    def _assign_getters(self) -> None:
        """Assign getters to the multi simulator, using sub-simulator."""
        # Check the first sim get_ methods
        target_sim = self.simulators[0]
        # Only accepts getter that are not already implemented
        getters = [method for method in dir(target_sim)
                    if method.startswith('get_') and (method not in dir(self))]
        for getter_name in getters:
            setattr(self, getter_name, self._create_multi_getter(getter_name))

    def _create_multi_getter(self, getter_name:str) -> GetMethod:
        """Create a getter methods that wraps up the subsimulators get
        methods.

        Args:
            getter_name: The name of the getter method
        """
        this_getter_name = getter_name
        def getter(n_ieth_hh:int = None):
            # Implement a cache system
            if this_getter_name not in self._cache:
                # Calls the getter for each subsimulator
                value = np.concatenate([
                    getattr(sim, this_getter_name)() for sim in self.simulators
                ])
                self._cache[this_getter_name] = value

            # Access a requested household
            if n_ieth_hh is None:
                return self._cache[this_getter_name]
            else:
                return self._cache[this_getter_name][n_ieth_hh]

        # also assing the doc
        getter.__doc__ = getattr(self.simulators[0], getter_name).__doc__

        return getter

    def get_mask_subgroup(self, subgroup):
        """Return a mask of the households from the subgroup.

        Args:
            subgroup: dictonary for the requuested subgroup of households

        Returns:
            The mask, where households corresponding to subgroup
                (false if not in subgroup, true if in subgroup)
        """
        # check that the subrgoup requested is in all subgroups
        is_subgroup = [np.all([
            sim.subgroup[key] == value
            if key in sim.subgroup else False
            for key, value in subgroup.items()
        ]) for sim in self.simulators]
        # gets dictonaries containing the items pair that are the same in both dicts

        # returns a mask where the subgroup of the sim is the one requested
        arrays = [
            np.ones(sim.n_households, dtype=bool)
            if subgroup else np.zeros(sim.n_households, dtype=bool)
            for subgroup, sim  in zip(is_subgroup, self.simulators)]
        return np.concatenate(arrays)



class TimeAwareSimulator(Simulator):
    """Time aware simulator remembers the time during the simulation.

    This simulator provides an initialization methods that computes how
    many steps are required to get the the start of the simulation.
    For example, this is useful when a simulator
    is initialized for 4 am but should start at 6:30 am.
    The step function keeps track of the time.

    Attributes:
        current_time: The current time of the simulation.
        step_size: The step size of the simulation.
    """

    current_time: datetime.datetime
    step_size: datetime.timedelta

    def __init__(
            self, n_households: int,
            start_datetime: datetime.datetime = (
                datetime.datetime(2014, 1, 1, 4)
            ),
            step_size: datetime.timedelta = datetime.timedelta(minutes=1),
            logger: SimLogger = None, **kwargs) -> None:
        """Create a Time Aware Simulator.

        Args:
            n_households: The number of households
            start_datetime: The start of the simulation.
                Defaults starts at 4 am on the 1rs January 2014.
            step_size: The duration of a step.
                Defaults to 1 minute.
            logger: An optional logger object. Defaults to None.

        Raises:
            TypeError: If the types of the inputs does not match.
        """

        super().__init__(n_households, logger=logger, **kwargs)
        if not isinstance(start_datetime, datetime.datetime):
            raise TypeError("Argument 'start_datetime' must be of type 'datetime'")
        if not isinstance(step_size, datetime.timedelta):
            raise TypeError("Argument 'step_size' must be of type 'timedelta'")
        self.current_time = start_datetime
        self.step_size = step_size

    def initialize_starting_state(
                self, *args,
                initialization_time: InitilizationTime = None,
                **kwargs,
            ) -> None:
        """Initialize the starting state of a time aware simulator.

        This method will compute how many initialization
        steps are required to get to :py:attr:`current_time`.
        The steps are calculated based on  :py:attr:`step_size`
        and :py:attr:`initialization_time`.
        The initial datetime will be :py:attr:`current_time`
        or if it is not possible due to step_size constraints,
        the first datetime reachable by the
        simulator before :py:attr:`current_time`.

        Args:
            *args: Any arg that must be passed to the step method
                during the initialization.
            initialization_time: The time or datetime for which the
                simulator is
                initalized.
                Defaults to None (no initialization steps, initialized
                at the current datetime).
            *kwargs: Any keyword arg that must be passed to the step method
                during the initialization.

        Raises:
            ValueError: If the initialization_time is not reachable by
                the simulator.
            NotImplementedError: If the type of the initialization_time
                is not recognized.
        """
        # Gets the time difference
        if isinstance(initialization_time, datetime.time):
            # converts to a datetime object using current time
            initialization_datetime = datetime.datetime(
                self.current_time.year,
                self.current_time.month,
                self.current_time.day,
                initialization_time.hour,
                initialization_time.minute,
                initialization_time.second,
                initialization_time.microsecond,
                tzinfo=self.current_time.tzinfo
            )
            if initialization_datetime > self.current_time:
                # check if initialized too far
                initialization_datetime -= datetime.timedelta(days=1)

        elif isinstance(initialization_time, datetime.datetime):
            initialization_datetime = initialization_time
            if initialization_datetime > self.current_time:
                raise ValueError("Cannot initialize 'initialization time': {} to the 'current_time': {}. (cannot go back in time).".format(initialization_time, self.current_time))
        elif initialization_time is None:
            initialization_datetime = self.current_time
        else:
            raise NotImplementedError("No implementation exists for initialization_time of type '{}', try using datetime.datetime instead.".format(type(initialization_time)))
        # calculate the number of steps to perform during initialization
        timedelta = self.current_time - initialization_datetime
        initialization_n_steps = int(timedelta / self.step_size)
        remainder = timedelta % self.step_size
        # initialize will run steps
        self.current_time = initialization_datetime + remainder

        # remove from kwargs the useless attribute
        kwargs.pop('start_time_step', None)

        return super().initialize_starting_state(initialization_n_steps, *args, **kwargs)

    def step(self) -> None:
        """Perform a time aware step.

        Update the :py:attr:`current_time` according to the
        :py:attr:`step_size`.
        """
        # update the internal time
        self.current_time += self.step_size
        super().step()

    def on_before_next_day(self) -> None:
        """Call back when current time passes a new day.

        This function is called by the corresponding set callback
        decorator method: Callbacks.before_next_day().
        """
        raise NotImplementedError(
            "Method 'on_before_next_day' has no implementation "
            "in {}".format(type(self).__name__)
        )

    def on_before_next_day_4am(self) -> None:
        """Call back when current time passes a new day.

        This function is called by the corresponding set callback
        decorator method: Callbacks.before_next_day_4am().
        """
        raise NotImplementedError("Method 'on_before_next_day_4am' has no implementation in {}".format(type(self).__name__))

    def on_after_next_day(self) -> None:
        """Call back when current time passes a new day.

        This function is called by the corresponding set callback
        decorator method: Callbacks.after_next_day().
        """
        raise NotImplementedError(
            "Method 'on_after_next_day' has no implementation "
            "in {}".format(type(self).__name__)
        )

    def on_after_next_day_4am(self) -> None:
        """Call back when current time passes a new day.

        This function is called by the corresponding set callback
        decorator method: Callbacks.after_next_day_4am().
        """
        raise NotImplementedError("Method 'on_after_next_day_4am' has no implementation in {}".format(type(self).__name__))

    def on_before_refresh_time(self) -> None:
        """Call back when current time passes the refresh time form data.

        This function is called by the corresponding set callback
        decorator method: Callbacks.before_refresh_time().
        """
        raise NotImplementedError(
            "Method 'on_before_refresh_time' has no implementation "
            "in {}".format(type(self).__name__)
        )

    def on_after_refresh_time(self) -> None:
        """Call back when current time passes the refresh time form data.

        This function is called by the corresponding set callback
        decorator method: Callbacks.after_refresh_time().
        """
        raise NotImplementedError(
            "Method 'on_after_refresh_time' has no implementation "
            "in {}".format(type(self).__name__)
        )

# Type for the step method of the simulators (variable input, None output)


class Callbacks():
    """Defines different Callbacks for TimeAwareSimulator children.

    Callbacks are decorators for the step functions.
    Each callback will call a specific method from the class which has
    its step
    function decorated.
    See below the function called by each callbacks, and when it is
    called.

    Usage:
        You need to decorate the step method of your class.
        Example::

            @ Callbacks.desired_callback
            def step(self, *args, **kwargs):
                # ... your code ...
                super().step()

    """
    @ staticmethod
    def before_refresh_time(step_method: StepMethod) -> StepMethod:
        """Set a call back before the .data.refresh_time, calling
        'self.on_before_refresh_time()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self: TimeAwareSimulator, *args, **kwargs) -> None:
            if self.current_time.time() == self.data.refresh_time:
                self.on_before_refresh_time()
            return step_method(self, *args, **kwargs)
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

    @ staticmethod
    def after_refresh_time(step_method: StepMethod) -> StepMethod:
        """Set a call back after the .data.refresh_time, calling
        'self.on_after_refresh_time()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self: TimeAwareSimulator, *args, **kwargs) -> None:
            previous_time = self.current_time.time()
            out = step_method(self, *args, **kwargs)
            if previous_time == self.data.refresh_time:
                self.on_after_refresh_time()
            return out
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

    @ staticmethod
    def before_next_day(step_method: StepMethod) -> StepMethod:
        """Set a call back before the next day, calling
        'self.on_before_next_day()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self: TimeAwareSimulator, *args, **kwargs) -> None:
            next_datetime = self.current_time + self.step_size
            if next_datetime.day > self.current_time.day:
                self.on_before_next_day()
            return step_method(self, *args, **kwargs)
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

    @ staticmethod
    def after_next_day(step_method: StepMethod) -> StepMethod:
        """Set a call back after the next day, calling
        'self.on_after_next_day()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self: TimeAwareSimulator, *args, **kwargs) -> None:
            previous_day = self.current_time.day
            out = step_method(self, *args, **kwargs)
            if self.current_time.day != previous_day:
                self.on_after_next_day()
            return out
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

    @ staticmethod
    def before_next_day_4am(step_method: StepMethod) -> StepMethod:
        """Set a call back before the next day at 4 am, calling
        'self.on_before_next_day_4am()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self: TimeAwareSimulator, *args, **kwargs) -> None:
            next_hour = (self.current_time + self.step_size).hour
            if next_hour == 4 and self.current_time.hour != 4:
                self.on_before_next_day_4am()
            return step_method(self, *args, **kwargs)
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

    @ staticmethod
    def after_next_day_4am(step_method: StepMethod) -> StepMethod:
        """Set a call back after the next day at 4 am, calling
        'self.on_after_next_day_4am()'

        Args:
            step_method: the step method to decorate.
        """
        def decorated_step(self:TimeAwareSimulator, *args, **kwargs)-> None:
            previous_hour = self.current_time.hour
            out = step_method(self, *args, **kwargs)
            if self.current_time.hour == 4 and previous_hour != 4:
                self.on_after_next_day_4am()
            return out
        decorated_step.__doc__ = step_method.__doc__
        return decorated_step

# TODO: remove the following once backward compatibility is ensured
def before_next_day_callback(step_method):
    def decorated_step(self:TimeAwareSimulator, *args, **kwargs):
        if (self.current_time + self.step_size).day > self.current_time.day:
            self.on_before_next_day()
        return step_method(self, *args, **kwargs)
    return decorated_step
def after_next_day_callback(step_method):
    def decorated_step(self:TimeAwareSimulator, *args, **kwargs):
        previous_day = self.current_time.day
        out = step_method(self, *args, **kwargs)
        if self.current_time.day != previous_day:
            self.on_after_next_day()
        return out
    return decorated_step
def before_next_day_callback_4am(step_method):
    def decorated_step(self:TimeAwareSimulator, *args, **kwargs):
        next_hour = (self.current_time + self.step_size).hour
        if  next_hour == 4 and self.current_time.hour != 4:
            self.on_before_next_day_4am()
        return step_method(self, *args, **kwargs)
    return decorated_step
def after_next_day_callback_4am(step_method):
    def decorated_step(self:TimeAwareSimulator, *args, **kwargs):
        previous_hour = self.current_time.hour
        out = step_method(self, *args, **kwargs)
        if self.current_time.hour == 4 and previous_hour != 4:
            self.on_after_next_day_4am()
        return out
    return decorated_step






class ExampleSimulator(Simulator):
    """This is a simple example simulator showing how to implement a sim.

    It implements a simple simulator that memorize the number of occupants
    inside the households, based on arrivals and departures.

    """
    def __init__(self, n_households, max_residents, *args, **kwargs):
        """Initialization of the function parameters and set-up.

        Put here everything you want to be performed before the simulation.
        This can inclue loading files or setting variable.
        *args, **kwargs are additional arguments that are passed to the
        parent simulator object.

        Args:
            n_households: The number of households.
                A parameter that is passed to any Simulator.
            max_residents: The maximum number of residents
                that are allowed in the households.
        """
        # Initialize the parent instance of the simulator
        super().__init__(n_households, *args, **kwargs)

        # Attributes a variable that will store the position of the households
        # Empty array of the size of the number of households
        self.occupants = np.empty(n_households, dtype=int)
        # This function must be called, as it sets up information
        # on the current time step.
        self.initialize_starting_state(max_residents)



    def initialize_starting_state(self, max_residents):
        """This is the function initializing the starting state.

        """
        # Randomly initialize the number, using the n_household class attribute
        self.occupants = np.random.randint(0, max_residents, size=self.n_households)

        # Call to the parent initialization
        super().initialize_starting_state(start_time_step=0)

    def step(self, arriving, leaving):
        """Step function of the simulator.
        """
        self.occupants += arriving
        self.occupants -= leaving
        super().step()


class TimeAwareExampleSimulator(TimeAwareSimulator):
    """This is a simple example simulator showing how to implement a sim.

    It implements a simple simulator that memorize the number of occupants
    inside the households, based on arrivals and departures.
    """
    def __init__(
            self, n_hh, max_residents, *args,
            initialization_algo='all_inside_at_4am', **kwargs
        ):
        """Initialization of the function parameters and set-up.

        Put here everything you want to be performed before the simulation.
        This can inclue loading files, setting variable, or initializing the simulation.
        *args, **kwargs are additional arguments that are passed to the parent simulator object.

        Args:
            n_hh (int): The number of households. A parameter that is passed to any Simulator.
        """
        # Initialize the super instance of the simulator
        super().__init__(n_hh, *args, **kwargs)

        # Attributes a variable that will store the position of the households
        self.occupants = np.empty(n_hh, dtype=int)
        # This function must be called, as it sets up information on the current time step.
        self.initialize_starting_state(max_residents, initialization_algo)



    def initialize_starting_state(self, max_residents, initialization_algo):
        """This is the function initializing the starting state.

        Different methods, depending on the algorithm.
        """
        if initialization_algo == 'random':
            # Same as previous initialization
            self.occupants = np.random.randint(
                0, max_residents, size=self.n_households
            )
            super().initialize_starting_state(
                # Will not run any step
                # self.current_time tracks the time during simulation
                initialization_time=self.current_time
            )
        elif initialization_algo == 'all_inside_at_4am':
            # Now assume that all the residents are at home at 4 AM

            # Sets the residents to be all there
            self.occupants = max_residents * np.ones(self.n_households)

            # Call to the parent initialization
            super().initialize_starting_state(
                # Specifies that the method has been initialized for 4 AM
                initialization_time=datetime.time(4, 0, 0),
                # Sets dummy variables for the step function during
                # the initial steps
                arriving=np.zeros(self.n_households),
                leaving=np.zeros(self.n_households)
            )
        else:
            # Import a specific error message
            from demod.utils.error_messages import UNIMPLEMENTED_ALGO_IN_METHOD
            # Raise the error message
            raise NotImplementedError(UNIMPLEMENTED_ALGO_IN_METHOD.format(
                algo=initialization_algo,
                method=self.initialize_starting_state,
            ))

    @ Callbacks.before_next_day_4am
    def step(self, arriving, leaving):
        """Step function of the simulator.
        """
        self.occupants += arriving
        self.occupants -= leaving
        super().step()

    def on_before_next_day_4am(self):
        """This function is called by the Callbacks.before_next_day_4am

        It will be called every day at 4 am, right before the step
        function is called.
        """
        # We want to print the percentage households with no one at home.
        print("There is {} percent of the households that are empty".format(
            np.mean(self.occupants==0) * 100
        ))
