"""Plotter functions for visualizing results."""
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .sim_types import AppliancesDict, StateLabels, States

# Set the default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
    '#ccebc5', '#ffed6f'
])


FIGSIZE = (16, 9)
MAIN_ACTIVITIES = ['active_occupancy', 'away', 'sleeping']



def optional_ax(function):
    """Check or give an ax was given to the function.

    If an ax is given, plot on the ax.
    If no ax is given, create one and show the plot.
    """
    def plot_function(*args, ax: plt.Axes = None, **kwargs):
        if ax is None:
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.add_subplot(1, 1, 1)
            need_plot = True
        else:
            need_plot = False

        function(*args, ax=ax, **kwargs)

        if need_plot:
            return plt.show()

    plot_function.__doc__ = function.__doc__

    return plot_function


@ optional_ax
def plot_household_activities(
    dict_states: Dict[str, np.ndarray],
    time_axis: np.ndarray = None,
    colors=np.array(['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']),
    main_states_on_top: bool = True,
    marker='>',
    ax: plt.Axes = None,
    **kwargs
) -> None:
    """Plot the household states during the day for a single household.

    Attributes:
        dict_states: A dictionary where keys are the names of the activities
            or states and the values are ndarray of size = n_steps and
            values are the number of residents in this states/activity
        time_axis: The datetime axis to plot
        colors: An array of color names to be used where the colors are
            matched to the number of occupants in each activity
        main_states_on_top: if true, will put the main states on top
            (active_occupancy, away, sleeping)
        ax: an ax on which to plot the activities
        kwargs: any other keywork argument from `ax.scatter
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html?highlight=scatter#matplotlib-axes-axes-scatter>`_

    """
    # Color for 0 person doing activity
    if colors is not None:
        colors = np.append(np.array(''), colors)
    # Creates a time axis
    if time_axis is None:
        # Creates an ax of the length of states
        time_axis = np.arange(len(dict_states[list(dict_states.keys())[0]]))

    max_number = 1

    def scatter(array, i):
        mask_activity_occuring = array > 0
        ax.scatter(
            time_axis[mask_activity_occuring],
            i * np.ones(sum(mask_activity_occuring)),
            c=(
                # In case no color was given, this will use a colormap
                array[mask_activity_occuring]
                if colors is None
                else colors[array[mask_activity_occuring]]
            ),
            marker=marker, **kwargs
            )

    labels = []
    labels_main = []
    # Start with secondary activities
    i = 0
    for keys, array in dict_states.items():
        if main_states_on_top and keys in MAIN_ACTIVITIES:
            labels_main.append(keys)
            continue  # plot later
        scatter(array, i)
        labels.append(keys)
        # Records the max number of residents
        max_number = max(max_number, max(array))
        i += 1

    # Plot then the main activities
    for j, key in enumerate(labels_main):
        scatter(dict_states[key], i+j)
        labels.append(key)
        max_number = max(max_number, max(array))

    ax.set_yticks(np.arange(0, len(labels)))
    ax.set_yticklabels(labels)
    # Adds the legend
    if colors is not None:
        ax.legend(handles=[
            Circle((0, 0), color=colors[i], label='{}'.format(i))
            for i in range(1, max_number+1)
        ])


@ optional_ax
def plot_appliance_consumptions(
    consumptions: np.ndarray,
    appliances_dict: AppliancesDict,
    time_axis: np.ndarray = None,
    differentiative_factor: str = 'type',
    labels_list: List[Any] = None,
    ax: plt.Axes = None, **kwargs
) -> None:
    """Plot the consumption of the appliances.

    Args:
        consumptions: A ndarray of size = (n_steps, n_appliances) and
            values are the consumption of each appliance
        appliances_dict: The dictionary of appliances from simulator
        time_axis: The datetime axis to plot
        differentiative_factor: The key in the appliance_dict that
            should be used to differentiate the appiances
            ex: ('type', 'name', 'related_activity')
        labels_list: Optional list of the labels to be plotted
            Usefull for choosing an order
        ax: an ax on which to plot the activities

    """
    # Finds all the attributes or uses the ones given
    attributes_list = (
        np.unique(appliances_dict[differentiative_factor])
        if labels_list is None else labels_list
    )
    # Sum up consumption mapping these appliances
    consumptions_to_plot = [
        np.sum(consumptions[
            :, appliances_dict[differentiative_factor] == attribute
        ], axis=-1)
        for attribute in attributes_list
    ]
    if time_axis is None:
        # Creates an ax of the length of states
        time_axis = np.arange(len(consumptions))

    ax.stackplot(
        time_axis, *consumptions_to_plot, labels=attributes_list, step='post'
    )
    # Reverse the legend, in the same way as they are stacked
    handles, lab = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lab[::-1])


def plot_stack_states(
    states: States, labels: StateLabels, ax: plt.Axes = None, **kwargs
) -> Any:
    """Plot the stacked states.

    Args:
        states: the states to be stacked in the plot
        labels: the labels of the different states values in
            the states array
        ax: An optional ax object on which to plot the labels
        **kwargs: any arguments passed to 'plt.stackplot'

    Returns:
        fig, ax: a matplotlib figure and ax objects
    """
    # stack plot with an hourhly axis
    if ax is None:
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(1, 1, 1)
        need_return = True
    else:
        need_return = False
    # stack the states for the stack plot
    stacked_states = np.apply_along_axis(
        np.bincount, 0, np.array(states, int), minlength=np.max(states)+1
    )
    ax.stackplot(
        np.arange(0, states.shape[1]),  # Create an x axis
        stacked_states,
        labels=labels
    )
    handles, lab = ax.get_legend_handles_labels()
    # Reverse the legend, so they follow the states order
    ax.legend(handles[::-1], lab[::-1], loc='right')

    if need_return:
        return fig, ax


@ optional_ax
def plot_stacked_activities(
    dict_states: Dict[str, np.ndarray],
    time_axis: np.ndarray = None,
    ax: plt.Axes = None,
    step: str = 'post',
    **kwargs
) -> None:
    """Plot the stacked household states during time.

    Attributes:
        dict_states: A dictionary where keys are the names of the activities
            or states and the values are ndarray of size = n_steps and
            values are the number of residents in this states/activity
        time_axis: The datetime axis to plot
        ax: an ax on which to plot the activities
        kwargs: any other keywork argument from `ax.stackplot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stackplot.html#matplotlib.pyplot.stackplot>`_

    """
    # Creates a time axis
    if time_axis is None:
        # Creates an ax of the length of states
        time_axis = np.arange(len(dict_states[list(dict_states.keys())[0]]))

    ax.stackplot(
        time_axis,
        dict_states.values(),
        labels=dict_states.keys(),
        step='post',
        **kwargs
    )
    # Reverse the legend, in the same way as they are stacked
    handles, lab = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lab[::-1])
