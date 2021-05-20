"""Plotter functions for visualizing results."""
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .sim_types import StateLabels, States


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
