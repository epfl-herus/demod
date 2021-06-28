"""Support for Typing in simulators.

Currently not possible to place types definition there.
"""
from __future__ import annotations
from typing import Any, Dict, List, NewType, Union

import numpy as np

from ..simulators.base_simulators import (
    StepMethod, GetMethod, InitilizationTime
)

StepMethod = StepMethod
GetMethod = GetMethod
InitilizationTime = InitilizationTime
Subgroup = NewType(
    'Subgroup',
    Dict[str, Any]
)
Subgroups = List[Subgroup]
# Maps raw states to a labelled states
# ex 4-states: 0,1,2,3 -> 00, 01, 10, 11
StateLabels = Union[List[str], List[int], np.ndarray]
# Maps labelled states to activity names
ActivityLabels = Union[List[str], np.ndarray]
States = np.ndarray
Transitions = Dict[str, np.ndarray]
TPM = np.ndarray
TPMs = np.ndarray
# Probabilities
PDF = np.ndarray
CDF = np.ndarray

# Variables
ActivitiesDict = Dict[str, np.ndarray]
AppliancesDict = Dict[str, np.ndarray]
Temperatures = Dict[str, np.ndarray]
ThermostatsStates = Dict[str, np.ndarray]
HeatOutputs = Dict[str, np.ndarray]
HeatingControls = Dict[str, np.ndarray]
HeatingDemand = Dict[str, np.ndarray]
