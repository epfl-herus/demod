"""simulators is a library of simulators for energy demand modelling.

The module is splitted in different submodules which each provide
different implmenetations for different parts of the simulation.
"""
from __future__ import annotations

import os
import sys

from . import base_simulators
from . import crest_simulators
from . import sparse_simulators
from . import appliance_simulators
from . import heating_simulators
from . import lighting_simulators
from . import car_simulator
from . import load_simulators
from . import util

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
