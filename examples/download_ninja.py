import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from demod.datasets.RenewablesNinja.loader import NinjaRenewablesClimate


data = NinjaRenewablesClimate('FR')