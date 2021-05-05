"""Provides data types for various Dataset Functions."""
from demod.datasets.base_loader import DatasetLoader
from typing import Union


DataInput = Union[str, DatasetLoader]
