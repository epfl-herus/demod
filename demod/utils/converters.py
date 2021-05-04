"""Some units converters.
"""

from typing import Any


def joules_to_kwh(x: Any) -> Any:
    """Transforms joules to kwh.

    1 kwh = 3'600'000 Joules

    Args:
        x: The object to convert. Must support division by float.
    """
    return x / 3.6e6

def kwh_to_joules(x: Any) -> Any:
    """Transforms kwh to joules.

    1 kwh = 3'600'000 Joules

    Args:
        x: The object to convert. Must support multiplication by float.
    """
    return x * 3.6e6