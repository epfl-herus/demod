"""Helpers for managing countries.

We should use a library for that instead:
https://pypi.org/project/iso3166/#description
"""


def country_name_to_code(country_name: str) -> str:
    """Return the code corresponding the country.

    Args:
        country_name: the name

    Returns:
        str: The country code
    """
    dic = {
        'germany': 'DE',
        'switzerland': 'CH',
        'england': 'GB'
    }

    try:
        code = dic[country_name]
    except KeyError as key_err:
        raise NotImplementedError(
            'Must provide a country conversion to code.') from key_err

    return code
