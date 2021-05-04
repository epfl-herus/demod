"""Loader for Destatis."""
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from ...utils.sim_types import AppliancesDict, Subgroup, Subgroups
from ...utils.appliances import assign_ownership_from_prob1_and_number
from ..base_loader import ApplianceLoader, PopulationLoader
from ...utils.subgroup_handling import remove_time_attributues


class Destatis(ApplianceLoader, PopulationLoader):
    """Loader for the data from Destatis.

    Different sources.

    * population
    https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/lrbev05.html
    https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Haushalte-Familien/Tabellen/2-5-familien.html
    * appliances

    TODO: add references to this data
    """

    DATASET_NAME = 'DESTATIS'

    def _parse_population_subgroups(
        self, population_type: str,
    ) -> Tuple[Subgroups, List[float], int]:
        if population_type == 'resident_number' or population_type == 'crest':
            return self._parse_population_subgroups('resident_number_2019')
        elif population_type == 'resident_number_2013':
            return (
                [
                    {'n_residents': 1},
                    {'n_residents': 2},
                    {'n_residents': 3},
                    {'n_residents': 4},
                    {'n_residents': 5}
                ],
                # added a 2nd decimal to fill the cdf
                np.array([40.54, 34.43, 12.51, 9.21, 3.31]) / 100.,
                int(39933e3)
            )
        elif population_type == 'resident_number_2019':
            return (
                [
                    {'n_residents': 1},
                    {'n_residents': 2},
                    {'n_residents': 3},
                    {'n_residents': 4},
                    {'n_residents': 5}
                ],
                np.array([42.3, 33.2, 11.9, 9.1, 3.5]) / 100.,
                int(41506e3)
            )
        elif population_type == 'household_types':
            return self._parse_population_subgroups('household_types_2019')
        elif population_type == 'household_types_2019':
            hh_numbers = [
                17.067,     # leaving alone,  https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/unattached-people.html#fussnote-1-54116  ]
                11.850,     # couple withouth children https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/couples.html
                1.007,      # lone parents 1 children https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Haushalte-Familien/Tabellen/2-5-familien.html
                0.408,      # lone parents 2 children, (at least one under 18)
                0.109,      # lone parents 3 children, (at least one under 18)
                2.553,      # couples with 1 children (at least one under 18)
                2.370,      # couples with 2 children (at least one under 18)
                0.801,      # couples with 3 children (at least one under 18)
            ]
            hh_pdf = np.array(hh_numbers) / np.sum(hh_numbers)
            return (
                [
                    {
                        'n_residents': 1,
                        'household_type': 1
                    },{
                        'n_residents': 2,
                        'household_type': 2
                    },{
                        'n_residents': 2,
                        'household_type': 3
                    },{
                        'n_residents': 3,
                        'household_type': 3
                    },{
                        'n_residents': 4,
                        'household_type': 3
                    },{
                        'n_residents': 3,
                        'household_type': 4
                    },{
                        'n_residents': 4,
                        'household_type': 4
                    },{
                        'n_residents': 5,
                        'household_type': 4
                    }
                ],
                hh_pdf,
                int(41506e3)
            )
        else:
            raise NotImplementedError(
                "{} is not implemented for 'population_type' with value "
                "'{}'".format(
                    self._parse_population_subgroups,
                    population_type
                )
            )


    def _parse_appliance_ownership_dict(
        self,  subgroup: Subgroup
    ) -> Dict[str, float]:

        # The appliance ownership does not depend on the time attributes.
        subgroup = remove_time_attributues(subgroup)

        data_path = self.raw_path + os.sep + "appliances_penetration.ods"
        df = pd.read_excel(data_path, header=[0, 1, 2])

        try:
            col_n = self._get_column_appliance_subgroup(**subgroup)
        except ValueError as val_err:
            raise ValueError(
                "Could not load subgroup ownership from DESTATIS "
                " for subgroup with value : {}".format(subgroup)
            ) from val_err

        # Gets the column of the demod labels
        app_types = df[df.columns[0]].to_numpy()

        # Gets the column of the subgroup
        column = df[df.columns[col_n]].to_numpy()
        column[column == "-"] = np.nan  # replace the strig char
        column = np.asfarray(column) / 100.0

        # Offset for where the prercent is placed
        offset_percent = 51
        base_offset = 5

        # Creates the empty owenership dic
        ownership_dic = {}

        # Iterates over the appliances names to create the ownership dict.
        for it, app_type in enumerate(app_types):
            if it == offset_percent + base_offset:  # once we arrive to percent, stop
                break
            if (app_type is None) or (app_type is np.nan):
                continue
            number = column[it]
            prob = column[it + offset_percent]
            # removes spaces present in data
            app_type = str(app_type).replace(" ", "")
            probs = assign_ownership_from_prob1_and_number(prob, number)
            # adds to the dictionary
            for i, prob in enumerate(probs):
                ownership_dic[app_type + "_" + str(i + 1)] = (
                    0.0 if np.isnan(prob) else prob)

            prob_0 = ownership_dic.pop(app_type + '_1')
            ownership_dic[app_type] = prob_0

        return ownership_dic

    def _get_column_appliance_subgroup(
        self,
        n_residents=None,
        household_revenue=None,
        year=None,
        household_type=None,
        life_situation=None,
        age=None,
        geburtsland=None,
        gender=None,
        household_position=None,
        is_travelling=None,
        **kwargs
    ):
        """Return the column in the file that correspond to german appliances.

        Basic implementation cannot combine lots of attributes.
        It will first check if both household_type and n__residents are given
        The it will check n_residents, revenue and year

        Args:
            n_residents (int, optional):
            The number of residents in the household. Defaults to None.
            household_revenue ([type], optional):
                [description]. Defaults to None.
            year ([type], optional):
                [description]. Defaults to None.
            household_type (int, optional):
                The type of the household
                    (1 = Einpersonenhaushalt,
                    2 = Paare ohne Kinder,
                    3 = Alleinerziehende mit mindestens einem Kind
                    unter18 Jahren und ledigen Kindern unter 27 Jahren,
                    4= Paare mit mindestens einem Kind unter 18 Jahren
                    und ledigen Kindernledigen Kindern unter 27 Jahren,
                    5 = Sonstige Haushalte). Defaults to None.
            gender (int, optional):
                1 = man, 2 = woman. Defaults to None.
            age (int or tuple, optional):
                The age of the participants, if tuple returns all inside
                interval. Defaults to None.
        """
        # first check the family types
        offset_hh_type = 47
        offset_hh_with_children = 57
        if household_type == 1:  # leaving alone,
            if gender is None:
                return offset_hh_type + 0
            elif gender == 1:  # man
                return offset_hh_type + 1
            elif gender == 2:  # woman
                return offset_hh_type + 2
            else:
                raise ValueError("Gender must be 1, 2 or None")
        elif household_type == 2:  # couple withouth children
            return offset_hh_type + 5
        elif (
            household_type == 3 and n_residents == 2
        ):  # lone parents 1 children
            return offset_hh_with_children + 1
        elif (
            household_type == 3 and n_residents == 3
        ):  # lone parents 2 children
            return offset_hh_with_children + 2
        elif (
            household_type == 3 and n_residents == 4
        ):  # lone parents 3 children
            return offset_hh_with_children + 2
        elif (
            household_type == 4 and n_residents == 3
        ):  # couple with 1 children
            return offset_hh_with_children + 4
        elif (
            household_type == 4 and n_residents == 4
        ):  # lone parents 2 children
            return offset_hh_with_children + 5
        elif (
            household_type == 4 and n_residents == 5
        ):  # lone parents 3 children
            return offset_hh_with_children + 6

        # finds the columns corresponding to the desired socio economical types
        elif n_residents:
            assert (
                n_residents > 0 and n_residents < 6
            ), "n_residents must be from 1 to 5"
            offset_n_residents = 12
            return offset_n_residents + n_residents

        elif household_revenue:
            offset_revenue = 27
            assert (
                household_revenue >= 0
            ), " household_revenue cannot be negative "
            if household_revenue < 900:
                col = 0
            elif household_revenue < 1300:
                col = 1
            elif household_revenue < 1500:
                col = 2
            elif household_revenue < 2000:
                col = 3
            elif household_revenue < 2600:
                col = 4
            elif household_revenue < 3600:
                col = 5
            elif household_revenue < 5000:
                col = 6
            else:
                col = 7

            return offset_revenue + col

        elif year:
            if year == 2008:
                col = 2
            elif year == 2013:
                col = 3
            elif year == 2018:
                col = 4
            else:
                raise ValueError("Year not valid, only 2008, 2013 and 2018")

            return col
        else:
            raise ValueError("Could not find any appliance set for  subgroup'")
