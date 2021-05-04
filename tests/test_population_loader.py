from demod.datasets.DESTATIS.loader import Destatis



import unittest



class DestatisLoaderTest(unittest.TestCase):

    def test_loading(self):
        data = Destatis()

        data.load_population_subgroups('household_types_2019')
        data.load_population_subgroups('residents_number_2019')
        data.load_population_subgroups('residents_number_2013')