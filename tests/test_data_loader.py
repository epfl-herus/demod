from demod.datasets.Germany.loader import GermanDataHerus
import unittest
import numpy as np
from demod.datasets.ExampleData import ExampleLoader
from demod.datasets.GermanTOU.loader import GTOU


class TestBaseLoader(unittest.TestCase):

    loader = ExampleLoader
    test_file_name: str = "test"

    def test_instantiate(self):
        loader = self.loader()

    def test_errors(self):
        loader = self.loader()
        # TODO: should check the REGEX messages
        self.assertRaises(
            FileNotFoundError,
            loader._raise_missing_raw,
            "test_file",
        )
        self.assertRaises(
            FileNotFoundError,
            loader._raise_missing_raw,
            "test_file",
            optional_download_website="https://www.github.com",
        )

    def test_saving_loading_npy(self):
        loader = self.loader()
        array = np.array([1])
        loader._save_parsed_data(self.test_file_name, array)
        a = loader._load_parsed_data(self.test_file_name)
        self.assertTrue(np.all(array == a))

    def test_saving_loading_npz(self):
        loader = self.loader()
        array = np.array([1])
        loader._save_parsed_data(self.test_file_name, array, npz=True)
        a_list = loader._load_parsed_data(self.test_file_name)
        self.assertTrue(np.all(array == a_list[0]))

    def test_saving_loading_compress(self):
        loader = self.loader()
        array = np.array([1])
        loader._save_parsed_data(self.test_file_name, array, compress=True)
        a_list = loader._load_parsed_data(self.test_file_name)
        self.assertTrue(np.all(array == a_list[0]))

    def test_pickle_state(self):
        loader = self.loader()
        # check false by defualt
        self.assertFalse(loader.allow_pickle)
        loader = self.loader(allow_pickle=True)
        self.assertTrue(loader.allow_pickle)

    def test_example(self):
        loader = self.loader()
        arr = loader.get_example1_data()
        self.assertTrue(np.all(arr == np.array([1, 2])))
        arr1, arr2 = loader.get_example1_data(return_ex2=True)
        self.assertTrue(np.all(arr1 == np.array([1, 2])))
        print(arr2)
        self.assertTrue(np.all(arr2 == np.array([2, 3])))


class TestGTOU(unittest.TestCase):
    loader = GTOU

    def test_instantiate(self):
        loader = self.loader()

    def test_load_sparse(self):
        loader = self.loader("Sparse9States")
        loader.load_sparse_tpm({"n_residents": 1, "weekday": [6, 7]})


class TestHerus(unittest.TestCase):
    loader = GermanDataHerus

    def test_instantiate(self):
        loader = self.loader()

