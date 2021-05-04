
# TODO add test for the pdf function chaecks
import unittest



import numpy as np



from demod.utils.monte_carlo import monte_carlo_from_cdf
from demod.utils.distribution_functions import check_valid_cdf


class TestMonteCarlo(unittest.TestCase):

    def test_2by2(self):
        randomcdf = np.array([[0.2, 1.], [0.2, 1.]])
        out = monte_carlo_from_cdf(randomcdf)
        self.assertEqual(len(out), 2)

    def test_deterministic(self):
        deterministiccdf = np.array([[0.0, 1.], [1., 1.]])
        out = monte_carlo_from_cdf(deterministiccdf)
        self.assertTrue(np.all(out == np.array([1, 0])))

    def test_3by3(self):
        randomcdf = np.array([[0.2, 0.4, 1.], [0.2, 0.9, 1.], [0.0, 0.3, 1.]])
        out = monte_carlo_from_cdf(randomcdf)
        self.assertEqual(len(out), 3)

    def test_3by2(self):
        randomcdf = np.array([[0.2, 1.], [0.2,  1.], [0.0, 1.]])
        out = monte_carlo_from_cdf(randomcdf)
        self.assertEqual(len(out), 3)

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)




class Testcheck_valid_cdfs(unittest.TestCase):

    def test_valid(self):
        validcdf = np.array([[0.2, 1.], [0.7, 1.]])
        self.assertTrue(check_valid_cdf(validcdf))

    def test_constant(self):
        constantcdf = np.array([[0.0, 1.], [1., 1.]])
        self.assertTrue(check_valid_cdf(constantcdf))

    def test_invalidEnd(self):
        # check that fails if the last value is not 1
        constantcdf = np.array([[0.0, 0.2], [1., 1.]])
        with self.assertRaises(ValueError):
            check_valid_cdf(constantcdf)

    def test_decreasing(self):
        # check that cdf is not decreasing
        decreasingcdf = np.array([[0.2, 0.1, 1.], [0.2, 0.9, 1.], [0.0, 0.3, 1.]])
        with self.assertRaises(ValueError):
            check_valid_cdf(decreasingcdf)

    def test_smaller1(self):
        # check that cdf is not decreasing
        decreasingcdf = np.array([[-0.2, 0.4, 1.], [0.2, 0.9, 1.], [0.0, 0.3, 1.]])
        with self.assertRaises(ValueError):
            check_valid_cdf(decreasingcdf)


if __name__ == '__main__':
    unittest.main()
