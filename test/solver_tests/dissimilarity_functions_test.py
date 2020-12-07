"""
Tests for the dissimilarity functions that are supported by the DistanceSolver
module.
"""
import unittest

import numpy as np

from cassiopeia.solver import dissimilarity_functions


class TestDissimilarityFunctions(unittest.TestCase):
    def setUp(self):

        self.s1 = ["0", "1", "0", "-", "1", "2"]
        self.s2 = ["1", "1", "0", "0", "1", "3"]
        self.all_missing = ["-", "-", "-", "-", "-", "-"]

        self.priors = {
            0: {"1": 0.5, "2": 0.5},
            1: {"1": 0.5, "2": 0.5},
            2: {"1": 0.25, "2": 0.75},
            3: {"1": 0.3, "2": 0.7},
            4: {"1": 0.4, "2": 0.6},
            5: {"1": 0.1, "2": 0.05, "3": 0.85},
        }

    def test_weighted_hamming_distance_identical(self):

        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s1
        )

        self.assertEqual(dissimilarity, 0)

    def test_weighted_hamming_distance_no_priors(self):

        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s2
        )

        self.assertEqual(dissimilarity, 3 / 5)

    def test_weighted_hamming_distance_priors(self):

        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s2, priors=self.priors
        )

        expected_dissimilarity = np.sum(
            [
                -np.log(self.priors[0]["1"]),
                np.log(self.priors[1]["1"]) * 2,
                np.log(self.priors[4]["1"]) * 2,
                -(np.log(self.priors[5]["2"]) + np.log(self.priors[5]["3"])),
            ]
        )

        self.assertEqual(dissimilarity, expected_dissimilarity / 5)

    def test_weighted_hamming_distance_all_missing(self):

        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.all_missing, priors=self.priors
        )

        self.assertEqual(dissimilarity, 0)


if __name__ == "__main__":
    unittest.main()
