"""
Tests for the dissimilarity functions that are supported by the DistanceSolver
module.
"""
import unittest
from unittest import mock

import numpy as np

from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import solver_utilities


class TestDissimilarityFunctions(unittest.TestCase):
    def setUp(self):

        self.s1 = np.array([0, 1, 0, -1, 1, 2])
        self.s2 = np.array([1, 1, 0, 0, 1, 3])
        self.all_missing = np.array([-1, -1, -1, -1, -1, -1])
        self.ambiguous = [(0,), (-1, 0), (0,), (-1, 0), (1,), (1,)]

        self.priors = {
            0: {1: 0.5, 2: 0.5},
            1: {1: 0.5, 2: 0.5},
            2: {1: 0.25, 2: 0.75},
            3: {1: 0.3, 2: 0.7},
            4: {1: 0.4, 2: 0.6},
            5: {1: 0.1, 2: 0.05, 3: 0.85},
        }

        self.badpriors = {0: {1: 0}, 1: {1: -1, 2: -1.5}}

        self.nlweights = solver_utilities.transform_priors(
            self.priors, "negative_log"
        )

        self.iweights = solver_utilities.transform_priors(
            self.priors, "inverse"
        )

        self.sqiweights = solver_utilities.transform_priors(
            self.priors, "square_root_inverse"
        )

    def test_bad_prior_transformations(self):
        with self.assertRaises(solver_utilities.PriorTransformationError):
            solver_utilities.transform_priors(self.badpriors, "negative_log")

    def test_negative_log_prior_transformations(self):
        expectedweights = {
            0: {1: -np.log(0.5), 2: -np.log(0.5)},
            1: {1: -np.log(0.5), 2: -np.log(0.5)},
            2: {1: -np.log(0.25), 2: -np.log(0.75)},
            3: {1: -np.log(0.3), 2: -np.log(0.7)},
            4: {1: -np.log(0.4), 2: -np.log(0.6)},
            5: {1: -np.log(0.1), 2: -np.log(0.05), 3: -np.log(0.85)},
        }
        self.assertEqual(self.nlweights, expectedweights)

    def test_inverse_prior_transformations(self):
        expectedweights = {
            0: {1: 1 / (0.5), 2: 1 / (0.5)},
            1: {1: 1 / (0.5), 2: 1 / (0.5)},
            2: {1: 1 / (0.25), 2: 1 / (0.75)},
            3: {1: 1 / (0.3), 2: 1 / (0.7)},
            4: {1: 1 / (0.4), 2: 1 / (0.6)},
            5: {1: 1 / (0.1), 2: 1 / (0.05), 3: 1 / (0.85)},
        }
        self.assertEqual(self.iweights, expectedweights)

    def test_sq_inverse_prior_transformations(self):
        expectedweights = {
            0: {1: np.sqrt(1 / 0.5), 2: np.sqrt(1 / 0.5)},
            1: {1: np.sqrt(1 / 0.5), 2: np.sqrt(1 / 0.5)},
            2: {1: np.sqrt(1 / 0.25), 2: np.sqrt(1 / 0.75)},
            3: {1: np.sqrt(1 / 0.3), 2: np.sqrt(1 / 0.7)},
            4: {1: np.sqrt(1 / 0.4), 2: np.sqrt(1 / 0.6)},
            5: {
                1: np.sqrt(1 / 0.1),
                2: np.sqrt(1 / 0.05),
                3: np.sqrt(1 / 0.85),
            },
        }
        self.assertEqual(self.sqiweights, expectedweights)

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

    def test_weighted_hamming_distance_priors_negative_log(self):
        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s2, weights=self.nlweights
        )

        expected_dissimilarity = np.sum(
            [
                -np.log(self.priors[0][1]),
                -(np.log(self.priors[5][2]) + np.log(self.priors[5][3])),
            ]
        )

        self.assertEqual(dissimilarity, expected_dissimilarity / 5)

    def test_weighted_hamming_distance_priors_inverse(self):
        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s2, weights=self.iweights
        )

        expected_dissimilarity = np.sum(
            [
                1 / (self.priors[0][1]),
                1 / (self.priors[5][2]) + 1 / (self.priors[5][3]),
            ]
        )

        self.assertEqual(dissimilarity, expected_dissimilarity / 5)

    def test_weighted_hamming_distance_priors_sq_inverse(self):
        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.s2, weights=self.sqiweights
        )

        expected_dissimilarity = np.sum(
            [
                np.sqrt(1 / self.priors[0][1]),
                np.sqrt(1 / self.priors[5][2]) + np.sqrt(1 / self.priors[5][3]),
            ]
        )

        self.assertEqual(dissimilarity, expected_dissimilarity / 5)

    def test_weighted_hamming_distance_all_missing(self):

        dissimilarity = dissimilarity_functions.weighted_hamming_distance(
            self.s1, self.all_missing, weights=self.nlweights
        )

        self.assertEqual(dissimilarity, 0)

    def test_hamming_similarity_without_missing_identical(self):

        similarity = dissimilarity_functions.hamming_similarity_without_missing(
            self.s1, self.s1, -1
        )

        self.assertEqual(similarity, 3)

    def test_hamming_similarity_without_missing_no_priors(self):

        similarity = dissimilarity_functions.hamming_similarity_without_missing(
            self.s1, self.s2, -1
        )

        self.assertEqual(similarity, 2)

    def test_hamming_similarity_without_missing_priors(self):
        similarity = dissimilarity_functions.hamming_similarity_without_missing(
            self.s1, self.s2, -1, weights=self.nlweights
        )

        expected_similarity = np.sum(
            [-np.log(self.priors[1][1]), -np.log(self.priors[4][1])]
        )

        self.assertEqual(similarity, expected_similarity)

    def test_hamming_similarity_without_missing_all_missing(self):

        similarity = dissimilarity_functions.hamming_similarity_without_missing(
            self.s1, self.all_missing, -1, weights=self.nlweights
        )

        self.assertEqual(similarity, 0)

    def test_hamming_similarity_normalized_identical(self):

        similarity = (
            dissimilarity_functions.hamming_similarity_normalized_over_missing(
                self.s1, self.s1, -1
            )
        )

        self.assertEqual(similarity, 3 / 5)

    def test_hamming_similarity_normalized_no_priors(self):

        similarity = (
            dissimilarity_functions.hamming_similarity_normalized_over_missing(
                self.s1, self.s2, -1
            )
        )

        self.assertEqual(similarity, 2 / 5)

    def test_hamming_similarity_normalized_priors(self):
        similarity = (
            dissimilarity_functions.hamming_similarity_normalized_over_missing(
                self.s1, self.s2, -1, weights=self.nlweights
            )
        )

        expected_similarity = np.sum(
            [-np.log(self.priors[1][1]), -np.log(self.priors[4][1])]
        )

        self.assertEqual(similarity, expected_similarity / 5)

    def test_hamming_similarity_normalized_all_missing(self):

        similarity = (
            dissimilarity_functions.hamming_similarity_normalized_over_missing(
                self.s1, self.all_missing, -1, weights=self.nlweights
            )
        )

        self.assertEqual(similarity, 0)

    def test_weighted_hamming_similarity_identical(self):

        similarity = dissimilarity_functions.weighted_hamming_similarity(
            self.s1, self.s1, -1
        )

        self.assertEqual(similarity, 8 / 5)

    def test_weighted_hamming_similarity_no_priors(self):

        similarity = dissimilarity_functions.weighted_hamming_similarity(
            self.s1, self.s2, -1
        )

        self.assertEqual(similarity, 1)

    def test_weighted_hamming_similarity_priors(self):
        similarity = dissimilarity_functions.weighted_hamming_similarity(
            self.s1, self.s2, -1, weights=self.nlweights
        )

        expected_similarity = np.sum(
            [-np.log(self.priors[1][1]) * 2, -np.log(self.priors[4][1]) * 2]
        )

        self.assertEqual(similarity, expected_similarity / 5)

    def test_weighted_hamming_similarity_all_missing(self):

        similarity = dissimilarity_functions.weighted_hamming_similarity(
            self.s1, self.all_missing, -1, weights=self.nlweights
        )

        self.assertEqual(similarity, 0)

    def test_cluster_dissimilarity(self):
        dissimilarity_function = (
            dissimilarity_functions.weighted_hamming_distance
        )
        linkage_function = np.mean

        result = dissimilarity_functions.cluster_dissimilarity(
            dissimilarity_function,
            self.s1,
            self.ambiguous,
            -1,
            self.nlweights,
            linkage_function,
        )
        np.testing.assert_almost_equal(result, 1.2544, decimal=4)

    def test_hamming_distance(self):

        distance = dissimilarity_functions.hamming_distance(self.s1, self.s2)

        self.assertEqual(distance, 3)

    def test_hamming_distance_ignore_missing(self):

        distance = dissimilarity_functions.hamming_distance(
            self.s1, self.s2, ignore_missing_state=True
        )

        self.assertEqual(distance, 2)

        distance = dissimilarity_functions.hamming_distance(
            self.s1, self.all_missing, ignore_missing_state=True
        )

        self.assertEqual(distance, 0)


if __name__ == "__main__":
    unittest.main()
