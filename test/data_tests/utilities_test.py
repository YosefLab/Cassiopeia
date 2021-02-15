"""
This file tests the utilities stored in cassiopeia/data/utilities.py
"""

import unittest
from typing import Dict, Optional

import numpy as np
import pandas as pd

from cassiopeia.data import utilities as data_utilities


class TestDataUtilities(unittest.TestCase):
    def setUp(self):

        # this should obey PP for easy checking of ancestral states
        self.character_matrix = pd.DataFrame.from_dict(
            {
                "node3": [1, 0, 0, 0, 0, 0, 0, 0],
                "node7": [1, 1, 0, 0, 0, 0, 0, 0],
                "node9": [1, 1, 1, 0, 0, 0, 0, 0],
                "node11": [1, 1, 1, 1, 0, 0, 0, 0],
                "node13": [1, 1, 1, 1, 1, 0, 0, 0],
                "node15": [1, 1, 1, 1, 1, 1, 0, 0],
                "node17": [1, 1, 1, 1, 1, 1, 1, 0],
                "node18": [1, 1, 1, 1, 1, 1, 1, 1],
                "node5": [2, 0, 0, 0, 0, 0, 0, 0],
                "node6": [2, 2, 0, 0, 0, 0, 0, 0],
            },
            orient="index",
        )

        self.priors = {
            0: {1: 0.5, 2: 0.5},
            1: {1: 0.4, 2: 0.6},
            2: {1: 1.0},
            3: {1: 1.0},
            4: {1: 1.0},
            5: {1: 1.0},
            6: {1: 1.0},
            7: {1: 1.0},
        }

    def test_bootstrap_character_matrices_no_priors(self):

        random_state = np.random.RandomState(123431235)

        bootstrap_samples = data_utilities.sample_bootstrap_character_matrices(
            self.character_matrix, B=10, random_state=random_state
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (bootstrap_matrix, bootstrap_priors) in bootstrap_samples:
            self.assertCountEqual(
                self.character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                self.character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                self.character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_character_matrices_with_priors(self):

        random_state = np.random.RandomState(12345)

        bootstrap_samples = data_utilities.sample_bootstrap_character_matrices(
            self.character_matrix,
            B=10,
            prior_probabilities=self.priors,
            random_state=random_state,
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (bootstrap_matrix, bootstrap_priors) in bootstrap_samples:
            self.assertCountEqual(
                self.character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                self.character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                self.character_matrix,
                bootstrap_matrix,
            )

            self.assertEqual(
                len(bootstrap_priors), self.character_matrix.shape[1]
            )


if __name__ == "__main__":
    unittest.main()
