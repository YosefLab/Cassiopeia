"""
Tests for character matrix formation.
"""
import unittest

import numpy as np
import pandas as pd

import cassiopeia as cas


class TestCharacterMatrixFormation(unittest.TestCase):
    def setUp(self):

        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC"],
            "intBC": ["A", "B", "C", "A", "C"],
            "r1": ["None", "ATC", "GGG", "None", "GAA"],
            "r2": ["None", "AAA", "GAA", "None", "GAA"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA"],
        }

        self.alleletable_basic = pd.DataFrame.from_dict(at_dict)

        self.mutation_priors = pd.DataFrame.from_dict(
            {
                "ATC": 0.5,
                "GGG": 0.2,
                "GAA": 0.1,
                "AAA": 0.05,
                "TTT": 0.05,
                "ATA": 0.1,
            },
            orient="index",
            columns=["freq"],
        )

    def test_basic_character_matrix_formation(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic
        )

        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 9)

        expected_df = pd.DataFrame.from_dict(
            {
                "cellA": ["0", "0", "1", "1", "1", "1", "1", "1", "1"],
                "cellB": ["0", "0", "2", "-", "-", "-", "-", "-", "-"],
                "cellC": ["-", "-", "-", "-", "-", "-", "2", "1", "1"],
            },
            orient="index",
            columns=[f"r{i}" for i in range(1, 10)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_ignore_intbc(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, ignore_intbcs=["B"]
        )

        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 6)

        expected_df = pd.DataFrame.from_dict(
            {
                "cellA": ["0", "0", "1", "1", "1", "1"],
                "cellB": ["0", "0", "2", "-", "-", "-"],
                "cellC": ["-", "-", "-", "2", "1", "1"],
            },
            orient="index",
            columns=[f"r{i}" for i in range(1, 7)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_filter_out_low_diversity_intbcs(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, allele_rep_thresh=0.99
        )

        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 2)

        expected_df = pd.DataFrame.from_dict(
            {"cellA": ["1", "1"], "cellB": ["2", "-"], "cellC": ["-", "2"]},
            orient="index",
            columns=[f"r{i}" for i in range(1, 3)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_mutation_prior_formation(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, mutation_priors=self.mutation_priors
        )

        expected_priors_dictionary = {
            2: {"1": 0.5, "2": 0.1},
            3: {"1": 0.5},
            4: {"1": 0.05},
            5: {"1": 0.05},
            6: {"1": 0.2, "2": 0.1},
            7: {"1": 0.1},
            8: {"1": 0.1},
        }

        for char in expected_priors_dictionary.keys():
            for state in expected_priors_dictionary[char].keys():
                self.assertEqual(
                    priors[char][state], expected_priors_dictionary[char][state]
                )

    def test_indel_state_mapping_formation(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, mutation_priors=self.mutation_priors
        )

        expected_state_mapping_dictionary = {
            2: {"1": "ATC", "2": "ATA"},
            3: {"1": "ATC"},
            4: {"1": "AAA"},
            5: {"1": "TTT"},
            6: {"1": "GGG", "2": "GAA"},
            7: {"1": "GAA"},
            8: {"1": "ATA"},
        }

        for char in expected_state_mapping_dictionary.keys():
            for state in expected_state_mapping_dictionary[char].keys():
                self.assertEqual(
                    indel_states[char][state],
                    expected_state_mapping_dictionary[char][state],
                )


if __name__ == "__main__":
    unittest.main()
