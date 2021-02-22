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
            "UMI": [5, 10, 1, 30, 30],
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
                "cellA": [0, 0, 1, 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, -1, -1, -1, 2, 1, 1],
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
                "cellA": [0, 0, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1],
                "cellC": [-1, -1, -1, 2, 1, 1],
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
            {"cellA": [1, 1], "cellB": [2, -1], "cellC": [-1, 2]},
            orient="index",
            columns=[f"r{i}" for i in range(1, 3)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_mutation_prior_formation(self):

        character_matrix, priors, indel_states = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, mutation_priors=self.mutation_priors
        )

        expected_priors_dictionary = {
            2: {1: 0.5, 2: 0.1},
            3: {1: 0.5},
            4: {1: 0.05},
            5: {1: 0.05},
            6: {1: 0.2, 2: 0.1},
            7: {1: 0.1},
            8: {1: 0.1},
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
            2: {1: "ATC", 2: "ATA"},
            3: {1: "ATC"},
            4: {1: "AAA"},
            5: {1: "TTT"},
            6: {1: "GGG", 2: "GAA"},
            7: {1: "GAA"},
            8: {1: "ATA"},
        }

        for char in expected_state_mapping_dictionary.keys():
            for state in expected_state_mapping_dictionary[char].keys():
                self.assertEqual(
                    indel_states[char][state],
                    expected_state_mapping_dictionary[char][state],
                )

    def test_alleletable_to_lineage_profile(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_basic
        )

        expected_lineage_profile = pd.DataFrame.from_dict(
            {
                "cellA": [
                    "None",
                    "None",
                    "ATC",
                    "ATC",
                    "AAA",
                    "TTT",
                    "GGG",
                    "GAA",
                    "ATA",
                ],
                "cellB": [
                    "None",
                    "None",
                    "ATA",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "cellC": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "GAA",
                    "GAA",
                    "ATA",
                ],
            },
            orient="index",
            columns=[
                "A_r1",
                "A_r2",
                "A_r3",
                "B_r1",
                "B_r2",
                "B_r3",
                "C_r1",
                "C_r2",
                "C_r3",
            ],
        )
        expected_lineage_profile.index.name = "cellBC"

        pd.testing.assert_frame_equal(
            expected_lineage_profile,
            lineage_profile[expected_lineage_profile.columns],
        )

    def test_lineage_profile_to_character_matrix_no_priors(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_basic
        )

        character_matrix, priors, state_to_indel = cas.pp.convert_lineage_profile_to_character_matrix(
            lineage_profile
        )

        self.assertEqual(len(priors), 0)
        self.assertEqual(len(state_to_indel), 9)

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "cellA": [1, 1, 1, 0, 0, 1, 1, 1, 1],
                "cellB": [-1, -1, -1, 0, 0, 2, -1, -1, -1],
                "cellC": [2, 1, 1, -1, -1, -1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix.index.name = "cellBC"

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

    def test_lineage_profile_to_character_matrix_no_priors(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_basic
        )

        (
            character_matrix,
            priors,
            state_to_indel,
        ) = cas.pp.convert_lineage_profile_to_character_matrix(
            lineage_profile, self.mutation_priors
        )

        self.assertEqual(len(priors), 7)
        self.assertEqual(len(state_to_indel), 9)

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "cellA": [1, 1, 1, 0, 0, 1, 1, 1, 1],
                "cellB": [-1, -1, -1, 0, 0, 2, -1, -1, -1],
                "cellC": [2, 1, 1, -1, -1, -1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix.index.name = "cellBC"

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

        # test prior dictionary formation
        for character in priors.keys():
            for state in priors[character].keys():
                indel = state_to_indel[character][state]
                prob = self.mutation_priors.loc[indel].iloc[0]
                self.assertEqual(prob, priors[character][state])


if __name__ == "__main__":
    unittest.main()
