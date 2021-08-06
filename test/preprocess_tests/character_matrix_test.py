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

        ## setup complicated allele table
        at_dict = {
            "cellBC": [
                "cellA",
                "cellA",
                "cellA",
                "cellB",
                "cellB",
                "cellC",
                "cellD",
                "cellD",
            ],
            "intBC": ["A", "B", "C", "B", "C", "A", "A", "B"],
            "r1": ["AAA", "AAB", "AAC", "AAD", "ABA", "ABB", "AAA", "AAB"],
            "r2": ["BAA", "BAB", "BAC", "BAD", "BBA", "BBB", "BAA", "BAB"],
            "r3": ["CAA", "CAB", "CAC", "CAD", "CBA", "CBB", "CAA", "CAB"],
            "UMI": [5, 10, 30, 30, 10, 10, 3, 3],
            "Mouse": ["M1", "M1", "M1", "M1", "M1", "M1", "M2", "M2"],
        }
        self.allele_table_mouse = pd.DataFrame(at_dict)

        ## set up non-cassiopeia allele table
        self.noncassiopeia_alleletable = self.alleletable_basic.copy()
        self.noncassiopeia_alleletable.rename(
            columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True
        )

        # allele table with conflicts
        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC", "cellA"],
            "intBC": ["A", "B", "C", "A", "C", "A"],
            "r1": ["None", "ATC", "GGG", "None", "GAA", "None"],
            "r2": ["None", "AAA", "GAA", "None", "GAA", "ACT"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA", "None"],
            "UMI": [5, 10, 1, 30, 30, 5],
        }
        self.alleletable_conflict = pd.DataFrame.from_dict(at_dict)

    def test_basic_character_matrix_formation(self):

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
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

    def test_character_matrix_formation_custom_missing_data(self):
        
        self.alleletable_basic.loc[0, "r1"] = "missing"

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_basic, missing_data_allele="missing", missing_data_state=-3
        )

        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 9)

        expected_df = pd.DataFrame.from_dict(
            {
                "cellA": [-3, 0, 1, 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -3, -3, -3, -3, -3, -3],
                "cellC": [-3, -3, -3, -3, -3, -3, 2, 1, 1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(1, 10)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_character_matrix_formation_with_conflicts(self):
        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_conflict
        )
        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 9)

        expected_df = pd.DataFrame.from_dict(
            {
                "cellA": [0, (0, 1), (0, 1), 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, -1, -1, -1, 2, 1, 1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(1, 10)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_character_matrix_formation_with_conflicts_no_collapse(self):
        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
            self.alleletable_conflict, collapse_duplicates=False
        )
        self.assertEqual(character_matrix.shape[0], 3)
        self.assertEqual(character_matrix.shape[1], 9)

        expected_df = pd.DataFrame.from_dict(
            {
                "cellA": [(0, 0), (0, 1), (1, 0), 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, -1, -1, -1, 2, 1, 1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(1, 10)],
        )

        pd.testing.assert_frame_equal(character_matrix, expected_df)

    def test_ignore_intbc(self):

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
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

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
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

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
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

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
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

    def test_alleletable_to_lineage_profile_with_conflicts(self):
        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_conflict
        )

        expected_lineage_profile = pd.DataFrame.from_dict(
            {
                "cellA": [
                    "None",
                    ("ACT", "None"),
                    ("ATC", "None"),
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

    def test_alleletable_to_lineage_profile_with_conflicts_no_collapse(self):
        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_conflict, collapse_duplicates=False
        )

        expected_lineage_profile = pd.DataFrame.from_dict(
            {
                "cellA": [
                    ("None", "None"),
                    ("None", "ACT"),
                    ("ATC", "None"),
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

    def test_lineage_profile_to_character_matrix_with_conflicts(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_conflict, collapse_duplicates=False
        )

        (
            character_matrix,
            priors,
            state_to_indel,
        ) = cas.pp.convert_lineage_profile_to_character_matrix(lineage_profile)

        self.assertEqual(len(priors), 0)
        self.assertEqual(len(state_to_indel), 9)

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "cellA": [1, 1, 1, (0, 0), (0, 1), (1, 0), 1, 1, 1],
                "cellB": [-1, -1, -1, 0, 0, 2, -1, -1, -1],
                "cellC": [2, 1, 1, -1, -1, -1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix.index.name = "cellBC"
        # Behavior on ties is different depending on the numpy version. So we
        # need to check against two different expected character matrices.
        # Specifically, intBC A and C are tied.
        expected_character_matrix2 = pd.DataFrame.from_dict(
            {
                "cellA": [(0, 0), (0, 1), (1, 0), 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, 2, 1, 1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix2.index.name = "cellBC"

        try:
            pd.testing.assert_frame_equal(
                expected_character_matrix, character_matrix
            )
        except AssertionError:
            pd.testing.assert_frame_equal(
                expected_character_matrix2, character_matrix
            )

    def test_lineage_profile_to_character_matrix_no_priors(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_basic
        )

        (
            character_matrix,
            priors,
            state_to_indel,
        ) = cas.pp.convert_lineage_profile_to_character_matrix(lineage_profile)

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
        # Behavior on ties is different depending on the numpy version. So we
        # need to check against two different expected character matrices.
        # Specifically, intBC A and C are tied.
        expected_character_matrix2 = pd.DataFrame.from_dict(
            {
                "cellA": [0, 0, 1, 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, 2, 1, 1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix2.index.name = "cellBC"

        try:
            pd.testing.assert_frame_equal(
                expected_character_matrix, character_matrix
            )
        except AssertionError:
            pd.testing.assert_frame_equal(
                expected_character_matrix2, character_matrix
            )

    def test_lineage_profile_to_character_matrix_with_priors(self):

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
        # Behavior on ties is different depending on the numpy version. So we
        # need to check against two different expected character matrices.
        # Specifically, intBC A and C are tied.
        expected_character_matrix2 = pd.DataFrame.from_dict(
            {
                "cellA": [0, 0, 1, 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, 2, 1, 1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix2.index.name = "cellBC"

        try:
            pd.testing.assert_frame_equal(
                expected_character_matrix, character_matrix
            )
        except AssertionError:
            pd.testing.assert_frame_equal(
                expected_character_matrix2, character_matrix
            )

        # test prior dictionary formation
        for character in priors.keys():
            for state in priors[character].keys():
                indel = state_to_indel[character][state]
                prob = self.mutation_priors.loc[indel].iloc[0]
                self.assertEqual(prob, priors[character][state])

    def test_compute_empirical_indel_probabilities(self):

        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            self.alleletable_basic
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "GGG": [1, 1 / 3],
                "GAA": [1, 1 / 3],
                "ATA": [2, 2 / 3],
                "ATC": [2, 2 / 3],
                "AAA": [1, 1 / 3],
                "TTT": [1, 1 / 3],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_compute_empirical_indel_probabilities_with_conflicts(self):

        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            self.alleletable_conflict
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "ATC": [2, 2 / 3],
                "GGG": [1, 1 / 3],
                "GAA": [1, 1 / 3],
                "AAA": [1, 1 / 3],
                "ACT": [1, 1 / 3],
                "TTT": [1, 1 / 3],
                "ATA": [2, 2 / 3],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_compute_empirical_indel_probabilities_multiple_variables(self):

        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            self.allele_table_mouse, grouping_variables=["Mouse", "intBC"]
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "AAB": [2, 2 / 5],
                "BAB": [2, 2 / 5],
                "CAB": [2, 2 / 5],
                "AAA": [2, 2 / 5],
                "BAA": [2, 2 / 5],
                "CAA": [2, 2 / 5],
                "AAC": [1, 1 / 5],
                "BAC": [1, 1 / 5],
                "CAC": [1, 1 / 5],
                "AAD": [1, 1 / 5],
                "BAD": [1, 1 / 5],
                "CAD": [1, 1 / 5],
                "ABA": [1, 1 / 5],
                "BBA": [1, 1 / 5],
                "CBA": [1, 1 / 5],
                "ABB": [1, 1 / 5],
                "BBB": [1, 1 / 5],
                "CBB": [1, 1 / 5],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

        # make sure permuting grouping variables doesn't change result
        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            self.allele_table_mouse, grouping_variables=["intBC", "Mouse"]
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_noncanonical_cut_sites_allele_table_to_character_matrix(self):

        (
            character_matrix,
            priors,
            indel_states,
        ) = cas.pp.convert_alleletable_to_character_matrix(
            self.noncassiopeia_alleletable, cut_sites=["cs1", "cs2", "cs3"]
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

    def test_noncanonical_cut_sites_allele_table_to_lineage_profile(self):

        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.noncassiopeia_alleletable, cut_sites=["cs1", "cs2", "cs3"]
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
                "A_cs1",
                "A_cs2",
                "A_cs3",
                "B_cs1",
                "B_cs2",
                "B_cs3",
                "C_cs1",
                "C_cs2",
                "C_cs3",
            ],
        )
        expected_lineage_profile.index.name = "cellBC"

        pd.testing.assert_frame_equal(
            expected_lineage_profile,
            lineage_profile[expected_lineage_profile.columns],
        )

    def test_compute_empirical_indel_probabilities_multiple_variables_noncassiopeia_alleletable(
        self,
    ):

        noncassiopeia_at = self.allele_table_mouse.copy()
        noncassiopeia_at.rename(
            columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True
        )
        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            noncassiopeia_at,
            grouping_variables=["Mouse", "intBC"],
            cut_sites=["cs1", "cs2", "cs3"],
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "AAB": [2, 2 / 5],
                "BAB": [2, 2 / 5],
                "CAB": [2, 2 / 5],
                "AAA": [2, 2 / 5],
                "BAA": [2, 2 / 5],
                "CAA": [2, 2 / 5],
                "AAC": [1, 1 / 5],
                "BAC": [1, 1 / 5],
                "CAC": [1, 1 / 5],
                "AAD": [1, 1 / 5],
                "BAD": [1, 1 / 5],
                "CAD": [1, 1 / 5],
                "ABA": [1, 1 / 5],
                "BBA": [1, 1 / 5],
                "CBA": [1, 1 / 5],
                "ABB": [1, 1 / 5],
                "BBB": [1, 1 / 5],
                "CBB": [1, 1 / 5],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

        # make sure permuting grouping variables doesn't change result
        indel_probabilities = cas.pp.compute_empirical_indel_priors(
            self.allele_table_mouse, grouping_variables=["intBC", "Mouse"]
        )

        for indel in expected_priors.index:

            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_lineage_profile_to_character_matrix_custom_missing_data(self):

        self.alleletable_basic.fillna("MISSING", inplace=True)
        lineage_profile = cas.pp.convert_alleletable_to_lineage_profile(
            self.alleletable_basic,
        )

        (
            character_matrix,
            priors,
            state_to_indel,
        ) = cas.pp.convert_lineage_profile_to_character_matrix(
            lineage_profile, self.mutation_priors, missing_allele_indicator="MISSING",
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
        # Behavior on ties is different depending on the numpy version. So we
        # need to check against two different expected character matrices.
        # Specifically, intBC A and C are tied.
        expected_character_matrix2 = pd.DataFrame.from_dict(
            {
                "cellA": [0, 0, 1, 1, 1, 1, 1, 1, 1],
                "cellB": [0, 0, 2, -1, -1, -1, -1, -1, -1],
                "cellC": [-1, -1, -1, 2, 1, 1, -1, -1, -1],
            },
            orient="index",
            columns=[f"r{i}" for i in range(9)],
        )
        expected_character_matrix2.index.name = "cellBC"

        try:
            pd.testing.assert_frame_equal(
                expected_character_matrix, character_matrix
            )
        except AssertionError:
            pd.testing.assert_frame_equal(
                expected_character_matrix2, character_matrix
            )

        # test prior dictionary formation
        for character in priors.keys():
            for state in priors[character].keys():
                indel = state_to_indel[character][state]
                prob = self.mutation_priors.loc[indel].iloc[0]
                self.assertEqual(prob, priors[character][state])


if __name__ == "__main__":
    unittest.main()
