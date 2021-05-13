"""
This file tests the utilities stored in cassiopeia/data/utilities.py
"""

import unittest
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import utilities as data_utilities
from cassiopeia.preprocess import utilities as preprocessing_utilities


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

        # Test allele table
        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC"],
            "intBC": ["A", "B", "C", "A", "C"],
            "r1": ["None", "ATC", "GGG", "ATA", "GAA"],
            "r2": ["None", "AAA", "GAA", "TTT", "GAA"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA"],
            "UMI": [5, 10, 1, 30, 30],
        }

        self.allele_table = pd.DataFrame.from_dict(at_dict)
        self.indel_to_prior = pd.DataFrame.from_dict(
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

        # Test allele table without normal cassiopeia columns
        self.non_cassiopeia_allele_table = self.allele_table.copy()
        self.non_cassiopeia_allele_table.rename(
            columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True
        )

    def test_bootstrap_character_matrices_no_priors(self):

        random_state = np.random.RandomState(123431235)

        bootstrap_samples = data_utilities.sample_bootstrap_character_matrices(
            self.character_matrix, num_bootstraps=10, random_state=random_state
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
            num_bootstraps=10,
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

    def test_bootstrap_allele_tables(self):

        random_state = np.random.RandomState(123431235)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.allele_table
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.allele_table, num_bootstraps=10, random_state=random_state
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostarp_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_allele_tables_non_cassiopeia_allele_table(self):

        random_state = np.random.RandomState(123431235)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.non_cassiopeia_allele_table, cut_sites=["cs1", "cs2", "cs3"]
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.non_cassiopeia_allele_table,
            num_bootstraps=10,
            random_state=random_state,
            cut_sites=["cs1", "cs2", "cs3"],
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostrap_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.non_cassiopeia_allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_allele_tables_priors(self):

        random_state = np.random.RandomState(12345)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.allele_table
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.allele_table,
            num_bootstraps=10,
            indel_priors=self.indel_to_prior,
            random_state=random_state,
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostrap_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

            self.assertEqual(len(bootstrap_priors), character_matrix.shape[1])

    def test_to_newick_no_branch_lengths(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F","A",length=0.1)
        tree.add_edge("F","B",length=0.2)
        tree.add_edge("F","E",length=0.5)
        tree.add_edge("E","C",length=0.3)
        tree.add_edge("E","D",length=0.4)

        newick_string = data_utilities.to_newick(tree)
        self.assertEqual(newick_string, "(A,B,(C,D));")

    def test_to_newick_branch_lengths(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F","A",length=0.1)
        tree.add_edge("F","B",length=0.2)
        tree.add_edge("F","E",length=0.5)
        tree.add_edge("E","C",length=0.3)
        tree.add_edge("E","D",length=0.4)

        newick_string = data_utilities.to_newick(tree, record_branch_lengths = True)
        self.assertEqual(newick_string, "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")

    def test_lca_characters(self):
        vecs = [[1, 0, 3, 4, 5], [1, -1, -1, 3, -1], [1, 2, 3, 2, -1]]
        ret_vec = data_utilities.get_lca_characters(
            vecs, missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, [1, 0, 3, 0, 5])


if __name__ == "__main__":
    unittest.main()
