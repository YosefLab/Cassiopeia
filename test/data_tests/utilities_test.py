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

            self.assertEqual(len(bootstrap_priors), character_matrix.shape[1])

    def test_remove_and_prune_lineage_all(self):
        """Tests the case where all lineages are removed and pruned."""
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
        for i in leaves:
            data_utilities.remove_and_prune_lineage(i, tree)

        self.assertEqual(list(tree.edges), [])

    def test_remove_and_prune_lineage_some(self):
        """Tests a case where some lineages are removed"""
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        data_utilities.remove_and_prune_lineage(11, tree)
        data_utilities.remove_and_prune_lineage(13, tree)
        data_utilities.remove_and_prune_lineage(14, tree)

        expected_edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 7),
            (3, 8),
            (4, 9),
            (4, 10),
            (5, 12),
        ]
        self.assertEqual(list(tree.edges), expected_edges)

    def test_remove_and_prune_lineage_one_side(self):
        """Tests a case where the entire one side of a tree is removed."""
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        for i in range(7, 11):
            data_utilities.remove_and_prune_lineage(i, tree)

        expected_edges = [
            (0, 2),
            (2, 5),
            (2, 6),
            (5, 11),
            (5, 12),
            (6, 13),
            (6, 14),
        ]
        self.assertEqual(list(tree.edges), expected_edges)

    def test_collapse_unifurcations_source(self):
        """Tests a case where a non-root source is provided."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(6)))
        tree.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        data_utilities.collapse_unifurcations(tree, source=1)

        expected_edges = [
            (0, 1, {"weight": 1.5}),
            (1, 4, {"weight": 3.0}),
            (1, 5, {"weight": 4.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)

    def test_collapse_unifurcations(self):
        """Tests a general case with unifurcations throughout the tree."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(10)))
        tree.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (2, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (2, 9),
            ]
        )
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        data_utilities.collapse_unifurcations(tree)
        expected_edges = [
            (0, 1, {"weight": 1.5}),
            (0, 2, {"weight": 1.5}),
            (2, 9, {"weight": 1.5}),
            (2, 4, {"weight": 3.0}),
            (2, 6, {"weight": 3.0}),
            (6, 7, {"weight": 1.5}),
            (6, 8, {"weight": 1.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)

    def test_collapse_unifurcations_long_root_unifurcation(self):
        """Tests a case where there is a long chain at the root."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(15)))
        tree.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 6),
                (6, 7),
                (6, 8),
                (5, 9),
                (5, 10),
                (5, 11),
                (10, 12),
                (12, 13),
                (13, 14),
            ]
        )
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        data_utilities.collapse_unifurcations(tree)

        expected_edges = [
            (0, 5, {"weight": 6.0}),
            (0, 6, {"weight": 7.5}),
            (5, 9, {"weight": 1.5}),
            (5, 11, {"weight": 1.5}),
            (5, 14, {"weight": 6.0}),
            (6, 7, {"weight": 1.5}),
            (6, 8, {"weight": 1.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)


if __name__ == "__main__":
    unittest.main()
