"""
Test suite for the topology functions in
cassiopeia/tools/topology.py
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.mixins import CassiopeiaError, CassiopeiaTreeError
from cassiopeia.tools import topology


class TestTopology(unittest.TestCase):
    def setUp(self) -> None:

        tree = nx.DiGraph()
        tree.add_edge("0", "1")
        tree.add_edge("0", "2")
        tree.add_edge("1", "3")
        tree.add_edge("1", "4")
        tree.add_edge("1", "5")
        tree.add_edge("2", "6")
        tree.add_edge("2", "7")
        tree.add_edge("3", "8")
        tree.add_edge("3", "9")
        tree.add_edge("7", "10")
        tree.add_edge("7", "11")
        tree.add_edge("8", "12")
        tree.add_edge("8", "13")
        tree.add_edge("9", "14")
        tree.add_edge("9", "15")
        tree.add_edge("3", "16")
        tree.add_edge("16", "17")
        tree.add_edge("16", "18")

        character_matrix = pd.DataFrame.from_dict(
            {
                "12": [1, 2, 1, 1],
                "13": [1, 2, 1, 0],
                "14": [1, 0, 1, 0],
                "15": [1, 5, 1, 0],
                "17": [1, 4, 1, 0],
                "18": [1, 4, 1, 5],
                "6": [2, 0, 0, 0],
                "10": [2, 3, 0, 0],
                "11": [2, 3, 1, 0],
                "4": [1, 0, 0, 0],
                "5": [1, 5, 0, 0],
            },
            orient="index",
        )

        self.tree = cas.data.CassiopeiaTree(
            character_matrix=character_matrix, tree=tree
        )

    def test_simple_choose_function(self):

        num_choices = topology.nCk(10, 2)

        self.assertEqual(num_choices, 45)

        self.assertRaises(CassiopeiaError, topology.nCk, 5, 7)

    def test_simple_coalescent_probability(self):

        N = 100
        B = 2
        K = 60
        coalescent_probability = topology.simple_coalescent_probability(N, B, K)
        self.assertAlmostEqual(coalescent_probability, 0.24, delta=0.01)

        self.assertRaises(
            CassiopeiaError, topology.simple_coalescent_probability, 50, 2, 60
        )

    def test_expansion_probability(self):

        # make sure attributes are instantiated correctly
        cas.tl.compute_expansion_pvalues(self.tree, min_clade_size=20)
        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            self.assertEqual(
                1.0, self.tree.get_attribute(node, "expansion_pvalue")
            )

        cas.tl.compute_expansion_pvalues(self.tree, min_clade_size=2)
        expected_probabilities = {
            "0": 1.0,
            "1": 0.3,
            "2": 0.8,
            "3": 0.047,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0,
            "7": 0.5,
            "8": 0.6,
            "9": 0.6,
            "10": 1.0,
            "11": 1.0,
            "12": 1.0,
            "13": 1.0,
            "14": 1.0,
            "15": 1.0,
            "16": 0.6,
            "17": 1.0,
            "18": 1.0,
        }

        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            expected = expected_probabilities[node]
            self.assertAlmostEqual(
                expected,
                self.tree.get_attribute(node, "expansion_pvalue"),
                delta=0.01,
            )

    def test_expansion_probability_variable_depths(self):

        cas.tl.compute_expansion_pvalues(
            self.tree, min_clade_size=2, min_depth=3
        )
        expected_probabilities = {
            "0": 1.0,
            "1": 1.0,
            "2": 1.0,
            "3": 1.0,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0,
            "7": 1.0,
            "8": 0.6,
            "9": 0.6,
            "10": 1.0,
            "11": 1.0,
            "12": 1.0,
            "13": 1.0,
            "14": 1.0,
            "15": 1.0,
            "16": 0.6,
            "17": 1.0,
            "18": 1.0,
        }

        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            expected = expected_probabilities[node]
            self.assertAlmostEqual(
                expected,
                self.tree.get_attribute(node, "expansion_pvalue"),
                delta=0.01,
            )

    def test_expansion_probability_copy_tree(self):

        tree = cas.tl.compute_expansion_pvalues(
            self.tree, min_clade_size=2, min_depth=1, copy=True
        )

        expected_probabilities = {
            "0": 1.0,
            "1": 0.3,
            "2": 0.8,
            "3": 0.047,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0,
            "7": 0.5,
            "8": 0.6,
            "9": 0.6,
            "10": 1.0,
            "11": 1.0,
            "12": 1.0,
            "13": 1.0,
            "14": 1.0,
            "15": 1.0,
            "16": 0.6,
            "17": 1.0,
            "18": 1.0,
        }

        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            expected_copy = expected_probabilities[node]

            self.assertAlmostEqual(
                expected_copy,
                tree.get_attribute(node, "expansion_pvalue"),
                delta=0.01,
            )

            self.assertRaises(
                CassiopeiaTreeError,
                self.tree.get_attribute,
                node,
                "expansion_pvalue",
            )

    def test_cophenetic_correlation_perfect(self):

        custom_dissimilarity_map = pd.DataFrame.from_dict(
            {
                "12": [0, 2, 4, 4, 4, 4, 4, 4, 6, 7, 7],
                "13": [2, 0, 4, 4, 4, 4, 4, 4, 6, 7, 7],
                "14": [4, 4, 0, 2, 4, 4, 4, 4, 6, 7, 7],
                "15": [4, 4, 2, 0, 4, 4, 4, 4, 6, 7, 7],
                "17": [4, 4, 4, 4, 0, 2, 4, 4, 6, 7, 7],
                "18": [4, 4, 4, 4, 2, 0, 4, 4, 6, 7, 7],
                "4": [4, 4, 4, 4, 4, 4, 0, 2, 4, 5, 5],
                "5": [4, 4, 4, 4, 4, 4, 2, 0, 4, 5, 5],
                "6": [6, 6, 6, 6, 6, 6, 4, 4, 0, 3, 3],
                "10": [7, 7, 7, 7, 7, 7, 5, 5, 3, 0, 2],
                "11": [7, 7, 7, 7, 7, 7, 5, 5, 3, 2, 0],
            },
            orient="index",
            columns=[
                "12",
                "13",
                "14",
                "15",
                "17",
                "18",
                "4",
                "5",
                "6",
                "10",
                "11",
            ],
        )

        obs_cophenetic_correlation = cas.tl.compute_cophenetic_correlation(
            self.tree, dissimilarity_map=custom_dissimilarity_map
        )
        self.assertEquals(1.0, obs_cophenetic_correlation)

        # make sure weight matrix can be specified
        W = pd.DataFrame.from_dict(
            {
                "12": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "13": [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "14": [2, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10],
                "15": [3, 3, 3, 0, 4, 5, 6, 7, 8, 9, 10],
                "17": [4, 4, 4, 4, 0, 5, 6, 7, 8, 9, 10],
                "18": [5, 5, 5, 5, 5, 0, 6, 7, 8, 9, 10],
                "4": [6, 6, 6, 6, 6, 6, 0, 7, 8, 9, 10],
                "5": [7, 7, 7, 7, 7, 7, 7, 0, 8, 9, 10],
                "6": [8, 8, 8, 8, 8, 8, 8, 8, 0, 9, 10],
                "10": [9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 10],
                "11": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0],
            },
            orient="index",
            columns=[
                "12",
                "13",
                "14",
                "15",
                "17",
                "18",
                "4",
                "5",
                "6",
                "10",
                "11",
            ],
        )
        obs_cophenetic_correlation = cas.tl.compute_cophenetic_correlation(
            self.tree, weights=W, dissimilarity_map=W
        )
        self.assertAlmostEqual(1.0, obs_cophenetic_correlation, delta=1e-6)

    def test_cophenetic_correlation_no_input(self):

        obs_cophenetic_correlation = cas.tl.compute_cophenetic_correlation(
            self.tree
        )

        expected_correlation = 0.819

        self.assertAlmostEqual(
            expected_correlation, obs_cophenetic_correlation, delta=0.001
        )


if __name__ == "__main__":
    unittest.main()
