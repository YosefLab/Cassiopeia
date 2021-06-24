"""
Tests for the Layers class in the data module.
"""
import unittest

import ete3
import itertools
import networkx as nx
import numpy as np
from numpy.testing._private.utils import assert_equal
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities


def find_triplet_structure(triplet, T):
    a, b, c = triplet[0], triplet[1], triplet[2]
    a_ancestors = [node for node in nx.ancestors(T, a)]
    b_ancestors = [node for node in nx.ancestors(T, b)]
    c_ancestors = [node for node in nx.ancestors(T, c)]
    ab_common = len(set(a_ancestors) & set(b_ancestors))
    ac_common = len(set(a_ancestors) & set(c_ancestors))
    bc_common = len(set(b_ancestors) & set(c_ancestors))
    structure = "-"
    if ab_common > bc_common and ab_common > ac_common:
        structure = "ab"
    elif ac_common > bc_common and ac_common > ab_common:
        structure = "ac"
    elif bc_common > ab_common and bc_common > ac_common:
        structure = "bc"
    return structure


class TestCassiopeiaTree(unittest.TestCase):
    def setUp(self):

        character_matrix = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )
        topology = nx.DiGraph()
        topology.add_nodes_from(
            ["a", "b", "c", "d", "e", "root", "6", "7", "8", "9"]
        )
        topology.add_edges_from(
            [
                ("root", "9"),
                ("9", "8"),
                ("9", "7"),
                ("7", "6"),
                ("7", "a"),
                ("6", "b"),
                ("6", "c"),
                ("8", "e"),
                ("8", "d"),
            ]
        )

        self.base_character_matrix = character_matrix
        self.base_tree_topology = topology

        self.tree = cas.data.CassiopeiaTree(
            character_matrix=character_matrix, tree=topology
        )

    def test_basic_character_matrix(self):

        character_matrix = self.tree.character_matrix

        pd.testing.assert_frame_equal(
            character_matrix, self.base_character_matrix
        )

    def test_add_layer(self):

        character_matrix = self.tree.character_matrix.copy()
        character_matrix.loc["a"] = [0, 0, 0]

        self.tree.layers["modified"] = character_matrix

        modified_character_matrix = self.tree.layers["modified"]
        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "a": [0, 0, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        pd.testing.assert_frame_equal(
            modified_character_matrix, expected_character_matrix
        )

        # test layer updates
        self.assertListEqual([1, 2, 0], self.tree.get_character_states("b"))
        self.tree.set_character_states("b", [1, 0, 0], layer="modified")
        self.assertListEqual([1, 0, 0], self.tree.get_character_states("b"))

        # test that base character matrix is unchanged when we change something
        # in another layer
        layer_character_matrix = self.tree.layers["modified"]
        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "a": [0, 0, 0],
                "b": [1, 0, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        pd.testing.assert_frame_equal(
            layer_character_matrix, expected_character_matrix
        )

        pd.testing.assert_frame_equal(
            self.tree.character_matrix, self.base_character_matrix
        )

    def test_reconstruct_tree_with_layers(self):

        # first reconstruct basic tree
        greedy_solver = cas.solver.VanillaGreedySolver()
        greedy_solver.solve(self.tree)

        reconstructed_tree = self.tree.get_tree_topology()
        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(
                triplet, self.base_tree_topology
            )
            observed_triplet = find_triplet_structure(
                triplet, reconstructed_tree
            )
            self.assertEqual(expected_triplet, observed_triplet)

        # add layer and reconstruct tree
        modified_character_matrix = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [1, 0, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )
        self.tree.layers["modified"] = modified_character_matrix
        greedy_solver.solve(self.tree, layer="modified")

        expected_tree_topology = nx.DiGraph()
        expected_tree_topology.add_nodes_from(
            ["a", "b", "c", "d", "e", "root", "1", "2"]
        )
        expected_tree_topology.add_edges_from(
            [
                ("root", "d"),
                ("root", "1"),
                ("1", "e"),
                ("1", "a"),
                ("1", "2"),
                ("2", "c"),
                ("2", "b"),
            ]
        )
        reconstructed_tree = self.tree.get_tree_topology()

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(
                triplet, expected_tree_topology
            )
            observed_triplet = find_triplet_structure(
                triplet, reconstructed_tree
            )
            self.assertEqual(expected_triplet, observed_triplet)

        # make sure character states get set correctly
        self.assertEqual([1, 0, 0], self.tree.get_character_states("e"))

    def test_add_layer_with_incorrect_number_of_cells(self):

        modified_character_matrix = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [1, 0, 0],
                "f": [2, 2, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        def add_layer_fn(matrix):
            self.tree.layers["modified"] = matrix

        self.assertRaises(ValueError, add_layer_fn, modified_character_matrix)

    def test_add_layer_with_different_number_of_characters(self):

        modified_character_matrix = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0, 1],
                "b": [1, 2, 0, 5],
                "c": [1, 2, 1, 6],
                "d": [2, 0, 0, 2],
                "e": [1, 0, 0, 3],
            },
            orient="index",
            columns=["x1", "x2", "x3", "x4"],
        )

        self.tree.layers["modified"] = modified_character_matrix

        self.assertListEqual(
            [1, 2, 0, 5], self.tree.layers["modified"].loc["b"].tolist()
        )

        # make sure states change
        self.assertListEqual([1, 2, 0], self.tree.get_character_states("b"))
        self.tree.reinitialize_character_states_at_leaves(layer="modified")
        self.assertListEqual([1, 2, 0, 5], self.tree.get_character_states("b"))


if __name__ == "__main__":
    unittest.main()
