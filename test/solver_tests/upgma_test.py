"""
Test UPGMASolver in Cassiopeia.solver.
"""
import unittest
from typing import Dict

import itertools
import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver.UPGMASolver import UPGMASolver
from cassiopeia.solver import dissimilarity_functions


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


class TestUPGMASolver(unittest.TestCase):
    def setUp(self):

        # --------------------- General NJ ---------------------
        cm = pd.DataFrame.from_dict(
            {
                "a": [0, 1, 2],
                "b": [1, 1, 2],
                "c": [2, 2, 2],
                "d": [1, 1, 1],
                "e": [0, 0, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        delta = pd.DataFrame.from_dict(
            {
                "a": [0, 17, 21, 31, 23],
                "b": [17, 0, 30, 34, 21],
                "c": [21, 30, 0, 28, 39],
                "d": [31, 34, 28, 0, 43],
                "e": [23, 21, 39, 43, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.basic_dissimilarity_map = delta
        self.basic_tree = CassiopeiaTree(
            character_matrix=cm, dissimilarity_map=delta
        )

        self.upgma_solver = UPGMASolver()

        # ---------------- Lineage Tracing NJ ----------------

        pp_cm = pd.DataFrame.from_dict(
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

        self.pp_tree = CassiopeiaTree(character_matrix=pp_cm)

        self.upgma_solver_delta = UPGMASolver(
            dissimilarity_function=dissimilarity_functions.weighted_hamming_distance
        )

        # ------------- CM with Duplicates and Missing Data -----------------------
        duplicates_cm = pd.DataFrame.from_dict(
            {
                "a": [1, -1, 0],
                "b": [1, 2, 1],
                "c": [1, -1, 1],
                "d": [2, 0, -1],
                "e": [2, 0, 2],
                "f": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.duplicate_tree = CassiopeiaTree(character_matrix=duplicates_cm)

        # -------------  Hamming dissimilarity with weights  ------------
        priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
        self.pp_tree_priors = CassiopeiaTree(
            character_matrix=pp_cm, priors=priors
        )
        self.upgma_solver_modified = UPGMASolver(
            dissimilarity_function=dissimilarity_functions.weighted_hamming_distance
        )

    def test_constructor(self):

        self.assertIsNotNone(self.upgma_solver_delta.dissimilarity_function)
        self.assertIsNotNone(self.basic_tree.get_dissimilarity_map())

    def test_find_cherry(self):

        cherry = self.upgma_solver.find_cherry(
            self.basic_dissimilarity_map.values
        )
        delta = self.basic_dissimilarity_map
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        self.assertIn((node_i, node_j), [("a", "b"), ("b", "a")])

    def test_update_dissimilarity_map(self):

        delta = self.basic_dissimilarity_map

        cherry = self.upgma_solver.find_cherry(delta.values)
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        delta = self.upgma_solver.update_dissimilarity_map(
            delta, (node_i, node_j), "ab"
        )

        expected_delta = pd.DataFrame.from_dict(
            {
                "ab": [0, 25.5, 32.5, 22],
                "c": [25.5, 0, 28, 39],
                "d": [32.5, 28, 0, 43],
                "e": [22, 39, 43, 0],
            },
            orient="index",
            columns=["ab", "c", "d", "e"],
        )

        for sample in expected_delta.index:
            for sample2 in expected_delta.index:
                self.assertEqual(
                    delta.loc[sample, sample2],
                    expected_delta.loc[sample, sample2],
                )

        cherry = self.upgma_solver.find_cherry(delta.values)
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        delta = self.upgma_solver.update_dissimilarity_map(
            delta, (node_i, node_j), "abe"
        )

        expected_delta = pd.DataFrame.from_dict(
            {"abe": [0, 30, 36], "c": [30, 0, 28], "d": [36, 28, 0]},
            orient="index",
            columns=["abe", "c", "d"],
        )

        for sample in expected_delta.index:
            for sample2 in expected_delta.index:
                self.assertEqual(
                    delta.loc[sample, sample2],
                    expected_delta.loc[sample, sample2],
                )

    def test_basic_solver(self):

        self.upgma_solver.solve(self.basic_tree)

        # test leaves exist in tree
        _leaves = self.basic_tree.leaves

        self.assertEqual(len(_leaves), self.basic_dissimilarity_map.shape[0])
        for _leaf in _leaves:
            self.assertIn(_leaf, self.basic_dissimilarity_map.index.values)

        # test for expected number of edges
        edges = list(self.basic_tree.edges)
        self.assertEqual(len(edges), 8)

        # test relationships between samples
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "b"),
                ("6", "5"),
                ("6", "e"),
                ("7", "c"),
                ("7", "d"),
                ("root", "6"),
                ("root", "7"),
            ]
        )

        observed_tree = self.basic_tree.get_tree_topology()
        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        # compare tree distances
        observed_tree = observed_tree.to_undirected()
        expected_tree = expected_tree.to_undirected()
        for i in range(len(_leaves)):
            sample1 = _leaves[i]
            for j in range(i + 1, len(_leaves)):
                sample2 = _leaves[j]
                self.assertEqual(
                    nx.shortest_path_length(observed_tree, sample1, sample2),
                    nx.shortest_path_length(expected_tree, sample1, sample2),
                )

    def test_upgma_solver_weights(self):
        self.upgma_solver_modified.solve(self.pp_tree_priors)
        initial_d_map = self.pp_tree_priors.get_dissimilarity_map()
        expected_dissimilarity = (-np.log(0.2) - np.log(0.8)) / 3
        self.assertEqual(initial_d_map.loc["a", "b"], expected_dissimilarity)

        observed_tree = self.pp_tree_priors.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("root", "a"),
                ("root", "7"),
                ("7", "8"),
                ("7", "9"),
                ("8", "d"),
                ("8", "e"),
                ("9", "b"),
                ("9", "c"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        self.upgma_solver_modified.solve(
            self.pp_tree_priors, collapse_mutationless_edges=True
        )
        observed_tree = self.pp_tree_priors.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("root", "a"),
                ("root", "8"),
                ("root", "9"),
                ("8", "d"),
                ("8", "e"),
                ("9", "b"),
                ("9", "c"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_pp_solver(self):
        self.upgma_solver_delta.solve(self.pp_tree)
        initial_d_map = self.pp_tree.get_dissimilarity_map()
        expected_dissimilarity = 1 / 3
        self.assertEqual(initial_d_map.loc["d", "e"], expected_dissimilarity)

        observed_tree = self.pp_tree.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("root", "8"),
                ("root", "7"),
                ("9", "7"),
                ("7", "6"),
                ("7", "a"),
                ("6", "b"),
                ("6", "c"),
                ("8", "e"),
                ("8", "d"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        self.upgma_solver_delta.solve(self.pp_tree)
        observed_tree = self.pp_tree.get_tree_topology()
        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_duplicate(self):
        # In this case, we see that the missing data can break up a duplicate
        # pair if the behavior is to ignore missing data

        self.upgma_solver_delta.solve(self.duplicate_tree)
        observed_tree = self.duplicate_tree.get_tree_topology()
        initial_d_map = self.duplicate_tree.get_dissimilarity_map()
        expected_dissimilarity = 1.5
        self.assertEqual(initial_d_map.loc["b", "d"], expected_dissimilarity)

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("root", "9"),
                ("root", "8"),
                ("9", "a"),
                ("9", "6"),
                ("6", "b"),
                ("6", "c"),
                ("8", "7"),
                ("8", "f"),
                ("7", "d"),
                ("7", "e"),
            ]
        )
        triplets = itertools.combinations(["a", "b", "c", "d", "e", "f"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
