"""
Test basic CassiopeiaSolver constructor.
"""
import os
import unittest

import itertools
import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia import solver


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


class TestNeighborJoiningSolver(unittest.TestCase):
    def setUp(self):

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
                "a": [0, 15, 21, 17, 12],
                "b": [15, 0, 10, 6, 17],
                "c": [21, 10, 0, 10, 23],
                "d": [17, 6, 10, 0, 19],
                "e": [12, 17, 23, 19, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.nj_solver = solver.NeighborJoiningSolver(
            cm, dissimilarity_map=delta
        )

    def test_constructor(self):

        delta = self.nj_solver.dissimilarity_map

        self.assertEqual(delta.shape[0], 5)
        self.assertEqual(delta.shape[1], 5)
        self.assertTrue(np.allclose(delta.values, delta.T))

    def test_compute_q(self):

        q_vals = self.nj_solver.compute_q(
            self.nj_solver.dissimilarity_map.values
        )

        expected_q = pd.DataFrame.from_dict(
            {
                "a": [0, -22.67, -22, -22, -33.33],
                "b": [-22.67, 0, -27.33, -27.33, -22.67],
                "c": [-22, -27.33, 0, -28.67, -22],
                "d": [-22, -27.33, -28.67, 0, -22],
                "e": [-33.33, -22.67, -22, -22, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.assertTrue(np.allclose(q_vals, expected_q, atol=0.1))

    def test_find_cherry(self):

        cherry = self.nj_solver.find_cherry(
            self.nj_solver.dissimilarity_map.values
        )
        delta = self.nj_solver.dissimilarity_map
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        self.assertIn((node_i, node_j), [("a", "e"), ("e", "a")])

    def test_update_dissimilarity_map(self):

        delta = self.nj_solver.dissimilarity_map

        cherry = self.nj_solver.find_cherry(delta.values)
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        delta = self.nj_solver.update_dissimilarity_map(
            delta, (node_i, node_j), "f"
        )

        expected_delta = pd.DataFrame.from_dict(
            {
                "f": [0, 10, 16, 12],
                "b": [10, 0, 10, 6],
                "c": [16, 10, 0, 10],
                "d": [12, 6, 10, 0],
            },
            orient="index",
            columns=["f", "b", "c", "d"],
        )

        for sample in expected_delta.index:
            for sample2 in expected_delta.index:
                self.assertEqual(
                    delta.loc[sample, sample2],
                    expected_delta.loc[sample, sample2],
                )

    def test_solver(self):

        self.nj_solver.solve()

        T = self.nj_solver.tree

        # test leaves exist in tree
        _leaves = [n for n in T if T.degree(n) == 1]
        self.assertEqual(
            len(_leaves), self.nj_solver.dissimilarity_map.shape[0]
        )
        for sample in self.nj_solver.dissimilarity_map.index.values:
            self.assertIn(sample, _leaves)

        # test for expected number of edges
        edges = list(T.edges())
        self.assertEqual(len(edges), 7)

        # test relationships between samples
        expected_tree = nx.Graph()
        expected_tree.add_nodes_from(["a", "b", "c", "d", "e", "5", "6", "7"])
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "e"),
                ("6", "5"),
                ("6", "b"),
                ("7", "6"),
                ("7", "d"),
                ("7", "c"),
            ]
        )

        self.assertEqual(
            nx.to_nested_tuple(T, "a"), nx.to_nested_tuple(expected_tree, "a")
        )

        # compare tree distances
        for i in range(len(_leaves)):
            sample1 = _leaves[i]
            for j in range(i + 1, len(_leaves)):
                sample2 = _leaves[j]
                self.assertEqual(
                    nx.shortest_path_length(T, sample1, sample2),
                    nx.shortest_path_length(expected_tree, sample1, sample2),
                )


if __name__ == "__main__":
    unittest.main()
