"""
Test NeighborJoiningSolver in Cassiopeia.solver.
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
            cm, dissimilarity_map=delta, root_sample="b"
        )

        # ---------------- Lineage Tracing NJ ----------------

        cm = pd.DataFrame.from_dict(
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

        delta_fn = lambda x, y, priors, missing_state: np.sum([x[i] != y[i] for i in range(len(x))])
        self.nj_pp_solver = solver.NeighborJoiningSolver(
            cm, dissimilarity_function=delta_fn
        )

        # ------------- CM with Duplictes -----------------------
        cm = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
                "f": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )
        delta_fn = lambda x, y, priors, missing_state: np.sum([x[i] != y[i] for i in range(len(x))])
        self.nj_duplicates_solver = solver.NeighborJoiningSolver(
            cm, dissimilarity_function=delta_fn
        )

    def test_constructor(self):

        delta = self.nj_solver.dissimilarity_map

        self.assertEqual(delta.shape[0], 5)
        self.assertEqual(delta.shape[1], 5)
        self.assertTrue(np.allclose(delta.values, delta.T))
        self.assertEqual(self.nj_solver.root_sample, "b")

        # testing PP Neighbor Joining solver
        delta = self.nj_pp_solver.dissimilarity_map
        expected_map = pd.DataFrame.from_dict(
            {
                "state0": [0, 1, 2, 2, 3, 2],
                "state1": [1, 0, 1, 2, 3, 2],
                "state2": [2, 1, 0, 3, 3, 3],
                "state3": [2, 2, 3, 0, 1, 1],
                "state4": [3, 3, 3, 1, 0, 2],
                "state5": [2, 2, 3, 1, 2, 0],
            },
            orient="index",
            columns=["state0", "state1", "state2", "state3", "state4", "state5"],
        )

        for i in expected_map.index:
            for j in expected_map.columns:
                self.assertEqual(expected_map.loc[i, j], delta.loc[i, j])

    def test_compute_q(self):

        q_vals = self.nj_solver.compute_q(
            self.nj_solver.dissimilarity_map.values
        )

        expected_q = pd.DataFrame.from_dict(
            {
                "state0": [0, -22.67, -22, -22, -33.33],
                "state1": [-22.67, 0, -27.33, -27.33, -22.67],
                "state2": [-22, -27.33, 0, -28.67, -22],
                "state3": [-22, -27.33, -28.67, 0, -22],
                "state4": [-33.33, -22.67, -22, -22, 0],
            },
            orient="index",
            columns=["state0", "state2", "state3", "state4", "state5"],
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

    def test_basic_solver(self):

        self.nj_solver.solve()

        T = self.nj_solver.tree

        # test leaves exist in tree
        _leaves = [n for n in T if T.out_degree(n) == 0]

        self.assertEqual(
            len(_leaves), self.nj_solver.dissimilarity_map.shape[0] - 1
        )
        for _leaf in _leaves:
            self.assertIn(_leaf, self.nj_solver.dissimilarity_map.index.values)

        # test for expected number of edges
        edges = list(T.edges())
        self.assertEqual(len(edges), 7)

        # test relationships between samples
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(["a", "b", "c", "d", "e", "5", "6", "7"])
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "e"),
                ("6", "5"),
                ("b", "6"),
                ("6", "7"),
                ("7", "d"),
                ("7", "c"),
            ]
        )

        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

        # compare tree distances
        T = T.to_undirected()
        expected_tree = expected_tree.to_undirected()
        for i in range(len(_leaves)):
            sample1 = _leaves[i]
            for j in range(i + 1, len(_leaves)):
                sample2 = _leaves[j]
                self.assertEqual(
                    nx.shortest_path_length(T, sample1, sample2),
                    nx.shortest_path_length(expected_tree, sample1, sample2),
                )

    def test_pp_solver(self):

        self.nj_pp_solver.solve()
        T = self.nj_pp_solver.tree

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "root", "6", "7", "8", "9"]
        )
        expected_tree.add_edges_from(
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

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_duplicate_sample_neighbor_joining(self):

        self.nj_duplicates_solver.solve()
        T = self.nj_duplicates_solver.tree

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "f", "root", "6", "7", "8", "9", "10"]
        )
        expected_tree.add_edges_from(
            [
                ("root", "9"),
                ("9", "8"),
                ("9", "7"),
                ("7", "6"),
                ("7", "a"),
                ("6", "b"),
                ("6", "c"),
                ("8", "10"),
                ("10", "e"),
                ("10", "f"),
                ("8", "d"),
            ]
        )


if __name__ == "__main__":
    unittest.main()
