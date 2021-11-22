"""
Test SpectralNeighborJoiningSolver in Cassiopeia.solver.
"""
import unittest
from typing import List
from unittest import mock
import itertools
import networkx as nx
from networkx.classes.digraph import DiGraph
import numpy as np
import pandas as pd

import cassiopeia as cas


def find_triplet_structure(triplet, T):
    """Identify the two nodes with the most similar ancestry.

    Args:
        triplet: name of nodes to check
        T: tree in which the nodes are in

    Returns:
        The two nodes that have the most similar number of
            common ancestors.
    """
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


def assertTripletCorrectness(
    self, nodes: List[str], expected_tree: DiGraph, observed_tree: DiGraph
):
    """Checks if two trees are isomorphic.

    Args:
        nodes (List[str]): List of leaf nodes to get triplets from.
        expected_tree (DiGraph): expected tree that contains the leaf nodes.
        observed_tree (DiGraph): observed tree that contain the same leaf nodes.
    """
    # generate triplets
    triplets = itertools.combinations(nodes, 3)

    # check each triplet
    for triplet in triplets:
        expected_triplet = find_triplet_structure(triplet, expected_tree)
        observed_triplet = find_triplet_structure(triplet, observed_tree)
        self.assertEqual(expected_triplet, observed_triplet)


class TestSpectralNeighborJoiningSolver(unittest.TestCase):
    def setUp(self):
        """Instantiate instance variables for repeated use in tests."""
        self.snj_solver = cas.solver.SpectralNeighborJoiningSolver(
            add_root=True
        )

        # --------------------- General SNJ ---------------------
        cm_general = pd.DataFrame.from_dict(
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

        sim_general = pd.DataFrame(
            [
                [0.0, 0.51341712, 0.36787944, 0.36787944, 0.26359714],
                [0.51341712, 0.0, 0.71653131, 0.36787944, 0.26359714],
                [0.36787944, 0.71653131, 0.0, 0.26359714, 0.1888756],
                [0.36787944, 0.36787944, 0.26359714, 0.0, 0.71653131],
                [0.26359714, 0.26359714, 0.1888756, 0.71653131, 0.0],
            ],
            index=["a", "b", "c", "d", "e"],
            columns=["a", "b", "c", "d", "e"],
        )

        self.cm_general = cm_general
        self.sim_general = sim_general
        self.tree_general = cas.data.CassiopeiaTree(
            character_matrix=cm_general,
            dissimilarity_map=sim_general,
            root_sample_name="b",
        )

        # ---------------- Perfect Phylogeny ----------------
        cm_pp = pd.DataFrame.from_dict(
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

        self.tree_pp = cas.data.CassiopeiaTree(character_matrix=cm_pp)

        # ---------------- SNJ Integration Test 1 ----------------
        self.cm_int1 = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 1, 0],
                "b": [1, 1, 2, 1],
                "c": [1, 1, 2, 2],
                "d": [1, 2, 0, 0],
                "e": [2, 3, 0, 0],
                "f": [2, 4, 3, 0],
                "g": [2, 4, 4, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3", "x4"],
        )

        self.tree_int1 = cas.data.CassiopeiaTree(character_matrix=self.cm_int1)

        # --------------- SNJ Integration Test 2 ----------------
        cm_int2 = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 3, 3, 1],
                "b": [1, 1, 3, 3, 2],
                "c": [1, 1, 3, 4, 0],
                "d": [1, 1, 4, 0, 0],
                "e": [1, 2, 0, 0, 0],
                "f": [2, 3, 2, 1, 0],
                "g": [2, 3, 2, 2, 0],
                "h": [2, 3, 1, 0, 0],
                "i": [2, 4, 6, 0, 0],
                "j": [2, 4, 5, 0, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3", "x4", "x5"],
        )

        self.tree_int2 = cas.data.CassiopeiaTree(character_matrix=cm_int2)

        # ------------- Duplicate Nodes -----------------------
        cm_dupe = pd.DataFrame.from_dict(
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

        self.tree_dupe = cas.data.CassiopeiaTree(character_matrix=cm_dupe)

        # ------- Perfect Phylogenetic Tree + Priors --------
        priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
        self.tree_pp_priors = cas.data.CassiopeiaTree(
            character_matrix=cm_pp, priors=priors
        )

    def test_constructor(self):
        """Test for errors related to rooting trees."""
        self.assertIsNotNone(self.snj_solver.dissimilarity_function)
        self.assertIsNotNone(self.tree_general.get_dissimilarity_map())

        nothing_solver = cas.solver.SpectralNeighborJoiningSolver(
            similarity_function=None, add_root=False
        )

        no_root_tree = cas.data.CassiopeiaTree(
            character_matrix=self.cm_general,
            dissimilarity_map=self.sim_general,
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver.solve(no_root_tree)

        no_root_solver = cas.solver.SpectralNeighborJoiningSolver(
            similarity_function=None, add_root=True
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            no_root_solver.solve(no_root_tree)

        root_only_tree = cas.data.CassiopeiaTree(
            character_matrix=self.cm_general, root_sample_name="b"
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver.solve(root_only_tree)

        snj_solver_fn = cas.solver.SpectralNeighborJoiningSolver(add_root=True)
        snj_solver_fn.solve(self.tree_general)

        self.assertEqual(
            self.tree_general.get_dissimilarity_map().loc["a", "b"], 0.51341712
        )

    def test_compute_svd2_pairwise(self):
        """Test lambda matrix output for subsets of length 1."""
        lambda_indices = [[i] for i in range(self.sim_general.values.shape[0])]

        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_general)

        N = len(lambda_indices)
        lambda_matrix_arr = np.zeros([N, N])
        for (j_idx, i_idx) in itertools.combinations(range(N), 2):

            svd2_val = self.snj_solver._compute_svd2(
                pair=(i_idx, j_idx), lambda_indices=lambda_indices
            )

            lambda_matrix_arr[i_idx, j_idx] = lambda_matrix_arr[
                j_idx, i_idx
            ] = svd2_val
        np.fill_diagonal(lambda_matrix_arr, np.inf)

        expected_lambda_matrix = np.array(
            [
                [
                    np.inf,
                    1.55149248e-01,
                    1.53000011e-01,
                    3.20069388e-01,
                    3.25748925e-01,
                ],
                [
                    1.55149248e-01,
                    np.inf,
                    4.14776245e-09,
                    4.60567415e-01,
                    4.59750019e-01,
                ],
                [
                    1.53000011e-01,
                    4.14776245e-09,
                    np.inf,
                    4.44601553e-01,
                    4.57653025e-01,
                ],
                [
                    3.20069388e-01,
                    4.60567415e-01,
                    4.44601553e-01,
                    np.inf,
                    4.45164957e-09,
                ],
                [
                    3.25748925e-01,
                    4.59750019e-01,
                    4.57653025e-01,
                    4.45164957e-09,
                    np.inf,
                ],
            ]
        )

        self.assertTrue(
            np.allclose(lambda_matrix_arr, expected_lambda_matrix, atol=0.1)
        )

    def test_compute_svd2_N3(self):
        """Test lamba matrix output for when there are 3 subsets left."""
        lambda_indices = [[0], [1, 2], [3, 4]]

        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_pp)

        N = len(lambda_indices)
        lambda_matrix_arr = np.zeros([N, N])
        for (j_idx, i_idx) in itertools.combinations(range(N), 2):

            svd2_val = self.snj_solver._compute_svd2(
                pair=(i_idx, j_idx), lambda_indices=lambda_indices
            )

            lambda_matrix_arr[i_idx, j_idx] = lambda_matrix_arr[
                j_idx, i_idx
            ] = svd2_val
        np.fill_diagonal(lambda_matrix_arr, np.inf)

        expected_lambda_matrix = np.array(
            [
                [np.inf, 4.51233062e-17, 4.51233062e-17],
                [4.51233062e-17, np.inf, 7.77014257e-01],
                [4.51233062e-17, 7.77014257e-01, np.inf],
            ]
        )

        self.assertTrue(
            np.allclose(lambda_matrix_arr, expected_lambda_matrix, atol=0.1)
        )

    def test_update_dissimilarity_map_base(self):
        """Test the update method's lambda matrix output."""
        # run to generate similarity map
        lambda_matrix = self.snj_solver.get_dissimilarity_map(self.tree_pp)

        node_i, node_j = (
            lambda_matrix.index[0],
            lambda_matrix.index[1],
        )

        svd2_vals = self.snj_solver.update_dissimilarity_map(
            lambda_matrix, (node_i, node_j), "new_node"
        )

        expected_lambda = np.array(
            [
                [
                    np.inf,
                    4.75160977e-01,
                    4.69514735e-01,
                    3.77716429e-01,
                    1.35120582e-16,
                ],
                [
                    4.75160977e-01,
                    np.inf,
                    1.58395087e-16,
                    2.15559051e-01,
                    4.68450526e-01,
                ],
                [
                    4.69514735e-01,
                    1.58395087e-16,
                    np.inf,
                    2.10372511e-01,
                    4.45839079e-01,
                ],
                [
                    3.77716429e-01,
                    2.15559051e-01,
                    2.10372511e-01,
                    np.inf,
                    3.81880809e-01,
                ],
                [
                    1.35120582e-16,
                    4.68450526e-01,
                    4.45839079e-01,
                    3.81880809e-01,
                    np.inf,
                ],
            ]
        )

        self.assertTrue(np.allclose(svd2_vals, expected_lambda, atol=0.1))
        self.assertListEqual(
            svd2_vals.index.values.tolist(), ["c", "d", "e", "root", "new_node"]
        )
        self.assertListEqual(
            svd2_vals.columns.values.tolist(),
            ["c", "d", "e", "root", "new_node"],
        )

    def notest_update_dissimilarity_map_N3(self):
        """Test the update function when there are only 3 subsets left. It
        should return all zeros since the output is not used.
        """
        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_pp)
        self.snj_solver.lambda_indices = [[0], [1, 2], [3, 4]]
        lambda_matrix_arr = None

        node_names = ["g1", "g2", "g3"]
        lambda_matrix = pd.DataFrame(
            lambda_matrix_arr, index=node_names, columns=node_names
        )

        node_i, node_j = (
            lambda_matrix.index[0],
            lambda_matrix.index[1],
        )

        svd2_vals = self.snj_solver.update_dissimilarity_map(
            lambda_matrix, (node_i, node_j), "new_node"
        )

        expected_lambda = np.zeros([2, 2])

        self.assertTrue(np.allclose(svd2_vals, expected_lambda, atol=0.1))

    def test_get_dissimilarity_map(self):
        """Test the override method that outputs a lambda matrix."""
        # instantiate new solver
        new_solver = cas.solver.SpectralNeighborJoiningSolver(add_root=False)

        # get lambda products
        lambda_matrix = new_solver.get_dissimilarity_map(
            self.tree_general, None
        )
        lambda_indices = new_solver.lambda_indices
        expected_lambda_indices = [[0], [1], [2], [3], [4]]

        expected_lambda_matrix = np.array(
            [
                [
                    np.inf,
                    1.55149248e-01,
                    1.53000011e-01,
                    3.20069388e-01,
                    3.25748925e-01,
                ],
                [
                    1.55149248e-01,
                    np.inf,
                    4.14776245e-09,
                    4.60567415e-01,
                    4.59750019e-01,
                ],
                [
                    1.53000011e-01,
                    4.14776245e-09,
                    np.inf,
                    4.44601553e-01,
                    4.57653025e-01,
                ],
                [
                    3.20069388e-01,
                    4.60567415e-01,
                    4.44601553e-01,
                    np.inf,
                    4.45164957e-09,
                ],
                [
                    3.25748925e-01,
                    4.59750019e-01,
                    4.57653025e-01,
                    4.45164957e-09,
                    np.inf,
                ],
            ]
        )

        # get similarity matrices
        obs_sim = new_solver._similarity_map
        expected_sim = self.sim_general

        # compare lambda matrix output
        self.assertTrue(
            np.allclose(lambda_matrix, expected_lambda_matrix, atol=0.1)
        )

        # compare similarity matrices
        self.assertTrue(np.allclose(obs_sim, expected_sim, atol=0.1))

        # compare indices
        self.assertTrue(
            np.allclose(lambda_indices, expected_lambda_indices, atol=0)
        )

    def test_find_cherry(self):
        """Test the find_cherry method."""
        fake_map1 = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ]
        )

        fake_map2 = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        )

        cherry1 = self.snj_solver.find_cherry(fake_map1)
        cherry2 = self.snj_solver.find_cherry(fake_map2)

        self.assertEqual(len(cherry1), 2)
        self.assertEqual(cherry1[0], 0)
        self.assertEqual(cherry1[1], 3)
        self.assertEqual(cherry2[0], 0)
        self.assertEqual(cherry2[1], 3)

    def test_basic_solver(self):
        """Test the features of the output of the solver on a root-specified
        input tree.
        """
        self.snj_solver.solve(self.tree_general)

        # test leaves exist in tree
        _leaves = self.tree_general.leaves
        self.assertEqual(len(_leaves), self.sim_general.shape[0] - 1)
        for _leaf in _leaves:
            self.assertIn(_leaf, self.sim_general.index.values)

        # test for expected number of edges
        edges = list(self.tree_general.edges)
        self.assertEqual(len(edges), 6)

        # test relationships between samples
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("b", "c"),
                ("b", "1"),
                ("1", "a"),
                ("1", "2"),
                ("2", "d"),
                ("2", "e"),
            ]
        )

        # define leaf nodes.
        leaves = ["a", "c", "d", "e"]

        # solve without collapsing mutationless edges
        observed_tree = self.tree_general.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

        # solve with collapsing mutationless edges
        self.snj_solver.solve(
            self.tree_general, collapse_mutationless_edges=True
        )
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(["a", "b", "c", "d", "e", "5", "6", "7"])
        expected_tree.add_edges_from(
            [("6", "a"), ("6", "e"), ("b", "6"), ("6", "d"), ("6", "c")]
        )
        observed_tree = self.tree_general.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

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

    def test_snj_solver_weights(self):
        """Test a perfect phylogenetic tree with priors."""
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("root", "7"),
                ("7", "6"),
                ("6", "d"),
                ("6", "e"),
                ("7", "8"),
                ("8", "a"),
                ("8", "9"),
                ("9", "b"),
                ("9", "c"),
            ]
        )

        leaves = ["a", "b", "c", "d", "e"]

        # solve without collapsing mutationless edges
        self.snj_solver.solve(self.tree_pp_priors)
        observed_tree = self.tree_pp_priors.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

        # solve with collapsing mutationless edges
        self.snj_solver.solve(
            self.tree_pp_priors, collapse_mutationless_edges=True
        )
        observed_tree = self.tree_pp_priors.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

    def test_pp_solver(self):
        """Integration Test for a Perfect Phylogenetic tree."""
        # define expected tree
        expected_tree = nx.DiGraph()
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

        # get all triplets
        leaves = ["a", "b", "c", "d", "e"]

        # no collapsing mutationless edges
        self.snj_solver.solve(self.tree_pp)
        observed_tree = self.tree_pp.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

        # collapse mutationless edges
        self.snj_solver.solve(self.tree_pp, collapse_mutationless_edges=True)
        observed_tree = self.tree_pp.get_tree_topology()
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

    def test_duplicate_sample(self):
        """Test the solving of a tree with duplicate leaves."""

        self.snj_solver.solve(self.tree_dupe)
        observed_tree = self.tree_dupe.get_tree_topology()

        expected_tree = nx.DiGraph()
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

        leaves = ["a", "b", "c", "d", "e", "f"]
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

    def test_integration1(self):
        """Integration test on a 7-leaf tree."""

        # construct expected tree
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "i1",
                "i2",
                "i3",
                "i4",
                "i5",
                "i6",
            ]
        )
        expected_tree.add_edges_from(
            [
                ("i4", "i2"),
                ("i2", "d"),
                ("i2", "i1"),
                ("i1", "a"),
                ("i1", "i3"),
                ("i3", "b"),
                ("i3", "c"),
                ("i4", "i5"),
                ("i5", "e"),
                ("i5", "i6"),
                ("i6", "f"),
                ("i6", "g"),
            ]
        )

        # solve the tree
        self.snj_solver.solve(self.tree_int1)
        observed_tree = self.tree_int1.get_tree_topology()

        # check expected vs observed trees
        leaves = ["a", "b", "c", "d", "e", "f", "g"]
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

    def test_integration2(self):
        """Integration test on a 10-leaf tree."""
        # construct expected tree
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "i1",
                "i2",
                "i3",
                "i4",
                "i5",
                "i6",
                "i7",
                "i8",
                "i9",
            ]
        )
        expected_tree.add_edges_from(
            [
                ("i1", "i2"),
                ("i2", "e"),
                ("i2", "i3"),
                ("i3", "d"),
                ("i3", "i4"),
                ("i4", "c"),
                ("i4", "i5"),
                ("i5", "a"),
                ("i5", "b"),
                ("i1", "i6"),
                ("i6", "i7"),
                ("i7", "j"),
                ("i7", "i"),
                ("i6", "i8"),
                ("i8", "h"),
                ("i8", "i9"),
                ("i9", "f"),
                ("i9", "g"),
            ]
        )

        # solve the tree
        self.snj_solver.solve(self.tree_int2)
        observed_tree = self.tree_int2.get_tree_topology()

        # check expected vs observed tree
        leaves = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        assertTripletCorrectness(self, leaves, expected_tree, observed_tree)

    def test_setup_root_finder_missing_dissimilarity_map(self):
        """Check that root still set despite missing dissimilarity map."""
        tree = cas.data.CassiopeiaTree(character_matrix=self.cm_general)

        self.snj_solver.setup_root_finder(tree)

        self.assertEqual(tree.root_sample_name, "root")

    def test_setup_root_finder_existing_dissimilarity_map(self):
        """Check that root is still set with existing dissimilarity map."""
        tree = cas.data.CassiopeiaTree(
            character_matrix=self.cm_general,
            dissimilarity_map=self.sim_general,
        )
        with mock.patch.object(
            tree, "compute_dissimilarity_map"
        ) as compute_dissimilarity_map:
            self.snj_solver.setup_root_finder(tree)
            compute_dissimilarity_map.assert_not_called()
        self.assertEqual(tree.root_sample_name, "root")
        dissimilarity_map = tree.get_dissimilarity_map()
        self.assertEqual(
            {"a", "b", "c", "d", "e", "root"}, set(dissimilarity_map.index)
        )
        self.assertEqual(
            {"a", "b", "c", "d", "e", "root"}, set(dissimilarity_map.columns)
        )
        self.assertEqual(dissimilarity_map.loc["root", "root"], 0)


if __name__ == "__main__":
    unittest.main()
