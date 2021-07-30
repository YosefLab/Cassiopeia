"""
Test ILPSolver in Cassiopeia.solver.
"""
import os
import unittest

import itertools
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import ILPSolverError
from cassiopeia.solver import ilp_solver_utilities


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


class TestILPSolver(unittest.TestCase):
    def setUp(self):

        # basic PP example with no missing data
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

        # basic PP example with duplicates
        cm_duplicates = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
                "f": [1, 1, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        # basic example with missing data
        cm_missing = pd.DataFrame.from_dict(
            {
                "a": [1, 3, 1, 1],
                "b": [1, 3, 1, -1],
                "c": [1, 0, 1, 0],
                "d": [1, 1, 3, 0],
                "e": [1, 1, 0, 0],
                "f": [2, 0, 0, 0],
                "g": [2, 4, -1, -1],
                "h": [2, 4, 2, 0],
            },
            orient="index",
        )

        dir_path = os.path.dirname(os.path.realpath(__file__))

        open(os.path.join(dir_path, "test.log"), "a").close()
        self.pp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        self.duplicates_tree = cas.data.CassiopeiaTree(
            cm_duplicates, missing_state_indicator=-1
        )
        self.missing_tree = cas.data.CassiopeiaTree(
            cm_missing, missing_state_indicator=-1
        )
        self.logfile = os.path.join(dir_path, "test.log")

        self.ilp_solver = cas.solver.ILPSolver(mip_gap=0.0)

    def test_raises_error_on_ambiguous(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, (0, 1), 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, 2, 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        with self.assertRaises(ILPSolverError):
            solver = cas.solver.ILPSolver()
            solver.solve(tree)

    def test_get_lca_cython(self):

        # test single sample
        cm = self.missing_tree.character_matrix.copy().astype(str)

        lca = ilp_solver_utilities.get_lca_characters_cython(
            cm.loc["a"].values, cm.loc["b"].values, 4, "-1"
        )

        self.assertEqual(lca, "1|3|1|1")

        lca = ilp_solver_utilities.get_lca_characters_cython(
            cm.loc["h"].values, cm.loc["b"].values, 4, "-1"
        )
        self.assertEqual(lca, "0|0|0|0")

    def test_cython_hamming_dist(self):

        sample1 = np.array(["1", "2", "3", "0", "0"])
        sample2 = np.array(["1", "4", "0", "0", "1"])
        dist = ilp_solver_utilities.simple_hamming_distance_cython(
            sample1, sample2, "-"
        )
        self.assertEqual(dist, 3)

        sample1 = np.array(["1", "2", "3", "0", "-"])
        sample2 = np.array(["1", "-", "0", "0", "1"])
        dist = ilp_solver_utilities.simple_hamming_distance_cython(
            sample1, sample2, "-"
        )
        self.assertEqual(dist, 1)

    def test_single_sample_ilp(self):

        # test single sample
        cm = pd.DataFrame([1], index=["a"])
        tree = cas.data.CassiopeiaTree(cm)

        self.ilp_solver.solve(tree, logfile=self.logfile)
        expected_leaves = ["a"]
        self.assertCountEqual(expected_leaves, tree.leaves)

        # test single unique sample
        cm = pd.DataFrame([[1], [1], [1]], index=["a", "b", "c"])
        tree = cas.data.CassiopeiaTree(cm)

        self.ilp_solver.solve(tree, logfile=self.logfile)
        expected_leaves = ["a", "b", "c"]
        self.assertCountEqual(expected_leaves, tree.leaves)

    def test_basic_ilp_constructor(self):

        self.assertEqual(self.ilp_solver.convergence_time_limit, 12600)
        self.assertEqual(
            self.ilp_solver.maximum_potential_graph_layer_size, 10000
        )
        self.assertFalse(self.ilp_solver.weighted)

        expected_character_matrix = pd.DataFrame.from_dict(
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

        pd.testing.assert_frame_equal(
            expected_character_matrix, self.pp_tree.character_matrix.copy()
        )

    def test_get_layer_for_potential_graph(self):

        unique_character_matrix = (
            self.pp_tree.character_matrix.drop_duplicates()
        )
        source_nodes = unique_character_matrix.values
        dim = source_nodes.shape[1]

        source_node_strings = np.array(
            ["|".join(arr) for arr in source_nodes.astype(str)]
        )
        (
            layer_nodes,
            layer_edges,
        ) = ilp_solver_utilities.infer_layer_of_potential_graph(
            source_node_strings, 10
        )

        layer_nodes = np.array(
            [node.split("|") for node in layer_nodes], dtype=int
        )
        layer_nodes = np.unique(layer_nodes, axis=0)

        expected_next_layer = np.array(
            [[1, 0, 0], [1, 2, 0], [0, 0, 0], [2, 0, 0]]
        )

        for sample in expected_next_layer:
            self.assertIn(sample, layer_nodes)

        layer_edges = np.array(
            [edge.split("|") for edge in layer_edges], dtype=int
        )
        layer_edges = [(list(e[:dim]), list(e[dim:])) for e in layer_edges]
        expected_edges = [
            ([1, 0, 0], [1, 1, 0]),
            ([1, 0, 0], [1, 2, 0]),
            ([1, 0, 0], [1, 2, 1]),
            ([1, 2, 0], [1, 2, 0]),
            ([1, 2, 0], [1, 2, 1]),
            ([0, 0, 0], [1, 1, 0]),
            ([0, 0, 0], [1, 2, 0]),
            ([0, 0, 0], [1, 2, 1]),
            ([0, 0, 0], [2, 0, 0]),
            ([0, 0, 0], [2, 0, 2]),
            ([2, 0, 0], [2, 0, 0]),
            ([2, 0, 0], [2, 0, 2]),
        ]

        for edge in expected_edges:
            self.assertIn(edge, layer_edges)

        uniq_edges = []
        for edge in layer_edges:
            if edge not in uniq_edges:
                uniq_edges.append(edge)

        self.assertEqual(len(uniq_edges), len(expected_edges))

    def test_simple_potential_graph_inference(self):

        unique_character_matrix = (
            self.pp_tree.character_matrix.drop_duplicates()
        )

        max_lca_height = 10
        potential_graph = self.ilp_solver.infer_potential_graph(
            unique_character_matrix,
            0,
            max_lca_height,
            self.pp_tree.priors,
            self.pp_tree.missing_state_indicator,
        )

        # expected nodes
        expected_nodes = [
            (1, 1, 0),
            (1, 2, 0),
            (1, 2, 1),
            (2, 0, 0),
            (2, 0, 2),
            (1, 0, 0),
            (1, 2, 0),
            (0, 0, 0),
            (2, 0, 0),
        ]

        for node in expected_nodes:
            self.assertIn(node, potential_graph.nodes())

        # expected edges
        expected_edges = [
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 2, 0)),
            ((1, 0, 0), (1, 2, 1)),
            ((1, 2, 0), (1, 2, 1)),
            ((0, 0, 0), (1, 1, 0)),
            ((0, 0, 0), (1, 2, 0)),
            ((0, 0, 0), (1, 2, 1)),
            ((0, 0, 0), (2, 0, 0)),
            ((0, 0, 0), (2, 0, 2)),
            ((2, 0, 0), (2, 0, 2)),
            ((0, 0, 0), (1, 0, 0)),
        ]

        for edge in expected_edges:
            self.assertIn(edge, potential_graph.edges())

        self.assertEqual(len(potential_graph.edges()), len(expected_edges))

    def test_ilp_solver_perfect_phylogeny(self):

        self.ilp_solver.solve(self.pp_tree, logfile=self.logfile)
        tree = self.pp_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = self.pp_tree.leaves
        expected_leaves = ["a", "b", "c", "d", "e"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        # make sure every node has at most one parent
        multi_parents = [n for n in tree if tree.in_degree(n) > 1]
        self.assertEqual(len(multi_parents), 0)

        # make sure the resulting tree has no unifurcations
        one_child = [
            n for n in self.pp_tree.nodes if len(self.pp_tree.children(n)) == 1
        ]
        self.assertEqual(len(one_child), 0)

        # expected parsimony
        expected_parsimony = 6

        # apply camin-sokal parsimony
        self.pp_tree.reconstruct_ancestral_characters()
        observed_parsimony = 0

        for e in self.pp_tree.depth_first_traverse_edges():
            c1, c2 = (
                self.pp_tree.get_character_states(e[0]),
                self.pp_tree.get_character_states(e[1]),
            )
            observed_parsimony += cas.solver.dissimilarity.hamming_distance(
                np.array(c1), np.array(c2)
            )

        self.assertEqual(observed_parsimony, expected_parsimony)

        # expected tree structure
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
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_potential_graph_inference_with_duplicates(self):

        unique_character_matrix = (
            self.duplicates_tree.character_matrix.drop_duplicates()
        )

        max_lca_height = 10
        potential_graph = self.ilp_solver.infer_potential_graph(
            unique_character_matrix,
            0,
            max_lca_height,
            self.duplicates_tree.priors,
            self.duplicates_tree.missing_state_indicator,
        )

        # expected nodes
        expected_nodes = [
            (1, 1, 0),
            (1, 2, 0),
            (1, 2, 1),
            (2, 0, 0),
            (2, 0, 2),
            (1, 0, 0),
            (1, 2, 0),
            (0, 0, 0),
            (2, 0, 0),
        ]

        for node in expected_nodes:
            self.assertIn(node, potential_graph.nodes())

        # expected edges
        expected_edges = [
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 2, 0)),
            ((1, 0, 0), (1, 2, 1)),
            ((1, 2, 0), (1, 2, 1)),
            ((0, 0, 0), (1, 1, 0)),
            ((0, 0, 0), (1, 2, 0)),
            ((0, 0, 0), (1, 2, 1)),
            ((0, 0, 0), (2, 0, 0)),
            ((0, 0, 0), (2, 0, 2)),
            ((2, 0, 0), (2, 0, 2)),
            ((0, 0, 0), (1, 0, 0)),
        ]

        for edge in expected_edges:
            self.assertIn(edge, potential_graph.edges())

        self.assertEqual(len(potential_graph.edges()), len(expected_edges))

    def test_ilp_solver_with_duplicates(self):

        self.ilp_solver.solve(self.duplicates_tree, logfile=self.logfile)
        tree = self.duplicates_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = self.duplicates_tree.leaves
        expected_leaves = ["a", "b", "c", "d", "e", "f"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        # make sure every node has at most one parent
        multi_parents = [n for n in tree if tree.in_degree(n) > 1]
        self.assertEqual(len(multi_parents), 0)

        # make sure the resulting tree has no unifurcations
        one_child = [
            n
            for n in self.duplicates_tree.nodes
            if len(self.duplicates_tree.children(n)) == 1
        ]
        self.assertEqual(len(one_child), 0)

        # expected parsimony
        expected_parsimony = 6
        self.duplicates_tree.reconstruct_ancestral_characters()
        observed_parsimony = 0

        for e in self.duplicates_tree.depth_first_traverse_edges():
            c1, c2 = (
                self.duplicates_tree.get_character_states(e[0]),
                self.duplicates_tree.get_character_states(e[1]),
            )
            observed_parsimony += cas.solver.dissimilarity.hamming_distance(
                np.array(c1), np.array(c2)
            )

        self.assertEqual(observed_parsimony, expected_parsimony)

        # expected tree structure
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "f", "root", "6", "7", "8", "9"]
        )
        expected_tree.add_edges_from(
            [
                ("root", "9"),
                ("9", "8"),
                ("9", "7"),
                ("7", "6"),
                ("7", "5"),
                ("7", "5"),
                ("5", "a"),
                ("5", "f"),
                ("6", "b"),
                ("6", "c"),
                ("8", "e"),
                ("8", "d"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

        self.ilp_solver.solve(
            self.duplicates_tree,
            logfile=self.logfile,
            collapse_mutationless_edges=True,
        )
        tree = self.duplicates_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_ilp_solver_missing_data(self):

        self.ilp_solver.solve(self.missing_tree, logfile=self.logfile)
        tree = self.missing_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = [n for n in tree if tree.out_degree(n) == 0]
        expected_leaves = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("node0", "node1"),
                ("node0", "node2"),
                ("node1", "node3"),
                ("node1", "node4"),
                ("node3", "c"),
                ("node3", "node6"),
                ("node6", "a"),
                ("node6", "b"),
                ("node4", "d"),
                ("node4", "e"),
                ("node2", "f"),
                ("node2", "node5"),
                ("node5", "g"),
                ("node5", "h"),
            ]
        )

        triplets = itertools.combinations(
            ["a", "b", "c", "d", "e", "f", "g", "h"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

        self.ilp_solver.solve(
            self.missing_tree,
            logfile=self.logfile,
            collapse_mutationless_edges=True,
        )
        tree = self.missing_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def tearDown(self):

        os.remove(self.logfile)


if __name__ == "__main__":
    unittest.main()
