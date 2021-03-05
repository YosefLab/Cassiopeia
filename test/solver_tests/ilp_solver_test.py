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

        # basic PP example with no missing data
        cm_duplicates = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
                "f": [1, 1, 0]
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        dir_path = os.path.dirname(os.path.realpath(__file__))

        open(os.path.join(dir_path, "test.log"), "a").close()
        self.pp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        self.duplicates_tree = cas.data.CassiopeiaTree(
            cm_duplicates, missing_state_indicator=-1
        )
        self.logfile = os.path.join(dir_path, "test.log")

        self.ilp_solver = cas.solver.ILPSolver(mip_gap=0.0)

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
            expected_character_matrix,
            self.pp_tree.get_original_character_matrix(),
        )

    def test_get_layer_for_potential_graph(self):

        unique_character_matrix = (
            self.pp_tree.get_original_character_matrix().drop_duplicates()
        )
        source_nodes = unique_character_matrix.values
        dim = source_nodes.shape[1]

        (
            layer_nodes,
            layer_edges,
        ) = ilp_solver_utilities.infer_layer_of_potential_graph(
            source_nodes, 10, self.pp_tree.missing_state_indicator
        )

        layer_nodes = np.unique(layer_nodes, axis=0)

        expected_next_layer = np.array(
            [[1, 0, 0], [1, 2, 0], [0, 0, 0], [2, 0, 0]]
        )

        for sample in expected_next_layer:
            self.assertIn(sample, layer_nodes)

        # layer_edges = [(list(e[0]), list(e[1])) for e in layer_edges]
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
            self.pp_tree.get_original_character_matrix().drop_duplicates()
        )
        root = data_utilities.get_lca_characters(
            unique_character_matrix.values.tolist(),
            self.pp_tree.missing_state_indicator,
        )
        max_lca_height = 10
        potential_graph = self.ilp_solver.infer_potential_graph(
            unique_character_matrix,
            root,
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

        self.ilp_solver.solve(self.pp_tree, self.logfile)
        tree = self.pp_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = [n for n in tree if tree.out_degree(n) == 0]
        expected_leaves = ["a", "b", "c", "d", "e"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        # make sure every node has at most one parent
        multi_parents = [n for n in tree if tree.in_degree(n) > 1]
        self.assertEqual(len(multi_parents), 0)

        # expected parsimony
        expected_parsimony = 6
        root = roots[0]

        observed_parsimony = 0
        for e in nx.dfs_edges(tree, source=root):
            if tree.out_degree(e[1]) > 0:
                observed_parsimony += cas.solver.dissimilarity.hamming_distance(
                    e[0], e[1]
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
            self.duplicates_tree.get_original_character_matrix().drop_duplicates()
        )
        root = data_utilities.get_lca_characters(
            unique_character_matrix.values.tolist(),
            self.duplicates_tree.missing_state_indicator,
        )
        max_lca_height = 10
        potential_graph = self.ilp_solver.infer_potential_graph(
            unique_character_matrix,
            root,
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

        self.ilp_solver.solve(self.duplicates_tree, self.logfile)
        tree = self.duplicates_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = [n for n in tree if tree.out_degree(n) == 0]
        expected_leaves = ["a", "b", "c", "d", "e", "f"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        # make sure every node has at most one parent
        multi_parents = [n for n in tree if tree.in_degree(n) > 1]
        self.assertEqual(len(multi_parents), 0)

        # expected parsimony
        expected_parsimony = 6
        root = roots[0]

        observed_parsimony = 0
        for e in nx.dfs_edges(tree, source=root):
            if tree.out_degree(e[1]) > 0:
                observed_parsimony += cas.solver.dissimilarity.hamming_distance(
                    e[0], e[1]
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
                ("7", "a"),
                ("7", "f"),
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

    def tearDown(self):

        os.remove(self.logfile)


if __name__ == "__main__":
    unittest.main()
