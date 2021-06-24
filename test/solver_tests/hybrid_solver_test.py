"""
Test HybridSolver in Cassiopeia.solver.
"""
import os
import unittest

import itertools
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import solver_utilities


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


class TestHybridSolver(unittest.TestCase):
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

        cm_large = pd.DataFrame.from_dict(
            {
                "a": [1, 0, 0, 0, 0, 0, 0, 0],
                "b": [1, 1, 0, 0, 0, 0, 0, 0],
                "c": [1, 1, 1, 0, 0, 0, 0, 0],
                "d": [1, 1, 1, 1, 0, 0, 0, 0],
                "e": [1, 1, 1, 1, 1, 0, 0, 0],
                "f": [1, 1, 1, 1, 1, 1, 0, 0],
                "g": [1, 1, 1, 1, 1, 1, 1, 0],
                "h": [1, 1, 1, 1, 1, 1, 1, 1],
                "i": [2, 0, 0, 0, 0, 0, 0, 0],
                "j": [2, 2, 0, 0, 0, 0, 0, 0],
            },
            orient="index",
        )

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

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.logfile = os.path.join(self.dir_path, "test.log")

        self.pp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        self.large_tree = cas.data.CassiopeiaTree(
            cm_large, missing_state_indicator=-1
        )
        self.missing_tree = cas.data.CassiopeiaTree(
            cm_missing, missing_state_indicator=-1
        )

        ## smaller hybrid solver
        ilp_solver = cas.solver.ILPSolver(mip_gap=0.0)

        greedy_solver = cas.solver.VanillaGreedySolver()
        self.hybrid_pp_solver = cas.solver.HybridSolver(
            greedy_solver, ilp_solver, cell_cutoff=3, threads=2
        )

        ## larger hybrid solver
        self.hybrid_pp_solver_large = cas.solver.HybridSolver(
            greedy_solver, ilp_solver, cell_cutoff=3, threads=2
        )

        ## hybrid solver with missing data
        self.hybrid_pp_solver_missing = cas.solver.HybridSolver(
            greedy_solver, ilp_solver, cell_cutoff=3, threads=2
        )

        ## hybrid solver with MaxCut Greedy
        greedy_maxcut_solver = cas.solver.MaxCutGreedySolver()
        self.hybrid_pp_solver_maxcut = cas.solver.HybridSolver(
            greedy_maxcut_solver, ilp_solver, cell_cutoff=3, threads=2
        )

    def test_constructor(self):

        self.assertEqual(self.hybrid_pp_solver.cell_cutoff, 3)
        self.assertEqual(self.hybrid_pp_solver.lca_cutoff, None)

        # test bottom solver is populated correctly
        self.assertEqual(
            self.hybrid_pp_solver.bottom_solver.convergence_time_limit, 12600
        )
        self.assertEqual(
            self.hybrid_pp_solver.bottom_solver.maximum_potential_graph_layer_size,
            10000,
        )
        self.assertFalse(self.hybrid_pp_solver.bottom_solver.weighted)

        expected_unique_character_matrix = pd.DataFrame.from_dict(
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
            expected_unique_character_matrix,
            self.pp_tree.character_matrix.copy(),
        )

    def test_cutoff(self):

        character_matrix = self.pp_tree.character_matrix.copy()
        missing_state = self.pp_tree.missing_state_indicator
        self.assertTrue(
            self.hybrid_pp_solver.assess_cutoff(
                ["a", "b", "c"], character_matrix, missing_state
            ),
            True,
        )
        self.assertFalse(
            self.hybrid_pp_solver.assess_cutoff(
                ["a", "b", "c", "d"], character_matrix, missing_state
            ),
            False,
        )

        # test lca-based cutoff
        self.hybrid_pp_solver.cell_cutoff = None
        self.hybrid_pp_solver.lca_cutoff = 2

        self.assertTrue(
            self.hybrid_pp_solver.assess_cutoff(
                ["a", "b", "c"], character_matrix, missing_state
            )
        )
        self.assertFalse(
            self.hybrid_pp_solver.assess_cutoff(
                ["c", "d"], character_matrix, missing_state
            )
        )

    def test_top_down_split_manual(self):

        character_matrix = self.pp_tree.character_matrix.copy()
        # test manually
        mutation_frequencies = self.hybrid_pp_solver.top_solver.compute_mutation_frequencies(
            ["a", "b", "c", "d", "e"],
            character_matrix,
            self.pp_tree.missing_state_indicator,
        )

        expected_dictionary = {
            0: {1: 3, 2: 2, -1: 0},
            1: {1: 1, 2: 2, 0: 2, -1: 0},
            2: {0: 3, 1: 1, 2: 1, -1: 0},
        }
        self.assertDictEqual(mutation_frequencies, expected_dictionary)

        clades = self.hybrid_pp_solver.top_solver.perform_split(
            character_matrix, ["a", "b", "c", "d", "e"]
        )

        expected_split = (["a", "b", "c"], ["d", "e"])
        for expected_clade in expected_split:
            self.assertIn(expected_clade, clades)

    def test_apply_top_solver_small(self):

        character_matrix = self.pp_tree.character_matrix.copy()
        unique_character_matrix = character_matrix.drop_duplicates()
        names = solver_utilities.node_name_generator()

        _, subproblems = self.hybrid_pp_solver.apply_top_solver(
            unique_character_matrix, list(unique_character_matrix.index), names
        )

        expected_clades = (["a", "b", "c"], ["d", "e"])
        observed_clades = [subproblem[1] for subproblem in subproblems]
        self.assertEqual(len(expected_clades), len(observed_clades))

        for clade in expected_clades:
            self.assertIn(clade, observed_clades)

    def test_apply_top_solver_large(self):

        character_matrix = self.large_tree.character_matrix.copy()
        unique_character_matrix = character_matrix.drop_duplicates()
        names = solver_utilities.node_name_generator()

        _, subproblems = self.hybrid_pp_solver_large.apply_top_solver(
            unique_character_matrix, list(unique_character_matrix.index), names
        )

        expected_clades = (
            ["a"],
            ["b"],
            ["c"],
            ["d"],
            ["e"],
            ["f", "g", "h"],
            ["i", "j"],
        )
        observed_clades = [subproblem[1] for subproblem in subproblems]
        self.assertEqual(len(expected_clades), len(observed_clades))

        for clade in expected_clades:
            self.assertIn(clade, observed_clades)

    def test_apply_top_solver_missing(self):

        character_matrix = self.missing_tree.character_matrix.copy()
        unique_character_matrix = character_matrix.drop_duplicates()
        names = solver_utilities.node_name_generator()

        _, subproblems = self.hybrid_pp_solver_missing.apply_top_solver(
            unique_character_matrix, list(unique_character_matrix.index), names
        )

        expected_clades = (["a", "b", "c"], ["d", "e"], ["f", "g", "h"])
        observed_clades = [subproblem[1] for subproblem in subproblems]
        self.assertEqual(len(expected_clades), len(observed_clades))

        for clade in expected_clades:
            self.assertIn(clade, observed_clades)

    def test_full_hybrid(self):

        self.hybrid_pp_solver.solve(self.pp_tree, logfile=self.logfile)

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

    def test_full_hybrid_single_thread(self):

        self.hybrid_pp_solver.threads = 1
        self.hybrid_pp_solver.solve(self.pp_tree, logfile=self.logfile)

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

    def test_full_hybrid_large(self):

        self.hybrid_pp_solver_large.solve(self.large_tree, logfile=self.logfile)

        tree = self.large_tree.get_tree_topology()

        # make sure there's one root
        roots = [n for n in tree if tree.in_degree(n) == 0]
        self.assertEqual(len(roots), 1)

        # make sure all samples are leaves
        tree_leaves = [n for n in tree if tree.out_degree(n) == 0]
        expected_leaves = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        for leaf in expected_leaves:
            self.assertIn(leaf, tree_leaves)

        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                ("node0", "node1"),
                ("node0", "node2"),
                ("node1", "a"),
                ("node1", "node4"),
                ("node2", "i"),
                ("node2", "j"),
                ("node4", "b"),
                ("node4", "node8"),
                ("node8", "c"),
                ("node8", "node10"),
                ("node10", "d"),
                ("node10", "node12"),
                ("node12", "e"),
                ("node12", "node14"),
                ("node14", "f"),
                ("node14", "node16"),
                ("node16", "g"),
                ("node16", "h"),
            ]
        )

        triplets = itertools.combinations(
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_full_hybrid_maxcut(self):

        self.hybrid_pp_solver_maxcut.solve(
            self.missing_tree, logfile=self.logfile
        )

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

    def test_full_hybrid_missing(self):

        self.hybrid_pp_solver_missing.solve(
            self.missing_tree, logfile=self.logfile
        )

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

    def tearDown(self):

        for _file in os.listdir(self.dir_path):
            if ".log" in _file:
                os.remove(os.path.join(self.dir_path, _file))


if __name__ == "__main__":
    unittest.main()
