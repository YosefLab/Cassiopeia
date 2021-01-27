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

        dir_path = os.path.dirname(os.path.realpath(__file__))
        ilp_solver = cas.solver.ILPSolver(
            cm,
            missing_char=-1,
            logfile=os.path.join(dir_path, "test.log"),
            mip_gap=0.0,
        )

        greedy_solver = cas.solver.VanillaGreedySolver(cm, missing_char=-1)

        self.hybrid_pp_solver = cas.solver.HybridSolver(
            cm, greedy_solver, ilp_solver, missing_char=-1, cell_cutoff=3, threads=2
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
            self.hybrid_pp_solver.unique_character_matrix,
        )

    def test_cutoff(self):

        self.assertTrue(
            self.hybrid_pp_solver.assess_cutoff(["a", "b", "c"]), True
        )
        self.assertFalse(
            self.hybrid_pp_solver.assess_cutoff(["a", "b", "c", "d"]), False
        )

        # test lca-based cutoff
        self.hybrid_pp_solver.cell_cutoff = None
        self.hybrid_pp_solver.lca_cutoff = 2

        self.assertTrue(self.hybrid_pp_solver.assess_cutoff(["a", "b", "c"]))
        self.assertFalse(self.hybrid_pp_solver.assess_cutoff(["c", "d"]))

    def test_top_down_split(self):

        mutation_frequencies = self.hybrid_pp_solver.top_solver.compute_mutation_frequencies(["a", "b", "c", "d", "e"])

        expected_dictionary = {0: {1: 3, 2: 2, -1: 0}, 1: {1: 1, 2: 2, 0: 2, -1: 0}, 2: {0: 3, 1: 1, 2: 1, -1: 0}}
        self.assertDictEqual(mutation_frequencies, expected_dictionary)

        clades = self.hybrid_pp_solver.top_solver.perform_split(mutation_frequencies, ["a", "b", "c", "d", "e"])

        expected_split = (["a", "b", "c"], ["d", "e"])
        for expected_clade in expected_split:
            self.assertIn(expected_clade, clades)

    def test_full_hybrid(self):

        self.hybrid_pp_solver.solve()

        tree = self.hybrid_pp_solver.tree

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


    # def tearDown(self):

    #     for _file in os.listdir("."):
    #         if '.log' in _file:
                


if __name__ == "__main__":
    unittest.main()
