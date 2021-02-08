import unittest

import itertools
import networkx as nx
import pandas as pd
import random

from cassiopeia.solver.MaxCutSolver import MaxCutSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.data import utilities as tree_utilities


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


class MaxCutSolverTest(unittest.TestCase):
    def setUp(self):
        self.cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1],
                "c2": [5, 4, 0],
                "c3": [4, 4, 3],
                "c4": [4, 4, 3],
                "c5": [-1, 4, 0],
            },
            orient="index",
            columns=["a", "b", "c"],
        )

        self.cm2 = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1],
                "c2": [5, 4, 1],
                "c3": [4, 4, 3],
                "c4": [4, 4, 3],
                "c5": [-1, 4, 3],
            },
            orient="index",
            columns=["a", "b", "c"],
        )
        self.mcsolver = MaxCutSolver(character_matrix=self.cm, missing_char=-1)
        self.mutation_frequencies = self.mcsolver.compute_mutation_frequencies(
            self.mcsolver.unique_character_matrix.index
        )

    def test_check_if_cut(self):
        self.assertTrue(graph_utilities.check_if_cut(2, 4, [0, 1, 2]))
        self.assertFalse(
            graph_utilities.check_if_cut("c1", "c2", ["c1", "c2", "c3"])
        )

    def test_evaluate_cut(self):
        G = graph_utilities.construct_connectivity_graph(
            self.mcsolver.unique_character_matrix,
            self.mutation_frequencies,
            -1,
            self.mcsolver.unique_character_matrix.index,
        )
        cut_weight = self.mcsolver.evaluate_cut(["c2", "c3"], G)
        self.assertEqual(cut_weight, -4)

    def test_graph_construction(self):
        G = graph_utilities.construct_connectivity_graph(
            self.mcsolver.unique_character_matrix,
            self.mutation_frequencies,
            -1,
            self.mcsolver.unique_character_matrix.index,
        )

        self.assertEqual(G["c1"]["c2"]["weight"], -1)
        self.assertEqual(G["c1"]["c3"]["weight"], 3)
        self.assertEqual(G["c1"]["c5"]["weight"], 2)
        self.assertEqual(G["c2"]["c3"]["weight"], -2)
        self.assertEqual(G["c2"]["c5"]["weight"], -3)
        self.assertEqual(G["c3"]["c5"]["weight"], -3)

    def test_graph_construction_weights(self):
        weights = {0: {4: 1, 5: 2}, 1: {4: 2}, 2: {1: 1, 3: 1}}

        G = graph_utilities.construct_connectivity_graph(
            self.mcsolver.unique_character_matrix,
            self.mutation_frequencies,
            -1,
            self.mcsolver.unique_character_matrix.index,
            weights=weights,
        )

        self.assertEqual(G["c1"]["c2"]["weight"], -2)
        self.assertEqual(G["c1"]["c3"]["weight"], 6)
        self.assertEqual(G["c1"]["c5"]["weight"], 4)
        self.assertEqual(G["c2"]["c3"]["weight"], -4)
        self.assertEqual(G["c2"]["c5"]["weight"], -6)
        self.assertEqual(G["c3"]["c5"]["weight"], -6)

    def test_hill_climb(self):
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i)
        G.add_edge(0, 1, weight=3)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=2)
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=10)
        G.add_edge(2, 3, weight=1)

        new_cut = graph_utilities.max_cut_improve_cut(G, [0, 2])
        self.assertEqual(new_cut, [0, 1])

    def test_polytomy_base_case(self):
        # A case where samples c2, c3, c5 cannot be resolved, so they are returned
        # as a polytomy
        self.mcsolver.solve()
        expected_newick_string = "(c1,(c2,c5,(c3,c4)));"
        observed_newick_string = tree_utilities.to_newick(self.mcsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_simple_base_case(self):
        mcsolver2 = MaxCutSolver(character_matrix=self.cm2, missing_char=-1)
        mcsolver2.solve()
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from([5, 6, 7, 8, "c1", "c2", "c3", "c4", "c5"])
        expected_tree.add_edges_from(
            [
                (5, 6),
                (5, 7),
                (6, "c1"),
                (6, "c2"),
                (7, 8),
                (8, "c3"),
                (8, "c4"),
                (7, "c5"),
            ]
        )
        triplets = itertools.combinations(["c1", "c2", "c3", "c4", "c5"], 3)
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, mcsolver2.tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_simple_base_case_priors(self):
        # Priors can increase connectivity on another character
        priors = {
            0: {5: 0.05, 4: 0.05},
            1: {4: 0.01, 5: 0.99},
            2: {1: 0.5, 3: 0.5},
        }
        # random.seed(10)
        mcsolver2p = MaxCutSolver(
            character_matrix=self.cm2, missing_char=-1, priors=priors
        )
        mcsolver2p.solve()
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from([5, 6, 7, 8, "c1", "c2", "c3", "c4", "c5"])
        expected_tree.add_edges_from(
            [
                (5, "c1"),
                (5, 6),
                (6, "c2"),
                (6, 7),
                (7, "c5"),
                (7, 8),
                (8, "c3"),
                (8, "c4"),
            ]
        )
        triplets = itertools.combinations(["c1", "c2", "c3", "c4", "c5"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, mcsolver2p.tree)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
