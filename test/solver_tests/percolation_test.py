"""
Test PercolationSolver in Cassiopeia.solver.
"""

import unittest

import itertools
import networkx as nx
import numba
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver.NeighborJoiningSolver import NeighborJoiningSolver
from cassiopeia.solver.PercolationSolver import PercolationSolver
from cassiopeia.solver import dissimilarity_functions


def find_triplet_structure(triplet, T):
    a, b, c = str(triplet[0]), str(triplet[1]), str(triplet[2])
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


NB_SIMILARITY = numba.jit(
    dissimilarity_functions.hamming_similarity_without_missing, nopython=True
)


def neg_hamming_similarity_without_missing(
    s1: np.array,
    s2: np.array,
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:

    return -1 * NB_SIMILARITY(s1, s2, missing_state_indicator, weights)


class PercolationSolverTest(unittest.TestCase):
    def test_NJ_negative_similarity(self):

        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 0, 0],
                "c2": [0, 3, 4, 2, 1],
                "c3": [5, 0, 0, 0, 1],
                "c4": [5, 0, 4, 2, 0],
                "c5": [5, 0, 0, 0, 0],
                "c6": [5, 0, 0, 0, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )
        p_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        joining_solver = NeighborJoiningSolver(
            dissimilarity_function=neg_hamming_similarity_without_missing,
            add_root=True,
        )
        psolver = PercolationSolver(joining_solver=joining_solver)
        psolver.solve(p_tree)
        T = p_tree.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(["c1", "c2", "c3", "c4", "c5", "c6", "6", "7", "8", "9"])
        expected_tree.add_edges_from(
            [
                ("6", "7"),
                ("6", "8"),
                ("7", "c2"),
                ("7", "c4"),
                ("8", "c1"),
                ("8", "c3"),
                ("8", "9"),
                ("9", "c5"),
                ("9", "c6"),
            ]
        )

        triplets = itertools.combinations(["c1", "c2", "c3", "c4", "c5", "c6"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_NJ_weighted_hamming_distance(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 4, 0, -1, 1],
                "c2": [-1, 2, 0, 0, 1],
                "c3": [0, 0, 0, 0, 1],
                "c4": [5, 3, 0, 0, 0],
                "c5": [5, 3, 0, 0, 0],
                "c6": [5, 0, 1, -1, 0],
                "c7": [5, 0, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )
        p_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        joining_solver = NeighborJoiningSolver(
            dissimilarity_function=dissimilarity_functions.weighted_hamming_distance,
            add_root=True,
        )
        psolver = PercolationSolver(joining_solver=joining_solver)
        psolver.solve(p_tree)
        T = p_tree.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7", 7, 8, 10, 11, 12]
        )
        expected_tree.add_edges_from(
            [
                (7, 8),
                (7, 10),
                (8, "c1"),
                (8, "c2"),
                (8, "c3"),
                (10, 11),
                (10, 12),
                (11, "c7"),
                (11, "c6"),
                (12, "c4"),
                (12, "c5"),
            ]
        )

        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_Greedy(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 4, -1, 2, 1],
                "c2": [-1, 2, 0, 0, 1],
                "c3": [0, 0, 0, 0, 1],
                "c4": [5, 3, 1, 0, 0],
                "c5": [5, 3, 1, 0, 0],
                "c6": [0, 3, 0, 2, 4],
                "c7": [5, 0, 1, 2, 4],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )
        p_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        joining_solver = VanillaGreedySolver()
        psolver = PercolationSolver(joining_solver=joining_solver)
        psolver.solve(p_tree)
        T = p_tree.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7", 7, 8, 10, 11]
        )
        expected_tree.add_edges_from(
            [
                (7, 8),
                (7, 10),
                (7, "c6"),
                (8, "c1"),
                (8, "c2"),
                (8, "c3"),
                (10, "c7"),
                (10, 11),
                (11, "c4"),
                (11, "c5"),
            ]
        )

        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_priors_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 2, 0, -1, 1],
                "c2": [-1, 2, 3, 0, 1],
                "c3": [0, 2, 0, 0, 1],
                "c4": [5, 0, 3, 0, 1],
                "c5": [5, 3, 3, -1, 0],
                "c6": [5, 3, 3, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {5: 0.5},
            1: {2: 0.6, 3: 0.4},
            2: {3: 0.1},
            3: {2: 0.5},
            4: {1: 0.6},
        }

        p_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )
        joining_solver = NeighborJoiningSolver(
            dissimilarity_function=dissimilarity_functions.weighted_hamming_distance,
            add_root=True,
        )
        psolver = PercolationSolver(joining_solver=joining_solver)
        psolver.solve(p_tree)
        # Due to the way that networkx finds connected components, the ordering
        # of nodes is uncertain
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from([6, 7, 8, 9, "c1", "c2", "c3", "c4", "c5"])
        expected_tree.add_edges_from(
            [
                (6, 7),
                (6, 8),
                (7, "c1"),
                (7, "c3"),
                (8, 9),
                (9, "c2"),
                (9, "c4"),
                (9, 10),
                (10, "c5"),
                (10, "c6"),
            ]
        )

        observed_tree = p_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6"], 3
        )
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
