"""
Test PercolationSolver in Cassiopeia.solver.
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver.NeighborJoiningSolver import NeighborJoiningSolver
from cassiopeia.solver.PercolationSolver import PercolationSolver
from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities

def neg_hamming_similarity_without_missing(
    s1: np.array,
    s2: np.array,
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """A function to return the number of (non-missing) character/state
    mutations shared by two samples.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a mutation
    Returns:
        The number of shared mutations between two samples, weighted or unweighted
    """

    # TODO Optimize this using masks
    similarity = 0
    for i in range(len(s1)):

        if (
            s1[i] == missing_state_indicator
            or s2[i] == missing_state_indicator
            or s1[i] == 0
            or s2[i] == 0
        ):
            continue

        if s1[i] == s2[i]:
            if weights:
                similarity += weights[i][s1[i]]
            else:
                similarity += 1

    return -1 * similarity


class PercolationSolverTest(unittest.TestCase):
    def test_NJ_negative_similarity(self):

        cm = pd.DataFrame(
            [
                [5, 3, 0, 0, 0],
                [0, 3, 4, 2, 1],
                [5, 0, 0, 0, 1],
                [5, 0, 4, 2, 0],
                [5, 0, 0, 0, 0],
                [5, 0, 0, 0, 0],
            ]
        )
        p_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1
        )

        joining_solver = NeighborJoiningSolver(dissimilarity_function = neg_hamming_similarity_without_missing, add_root = True)
        psolver = PercolationSolver(joining_solver = joining_solver)
        psolver.solve(p_tree)
        expected_newick_string = "((1,3),(2,0,(4,5)));"
        observed_newick_string = p_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

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
        p_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1
        )

        joining_solver = NeighborJoiningSolver(dissimilarity_function = dissimilarity_functions.weighted_hamming_distance, add_root = True)
        psolver = PercolationSolver(joining_solver = joining_solver)
        psolver.solve(p_tree)
        # Due to the way that networkx finds connected components, the ordering
        # of nodes is uncertain
        expected_edges = [
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
        for i in expected_edges:
            self.assertIn(i, p_tree.get_tree_topology().edges)

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
        p_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1
        )

        joining_solver = VanillaGreedySolver()
        psolver = PercolationSolver(joining_solver = joining_solver)
        psolver.solve(p_tree)
        # Due to the way that networkx finds connected components, the ordering
        # of nodes is uncertain
        expected_edges = [
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
        for i in expected_edges:
            self.assertIn(i, p_tree.get_tree_topology().edges)

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
        joining_solver = NeighborJoiningSolver(dissimilarity_function = dissimilarity_functions.weighted_hamming_distance, add_root = True)
        psolver = PercolationSolver(joining_solver = joining_solver)
        psolver.solve(p_tree)
        # Due to the way that networkx finds connected components, the ordering
        # of nodes is uncertain
        expected_newick_strings = [
            "((c2,c4,(c5,c6)),(c1,c3));",
            "((c2,c4,(c6,c5)),(c1,c3));",
            "((c2,(c5,c6),c4),(c1,c3));",
            "((c2,(c6,c5),c4),(c1,c3));",
        ]
        observed_newick_string = data_utilities.to_newick(p_tree.get_tree_topology())
        self.assertIn(observed_newick_string, expected_newick_strings)

if __name__ == "__main__":
    unittest.main()
