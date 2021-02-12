"""
Test PercolationSolver in Cassiopeia.solver.
"""

import unittest

import networkx as nx
import pandas as pd

from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver.PercolationSolver import PercolationSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class PercolationSolverTest(unittest.TestCase):
    def test_simple_base_case1(self):
        cm = pd.DataFrame(
            [
                [5, 3, 0, 0, 0],
                [0, 3, 4, 2, 1],
                [5, 0, 0, 0, 1],
                [5, 0, 4, 2, 0],
                [5, 0, 0, 0, 0],
            ]
        )
        psolver = PercolationSolver(character_matrix=cm, missing_char=-1)
        psolver.solve()
        expected_newick_string = "((1,3),(2,4,0));"
        observed_newick_string = data_utilities.to_newick(psolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_simple_base_case2(self):
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
        psolver = PercolationSolver(character_matrix=cm, missing_char=-1)
        psolver.solve()
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
            self.assertIn(i, psolver.tree.edges)

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

        psolver = PercolationSolver(
            character_matrix=cm, missing_char=-1, priors=priors
        )
        psolver.solve()
        # Due to the way that networkx finds connected components, the ordering
        # of nodes is uncertain
        expected_newick_strings = [
            "((c2,c4,(c5,c6)),(c1,c3));",
            "((c2,c4,(c6,c5)),(c1,c3));",
            "((c2,(c5,c6),c4),(c1,c3));",
            "((c2,(c6,c5),c4),(c1,c3));",
        ]
        observed_newick_string = data_utilities.to_newick(psolver.tree)
        self.assertIn(observed_newick_string, expected_newick_strings)


if __name__ == "__main__":
    unittest.main()
