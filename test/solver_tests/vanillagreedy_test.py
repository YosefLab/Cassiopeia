import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver import solver_utilities


class VanillaGreedySolverTest(unittest.TestCase):
    def test_base_case_1(self):
        cm = pd.DataFrame(
            [
                [5, 0, 1, 2, 0],
                [5, 0, 0, 2, -1],
                [4, 0, 3, 2, -1],
                [-1, 4, 0, 2, 2],
                [0, 4, 1, 2, 2],
                [4, 0, 0, 2, 2],
            ]
        )

        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char=-1)

        mut_freqs = vgsolver.compute_mutation_frequencies()
        left, right = vgsolver.perform_split(mut_freqs, list(range(6)))

        self.assertListEqual(left, [3, 4, 5, 2])
        self.assertListEqual(right, [0, 1])

        vgsolver.solve()
        expected_newick_string = "(((2,5),(4,3)),(0,1));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_base_case_2(self):
        cm = pd.DataFrame(
            [
                [0, 0, 1, 2, 0],
                [0, 0, 1, 2, 0],
                [1, 2, 0, 2, -1],
                [1, 2, 3, 2, -1],
                [1, 0, 3, 4, 5],
                [1, 0, -1, 4, 5],
                [1, 0, -1, -1, 5],
            ]
        )

        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char=-1)

        vgsolver.solve()
        expected_newick_string = "(((2,1),(3,4,5)),0);"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_weighted_case_trivial(self):
        cm = pd.DataFrame(
            [
                [0, 0, 1, 2, 0],
                [0, 0, 1, 2, 0],
                [1, 2, 0, 2, -1],
                [1, 2, 3, 2, -1],
                [1, 0, 3, 4, 5],
                [1, 0, -1, 4, 5],
                [1, 0, -1, -1, 5],
            ]
        )

        weights = {
            0: {1: 1},
            1: {2: 1},
            2: {1: 1, 3: 1},
            3: {2: 1, 4: 1},
            4: {5: 1},
        }

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char=-1, weights=weights
        )

        vgsolver.solve()
        expected_newick_string = "(((2,1),(3,4,5)),0);"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_weighted_case_non_trivial(self):
        cm = pd.DataFrame(
            [
                [0, 0, 1, 2, 0],
                [0, 0, 1, 2, 0],
                [1, 2, 0, 2, -1],
                [1, 2, 3, 2, -1],
                [1, 0, 3, 4, 5],
                [1, 0, -1, 4, 5],
                [1, 0, -1, -1, 5],
            ]
        )

        weights = {
            0: {1: 1},
            1: {2: 1},
            2: {1: 2, 3: 3},
            3: {2: 1, 4: 1},
            4: {5: 2},
        }

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char=-1, weights=weights
        )

        vgsolver.solve()
        expected_newick_string = "(((3,4,5),2),(0,1));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_priors_case(self):
        cm = pd.DataFrame(
            [
                [0, 0, 1, 2, 0],
                [0, 0, 1, 2, 0],
                [1, 2, 0, 2, -1],
                [1, 2, 3, 2, -1],
                [1, 0, 3, 4, 5],
                [1, 0, -1, 4, 5],
                [1, 0, -1, -1, 5],
            ]
        )

        priors = {
            0: {1: 1},
            1: {2: 1},
            2: {1: 0.8, 3: 0.2},
            3: {2: 0.9, 4: 0.1},
            4: {5: 1},
        }

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char=-1, priors=priors
        )

        vgsolver.solve()
        expected_newick_string = "((3,4,5),(2,0,1));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
