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

        # mut_freqs = vgsolver.compute_mutation_frequencies(
        #     list(range(vgsolver.unique_character_matrix.shape[0]))
        # )
        left, right = vgsolver.perform_split(list(range(6)))

        self.assertListEqual(left, [3, 4, 5, 2])
        self.assertListEqual(right, [0, 1])

        vgsolver.solve()
        expected_newick_string = "(((2,5),(4,3)),(0,1));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_base_case_2(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )
        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char=-1)

        vgsolver.solve()
        expected_newick_string = "(((c4,c3),(c5,c6,c7)),(c1,c2));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_weighted_case_trivial(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {1: 0.5},
            1: {2: 0.5},
            2: {1: 0.5, 3: 0.5},
            3: {2: 0.5, 4: 0.5},
            4: {5: 0.5},
        }

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char=-1, priors=priors
        )

        vgsolver.solve()
        expected_newick_string = "(((c4,c3),(c5,c6,c7)),(c1,c2));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_priors_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
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
        expected_newick_string = "((c5,c6,c7),(c4,c3,(c1,c2)));"
        observed_newick_string = solver_utilities.to_newick(vgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
