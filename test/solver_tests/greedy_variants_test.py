import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.SpectralGreedySolver import SpectralGreedySolver
from cassiopeia.solver.MaxCutGreedySolver import MaxCutGreedySolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class GreedyVariantsTest(unittest.TestCase):
    def test_spectral_base_case(self):
        cm = pd.DataFrame(
            [
                ["5", "3", "0", "0", "0"],
                ["0", "3", "4", "2", "1"],
                ["5", "0", "0", "0", "1"],
                ["5", "0", "4", "2", "0"],
            ]
        )

        sgsolver = SpectralGreedySolver(character_matrix=cm, missing_char="-")
        freq_dict = sgsolver.compute_mutation_frequencies()
        left, right = sgsolver.perform_split(freq_dict, list(range(4)))
        self.assertListEqual(left, [0, 2])
        self.assertListEqual(right, [1, 3])

        sgsolver.solve()
        expected_newick_string = "((0,2),(3,1));"
        observed_newick_string = solver_utilities.to_newick(sgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_spectral_base_case_weights_trivial(self):
        cm = pd.DataFrame(
            [
                ["5", "3", "0", "0", "0"],
                ["0", "3", "4", "2", "1"],
                ["5", "0", "0", "0", "1"],
                ["5", "0", "4", "2", "0"],
            ]
        )

        weights = {
            0: {"5": 1},
            1: {"3": 1},
            2: {"4": 1},
            3: {"2": 1},
            4: {"1": 1},
        }

        sgsolver = SpectralGreedySolver(
            character_matrix=cm, missing_char="-", priors=weights
        )
        freq_dict = sgsolver.compute_mutation_frequencies()
        left, right = sgsolver.perform_split(freq_dict, list(range(4)))
        self.assertListEqual(left, [0, 2])
        self.assertListEqual(right, [1, 3])

        sgsolver.solve()
        expected_newick_string = "((0,2),(3,1));"
        observed_newick_string = solver_utilities.to_newick(sgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_maxcut_base_case(self):
        # A case in which the connectivity only has negative weights, so the
        # hill-climbing procedure favors a cut with 0 weight
        cm = pd.DataFrame(
            [
                ["5", "3", "0", "0", "0"],
                ["0", "3", "4", "2", "1"],
                ["5", "0", "0", "0", "1"],
                ["5", "0", "4", "2", "0"],
            ]
        )

        mcgsolver = MaxCutGreedySolver(character_matrix=cm, missing_char="-")
        freq_dict = mcgsolver.compute_mutation_frequencies()
        left, right = mcgsolver.perform_split(freq_dict, list(range(4)))
        self.assertListEqual(left, [0, 2, 3, 1])
        self.assertListEqual(right, [])

        mcgsolver.solve()
        expected_newick_string = "(0,2,3,1);"
        observed_newick_string = solver_utilities.to_newick(mcgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_maxcut_base_case_weights_trivial(self):
        # A case in which the connectivity only has negative weights, so the
        # hill-climbing procedure favors a cut with 0 weight
        cm = pd.DataFrame(
            [
                ["5", "3", "0", "0", "0"],
                ["0", "3", "4", "2", "1"],
                ["5", "0", "0", "0", "1"],
                ["5", "0", "4", "2", "0"],
            ]
        )

        weights = {
            0: {"5": 1},
            1: {"3": 1},
            2: {"4": 1},
            3: {"2": 1},
            4: {"1": 1},
        }

        mcgsolver = MaxCutGreedySolver(
            character_matrix=cm, missing_char="-", priors=weights
        )
        freq_dict = mcgsolver.compute_mutation_frequencies()
        left, right = mcgsolver.perform_split(freq_dict, list(range(4)))
        self.assertListEqual(left, [0, 2, 3, 1])
        self.assertListEqual(right, [])

        mcgsolver.solve()
        expected_newick_string = "(0,2,3,1);"
        observed_newick_string = solver_utilities.to_newick(mcgsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
