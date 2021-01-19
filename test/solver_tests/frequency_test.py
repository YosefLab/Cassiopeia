import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver


class TestComputeFrequency(unittest.TestCase):
    def test_basic(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, 2, 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char=-1)
        freq_dict = vgsolver.compute_mutation_frequencies(
            list(range(vgsolver.unique_character_matrix.shape[0]))
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 3)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 2)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[2][-1], 0)
        self.assertNotIn(3, freq_dict[1].keys())

    def test_duplicate(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, -1],
                "c2": [5, 0, 1, 2, -1],
                "c3": [0, 0, 3, 2, -1],
                "c4": [-1, 4, 0, 2, 2],
                "c5": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char=-1)
        freq_dict = vgsolver.compute_mutation_frequencies(
            list(range(vgsolver.unique_character_matrix.shape[0]))
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 3)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 2)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[2][-1], 0)
        self.assertNotIn(3, freq_dict[1].keys())


if __name__ == "__main__":
    unittest.main()
