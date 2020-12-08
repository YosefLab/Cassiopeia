import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver


class VanillaGreedySolverTest(unittest.TestCase):
    def test1(self):
        cm = pd.DataFrame(
            [
                ["5", "0", "1", "2", "0"],
                ["5", "0", "0", "2", "-"],
                ["4", "0", "3", "2", "-"],
                ["-", "4", "0", "2", "2"],
                ["0", "4", "1", "2", "2"],
                ["4", "0", "0", "2", "2"],
            ]
        )

        vgsolver = VanillaGreedySolver(character_matrix=cm, missing_char="-")

        mut_freqs = vgsolver.compute_mutation_frequencies()
        left, right = vgsolver.perform_split(mut_freqs, list(range(6)))

        self.assertListEqual(left, [3, 4, 5, 2])
        self.assertListEqual(right, [0, 1])

        vgsolver.solve()
        self.assertIn((6, 7), vgsolver.tree.edges)
        self.assertIn((6, 10), vgsolver.tree.edges)
        self.assertIn((7, 8), vgsolver.tree.edges)
        self.assertIn((7, 9), vgsolver.tree.edges)
        self.assertIn((8, 2), vgsolver.tree.edges)
        self.assertIn((8, 5), vgsolver.tree.edges)
        self.assertIn((9, 3), vgsolver.tree.edges)
        self.assertIn((9, 4), vgsolver.tree.edges)
        self.assertIn((10, 0), vgsolver.tree.edges)
        self.assertIn((10, 1), vgsolver.tree.edges)

    def test2(self):
        cm = pd.DataFrame(
            [
                ["0", "0", "1", "2", "0"],
                ["0", "0", "1", "2", "0"],
                ["1", "2", "0", "2", "-"],
                ["1", "2", "3", "2", "-"],
                ["1", "0", "3", "4", "5"],
                ["1", "0", "-", "4", "5"],
                ["1", "0", "-", "-", "5"],
            ]
        )

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char="-", missing_data_classifier=None
        )

        vgsolver.solve()
        self.assertIn((6, 7), vgsolver.tree.edges)
        self.assertIn((6, 0), vgsolver.tree.edges)
        self.assertIn((7, 8), vgsolver.tree.edges)
        self.assertIn((7, 9), vgsolver.tree.edges)
        self.assertIn((8, 2), vgsolver.tree.edges)
        self.assertIn((8, 1), vgsolver.tree.edges)
        self.assertIn((9, 3), vgsolver.tree.edges)
        self.assertIn((9, 4), vgsolver.tree.edges)
        self.assertIn((9, 5), vgsolver.tree.edges)


if __name__ == "__main__":
    unittest.main()
