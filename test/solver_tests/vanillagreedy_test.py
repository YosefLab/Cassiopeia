import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver


class TestPerformSplit(unittest.TestCase):
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

        vgsolver = VanillaGreedySolver(
            character_matrix=cm, missing_char="-", missing_data_classifier=None
        )

        left, right = vgsolver.perform_split()

        self.assertListEqual(left, [3, 4, 5, 2])
        self.assertListEqual(right, [0, 1])


if __name__ == "__main__":
    unittest.main()
