import unittest

import networkx as nx
import pandas as pd

import cassiopeia.solver.solver_utilities as s_utils


class TestComputeFrequency(unittest.TestCase):
    def test1(self):
        cm = pd.DataFrame(
            [
                ["5", "0", "1", "2", "-"],
                ["0", "0", "3", "2", "-"],
                ["-", "4", "0", "2", "2"],
                ["4", "4", "1", "2", "0"],
            ]
        )

        freq_dict = s_utils.compute_mutation_frequencies(cm)

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 2)
        self.assertEqual(len(freq_dict[2]), 3)
        self.assertEqual(len(freq_dict[3]), 1)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0]["5"], 1)
        self.assertEqual(freq_dict[1]["0"], 2)
        self.assertNotIn("3", freq_dict[1].keys())


if __name__ == "__main__":
    unittest.main()
