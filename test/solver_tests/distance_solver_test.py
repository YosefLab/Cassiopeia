"""
Test DistanceSolver in Cassiopeia.solver.
"""
import unittest
from unittest import mock

import networkx as nx
import pandas as pd

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver.DistanceSolver import DistanceSolver


class DistanceSolverMock(DistanceSolver):
    def root_tree(self, *args, **kwargs): pass

    def find_cherry(self, *args, **kwargs): pass

    def update_dissimilarity_map(self, *args, **kwargs): pass

    def setup_root_finder(self, *args, **kwargs): pass


class TestDistanceSolver(unittest.TestCase):
    def setUp(self):

        cm = pd.DataFrame.from_dict(
            {
                "1": [0, 1, 2],
                "2": [1, 1, 2],
                "3": [2, 2, 2],
                "4": [1, 1, 1],
                "5": [0, 0, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        delta = pd.DataFrame.from_dict(
            {
                "1": [0, 17, 21, 31, 23],
                "2": [17, 0, 30, 34, 21],
                "3": [21, 30, 0, 28, 39],
                "4": [31, 34, 28, 0, 43],
                "5": [23, 21, 39, 43, 0],
            },
            orient="index",
            columns=["1", "2", "3", "4", "5"],
        )

        self.basic_dissimilarity_map = delta
        self.basic_tree = CassiopeiaTree(
            character_matrix=cm, dissimilarity_map=delta
        )

        self.distance_solver = DistanceSolverMock()
        self.distance_solver.root_tree = mock.MagicMock()
        self.distance_solver.find_cherry = mock.MagicMock()
        self.distance_solver.update_dissimilarity_map = mock.MagicMock()
        self.distance_solver.setup_root_finder = mock.MagicMock()

    def test_solve_doesnt_add_duplicate_node(self):
        self.basic_tree.root_sample_name = 'root'
        self.distance_solver.find_cherry.return_value = (0, 1)
        # Return dummy dissimilarity map with first dim = 2 to break the while loop.
        self.distance_solver.update_dissimilarity_map.return_value = self.basic_dissimilarity_map.iloc[[0, 1]]
        self.basic_tree.populate_tree = mock.MagicMock()
        self.distance_solver.solve(self.basic_tree)
        new_node_name = self.distance_solver.update_dissimilarity_map.call_args.args[2]
        self.assertNotIn(
            new_node_name, list(self.basic_tree.get_dissimilarity_map().index)
        )


if __name__ == "__main__":
    unittest.main()
