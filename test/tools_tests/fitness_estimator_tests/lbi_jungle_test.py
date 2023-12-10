"""
Test LBIJungle in cassiopeia.tools.
"""
import unittest

import networkx as nx

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import FitnessEstimatorError, LBIJungle


class TestLBIJungle(unittest.TestCase):
    def test_small_tree(self):
        """
        Run LBI jungle on small tree and see that fitness estimates make sense.
        """
        tree = nx.DiGraph()
        nodes = [
            "root",
            "internal-1",
            "internal-2",
            "internal-3",
            "leaf-1",
            "leaf-2",
            "leaf-3",
            "leaf-4",
            "leaf-5",
        ]
        tree.add_nodes_from(nodes)
        tree.add_edges_from(
            [
                ("root", "internal-1"),
                ("internal-1", "internal-2"),
                ("internal-1", "internal-3"),
                ("internal-2", "leaf-1"),
                ("internal-2", "leaf-2"),
                ("internal-2", "leaf-3"),
                ("internal-3", "leaf-4"),
                ("internal-3", "leaf-5"),
            ]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {
                "root": 0.0,
                "internal-1": 0.25,
                "internal-2": 0.5,
                "internal-3": 0.5,
                "leaf-1": 1.0,
                "leaf-2": 1.0,
                "leaf-3": 1.0,
                "leaf-4": 1.0,
                "leaf-5": 1.0,
            }
        )
        fitness_estimator = LBIJungle()
        fitness_estimator.estimate_fitness(tree)
        fitness_estimates = {
            node: tree.get_attribute(node, "fitness")
            for node in nodes
            if node != tree.root  # LBIJungle doesn't report root fitness.
        }
        # internal node 2 has strictly more branching than internal node 3, so
        # fitness estimate should be higher
        self.assertGreater(
            fitness_estimates["internal-2"], fitness_estimates["internal-3"]
        )
        # Leaves 1, 2, 3 should have the same fitness
        self.assertAlmostEqual(
            fitness_estimates["leaf-1"], fitness_estimates["leaf-2"]
        )
        self.assertAlmostEqual(
            fitness_estimates["leaf-2"], fitness_estimates["leaf-3"]
        )
        # Leaves 4, 5 should have the same fitness
        self.assertAlmostEqual(
            fitness_estimates["leaf-4"], fitness_estimates["leaf-5"]
        )
        # Leaves 1, 2, 3 should have higher fitness than leaves 4, 5
        self.assertGreater(
            fitness_estimates["leaf-1"], fitness_estimates["leaf-4"]
        )
        # Leaves should have lower fitness than their parent (by LBI property)
        self.assertGreater(
            fitness_estimates["internal-2"], fitness_estimates["leaf-1"]
        )
        self.assertGreater(
            fitness_estimates["internal-3"], fitness_estimates["leaf-4"]
        )

    def test_raises_error_if_leaf_name_startswith_underscore(self):
        """
        Leaf names cannot start with an underscore.

        (This is due to the underlying Jungle implementation we wrap.)
        """
        tree = nx.DiGraph()
        nodes = [
            "root",
            "_leaf",
        ]
        tree.add_nodes_from(nodes)
        tree.add_edges_from(
            [
                ("root", "_leaf"),
            ]
        )
        tree = CassiopeiaTree(tree=tree)
        fitness_estimator = LBIJungle()
        with self.assertRaises(FitnessEstimatorError):
            fitness_estimator.estimate_fitness(tree)

if __name__ == "__main__":
    unittest.main()
