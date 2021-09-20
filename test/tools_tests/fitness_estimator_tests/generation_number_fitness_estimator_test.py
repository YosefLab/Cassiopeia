"""
Test BirthProcessFitnessEstimator in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import GenerationNumberFitnessEstimator


class TestGenerationNumberFitnessEstimator(unittest.TestCase):
    def test_basic(self):
        """
        Verified by hand.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5"]),
        tree.add_edges_from(
            [("0", "1"), ("1", "2"), ("2", "3"), ("2", "4"), ("1", "5")]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {
                "0": 0,
                "1": 3,
                "2": 5,
                "3": 6,
                "4": 6,
                "5": 6,
            }
        )
        fe = GenerationNumberFitnessEstimator()
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("0", "fitness"), (3 / 6 + 3 / 6 + 2 / 6) / 3)
        self.assertAlmostEqual(tree.get_attribute("1", "fitness"), 1 / 3)
        self.assertAlmostEqual(tree.get_attribute("2", "fitness"), 2 / 5)
        self.assertAlmostEqual(tree.get_attribute("3", "fitness"), 3 / 6)
        self.assertAlmostEqual(tree.get_attribute("4", "fitness"), 3 / 6)
        self.assertAlmostEqual(tree.get_attribute("5", "fitness"), 2 / 6)
