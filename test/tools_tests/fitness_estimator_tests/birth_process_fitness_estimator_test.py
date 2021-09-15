"""
Test BirthProcessFitnessEstimator in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import BirthProcessFitnessEstimator


class TestBirthProcessFitnessEstimator(unittest.TestCase):
    def test_no_mutations(self):
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
        fe = BirthProcessFitnessEstimator(smooth=False)
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("0", "fitness"), 2 / 10)
        self.assertAlmostEqual(tree.get_attribute("1", "fitness"), 1 / 7)
        self.assertAlmostEqual(tree.get_attribute("2", "fitness"), 2 / 4)
        self.assertAlmostEqual(tree.get_attribute("3", "fitness"), 1 / 2)
        self.assertAlmostEqual(tree.get_attribute("4", "fitness"), 1 / 2)
        self.assertAlmostEqual(tree.get_attribute("5", "fitness"), 1 / 6)

        fe = BirthProcessFitnessEstimator(smooth=True)
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("0", "fitness"), 2 / 10)
        self.assertAlmostEqual(
            tree.get_attribute("1", "fitness"), (1 / 7 + 2 / 10) / 2
        )
        self.assertAlmostEqual(
            tree.get_attribute("2", "fitness"), (2 / 4 + 1 / 7 + 2 / 10) / 3
        )
        self.assertAlmostEqual(
            tree.get_attribute("3", "fitness"),
            (1 / 2 + 2 / 4 + 1 / 7 + 2 / 10) / 4,
        )
        self.assertAlmostEqual(
            tree.get_attribute("4", "fitness"),
            (1 / 2 + 2 / 4 + 1 / 7 + 2 / 10) / 4,
        )
        self.assertAlmostEqual(
            tree.get_attribute("5", "fitness"), (1 / 6 + 1 / 7 + 2 / 10) / 3
        )
