"""
Test BirthProcessFitnessEstimator in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import BirthProcessFitnessEstimator


class TestBirthProcessFitnessEstimator(unittest.TestCase):
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
        fe = BirthProcessFitnessEstimator(smooth=False)
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("0", "fitness"), 2 / 10)
        self.assertAlmostEqual(tree.get_attribute("1", "fitness"), 1 / 7)
        self.assertAlmostEqual(tree.get_attribute("2", "fitness"), 1 / 2)
        self.assertAlmostEqual(tree.get_attribute("3", "fitness"), 1 / 2)
        self.assertAlmostEqual(tree.get_attribute("4", "fitness"), 1 / 2)
        self.assertAlmostEqual(tree.get_attribute("5", "fitness"), 1 / 3)

        f0_smooth = 2 / 10
        f1_smooth = (1 / 7 + 2 / 10) / 2
        f2_smooth = (1 / 2 + 1 / 7 + 2 / 10) / 3
        f3_smooth = (1 / 2 + 1 / 2 + 1 / 7 + 2 / 10) / 4
        f4_smooth = (1 / 2 + 1 / 2 + 1 / 7 + 2 / 10) / 4
        f5_smooth = (1 / 3 + 1 / 7 + 2 / 10) / 3
        fe = BirthProcessFitnessEstimator(smooth=True)
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("0", "fitness"), f0_smooth)
        self.assertAlmostEqual(tree.get_attribute("1", "fitness"), f1_smooth)
        self.assertAlmostEqual(tree.get_attribute("2", "fitness"), f2_smooth)
        self.assertAlmostEqual(
            tree.get_attribute("3", "fitness"),
            f3_smooth,
        )
        self.assertAlmostEqual(
            tree.get_attribute("4", "fitness"),
            f4_smooth,
        )
        self.assertAlmostEqual(tree.get_attribute("5", "fitness"), f5_smooth)

        fe = BirthProcessFitnessEstimator(
            smooth=True, leaf_average_for_internal_nodes=True
        )
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(
            tree.get_attribute("0", "fitness"),
            (f3_smooth + f4_smooth + f5_smooth) / 3.0,
        )
        self.assertAlmostEqual(
            tree.get_attribute("1", "fitness"),
            (f3_smooth + f4_smooth + f5_smooth) / 3.0,
        )
        self.assertAlmostEqual(
            tree.get_attribute("2", "fitness"), (f3_smooth + f4_smooth) / 2.0
        )
        self.assertAlmostEqual(
            tree.get_attribute("3", "fitness"),
            f3_smooth,
        )
        self.assertAlmostEqual(
            tree.get_attribute("4", "fitness"),
            f4_smooth,
        )
        self.assertAlmostEqual(tree.get_attribute("5", "fitness"), f5_smooth)
