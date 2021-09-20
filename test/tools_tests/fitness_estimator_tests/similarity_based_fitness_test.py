"""
Test BirthProcessFitnessEstimator in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import SimilarityBasedFitness


class TestSimilarityBasedFitness(unittest.TestCase):
    def test_basic(self):
        """
        Verified by hand.
        """
        tree = CassiopeiaTree(
            character_matrix=pd.DataFrame(
                {
                    'ch_1': [0, 0, 0, 1],
                    'ch_2': [1, 1, 0, 0],
                },
                index=["leaf_1", "leaf_2", "leaf_3", "leaf_4"]
            ),
            tree="(((leaf_1,leaf_2),leaf_3),leaf_4);"
        )
        fe = SimilarityBasedFitness(power=2.0)
        fe.estimate_fitness(tree)
        self.assertAlmostEqual(tree.get_attribute("leaf_1", "fitness"), 1 / 1 + 1 / 1 + 1 / 1.5 ** 2 + 1 / 2 ** 2)
        self.assertAlmostEqual(tree.get_attribute("leaf_2", "fitness"), 1 / 1 + 1 / 1 + 1 / 1.5 ** 2 + 1 / 2 ** 2)
        self.assertAlmostEqual(tree.get_attribute("leaf_3", "fitness"), 1 / 1 + 1 / 1.5 ** 2 + 1 / 1.5 ** 2 + 1 / 1.5 ** 2)
        self.assertAlmostEqual(tree.get_attribute("leaf_4", "fitness"), 1 / 1 + 1 / 1.5 ** 2 + 1 / 2 ** 2 + 1 / 2 ** 2)
