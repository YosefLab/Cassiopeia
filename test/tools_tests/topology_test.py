"""
Test suite for the topology functions in
cassiopeia/tools/topology.py
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.mixins import CassiopeiaError
from cassiopeia.tools import topology


class TestTopology(unittest.TestCase):
    def setUp(self) -> None:

        tree = nx.DiGraph()
        tree.add_edge("0", "1")
        tree.add_edge("0", "2")
        tree.add_edge("1", "3")
        tree.add_edge("1", "4")
        tree.add_edge("1", "5")
        tree.add_edge("2", "6")
        tree.add_edge("2", "7")
        tree.add_edge("3", "8")
        tree.add_edge("3", "9")
        tree.add_edge("7", "10")
        tree.add_edge("7", "11")
        tree.add_edge("8", "12")
        tree.add_edge("8", "13")
        tree.add_edge("9", "14")
        tree.add_edge("9", "15")
        tree.add_edge("3", "16")
        tree.add_edge("16", "17")
        tree.add_edge("16", "18")

        self.tree = cas.data.CassiopeiaTree(tree=tree)

    def test_simple_choose_function(self):

        num_choices = topology.nCk(10, 2)

        self.assertEqual(num_choices, 45)

        self.assertRaises(CassiopeiaError, topology.nCk, 5, 7)

    def test_simple_coalescent_probability(self):

        N = 100
        B = 2
        K = 60
        coalescent_probability = topology.simple_coalescent_probability(N, B, K)
        self.assertAlmostEqual(coalescent_probability, 0.24, delta=0.01)

        self.assertRaises(
            CassiopeiaError, topology.simple_coalescent_probability, 50, 2, 60
        )

    def test_expansion_probability(self):

        # make sure attributes are instantiated correctly
        cas.tl.compute_expansion_probabilities(self.tree, min_clade_size=20)
        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            self.assertEqual(
                1.0, self.tree.get_attribute(node, "expansion_probability")
            )

        cas.tl.compute_expansion_probabilities(self.tree, min_clade_size=2)
        expected_probabilities = {
            "0": 1.0,
            "1": 1.0,
            "2": 1.0,
            "3": 0.047,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0,
            "7": 0.5,
            "8": 0.6,
            "9": 0.6,
            "10": 1.0,
            "11": 1.0,
            "12": 1.0,
            "13": 1.0,
            "14": 1.0,
            "15": 1.0,
            "16": 0.6,
            "17": 1.0,
            "18": 1.0,
        }

        for node in self.tree.depth_first_traverse_nodes(postorder=False):
            expected = expected_probabilities[node]
            self.assertAlmostEqual(
                expected,
                self.tree.get_attribute(node, "expansion_probability"),
                delta=0.01,
            )


if __name__ == "__main__":
    unittest.main()
