"""
Test IIDExponentialMLE in cassiopeia.tools.
"""
import unittest

import networkx as nx

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import ZeroOneBLE


class TestZeroOneBLE(unittest.TestCase):
    def test_basic(self):
        """
        Small tree verified by hand.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0], "1": [0, 1, 0], "2": [0, 1, 0], "3": [0, 1, -1]}
        )
        ble = ZeroOneBLE()
        ble.estimate_branch_lengths(tree)
        self.assertEqual(tree.get_times(), {"0": 0, "1": 1, "2": 1, "3": 2})
        ble = ZeroOneBLE(include_missing=False)
        ble.estimate_branch_lengths(tree)
        self.assertEqual(tree.get_times(), {"0": 0, "1": 1, "2": 1, "3": 1})

    def test_basic_2(self):
        """
        Small tree verified by hand, but larger than in test_basic.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        tree.add_edges_from(
            [
                ("0", "1"),
                ("1", "2"),
                ("1", "3"),
                ("2", "4"),
                ("2", "5"),
                ("3", "6"),
                ("3", "7"),
                ("7", "8"),
                ("7", "9")
            ]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "2": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "3": [1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                "4": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                "5": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "6": [1, 0, 0, -1, 1, 1, 0, 0, 0, 1],
                "7": [1, 0, 0, -1, 0, 0, 1, 1, 0, 1],
                "8": [1, 0, 0, -1, 0, 0, 1, 1, 0, 1],
                "9": [1, 0, 0, -1, 0, 0, 1, 1, 1, 1],
            }
        )
        ble = ZeroOneBLE()
        ble.estimate_branch_lengths(tree)
        self.assertEqual(
            tree.get_times(),
            {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 2,
                "4": 2,
                "5": 2,
                "6": 3,
                "7": 3,
                "8": 3,
                "9": 4,
            }
        )
