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
