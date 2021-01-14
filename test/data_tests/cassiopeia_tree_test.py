"""
Tests for the CassiopeiTree object in the data module.
"""
import unittest

import ete3
import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities


class TestCassiopeiaTree(unittest.TestCase):
    def setUp(self):

        # test_nwk and test_network should both have the same topology
        self.test_nwk = "((node3,(node7,(node9,(node11,(node13,(node15,(node17,node18)node16)node14)node12)node10)node8)node4)node1,(node5,node6)node2)node0;"
        self.test_network = nx.DiGraph()
        self.test_network.add_edges_from(
            [
                ("node0", "node1"),
                ("node0", "node2"),
                ("node1", "node3"),
                ("node1", "node4"),
                ("node2", "node5"),
                ("node2", "node6"),
                ("node4", "node7"),
                ("node4", "node8"),
                ("node8", "node9"),
                ("node8", "node10"),
                ("node10", "node11"),
                ("node10", "node12"),
                ("node12", "node13"),
                ("node12", "node14"),
                ("node14", "node15"),
                ("node14", "node16"),
                ("node16", "node17"),
                ("node16", "node18"),
            ]
        )

        self.character_matrix = pd.DataFrame.from_dict(
            {
                "node3": [0, 1, 1],
                "node7": [1, 1, 1],
                "node9": [0, 0, 0],
                "node11": [1, 1, 0],
                "node13": [1, 2, 0],
                "node15": [1, 0, 0],
                "node17": [0, 1, 0],
                "node18": [1, 1, 1],
                "node5": [2, 2, 2],
                "node6": [2, 3, 2],
            },
            orient="index",
        )

    def test_newick_to_networkx(self):

        network = data_utilities.newick_to_networkx(self.test_nwk)

        test_edges = [(u, v) for (u, v) in network.edges()]
        expected_edges = [(u, v) for (u, v) in self.test_network.edges()]
        for e in test_edges:
            self.assertIn(e, expected_edges)

    def test_newick_constructor(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_nwk
        )

        test_edges = tree.edges
        expected_edges = [(u, v) for (u, v) in self.test_network.edges()]
        for e in test_edges:
            self.assertIn(e, expected_edges)

        self.assertEqual(tree.n_cell, 10)
        self.assertEqual(tree.n_character, 3)

        test_nodes = tree.nodes
        expected_nodes = [u for u in self.test_network.nodes()]
        self.assertEqual(len(test_nodes), len(expected_nodes))
        for n in test_nodes:
            self.assertIn(n, expected_nodes)

        self.assertEqual(tree.root, "node0")

        obs_leaves = tree.leaves
        expected_leaves = [
            n for n in self.test_network if self.test_network.out_degree(n) == 0
        ]
        self.assertEqual(len(obs_leaves), len(expected_leaves))
        for l in obs_leaves:
            self.assertIn(l, expected_leaves)

        obs_internal_nodes = tree.internal_nodes
        expected_internal_nodes = [
            n for n in self.test_network if self.test_network.out_degree(n) > 0
        ]
        self.assertEqual(len(obs_internal_nodes), len(expected_internal_nodes))
        for n in obs_internal_nodes:
            self.assertIn(n, expected_internal_nodes)

        obs_nodes = tree.nodes
        expected_nodes = [n for n in self.test_network]
        self.assertEqual(len(obs_nodes), len(expected_nodes))
        for n in obs_nodes:
            self.assertIn(n, expected_nodes)

    def test_networkx_constructor(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        test_edges = tree.edges
        expected_edges = [(u, v) for (u, v) in self.test_network.edges()]
        for e in test_edges:
            self.assertIn(e, expected_edges)

        self.assertEqual(tree.n_cell, 10)
        self.assertEqual(tree.n_character, 3)

        test_nodes = tree.nodes
        expected_nodes = [u for u in self.test_network.nodes()]
        self.assertEqual(len(test_nodes), len(expected_nodes))
        for n in test_nodes:
            self.assertIn(n, expected_nodes)

        self.assertEqual(tree.root, "node0")

        obs_leaves = tree.leaves
        expected_leaves = [
            n for n in self.test_network if self.test_network.out_degree(n) == 0
        ]
        self.assertEqual(len(obs_leaves), len(expected_leaves))
        for l in obs_leaves:
            self.assertIn(l, expected_leaves)

        obs_internal_nodes = tree.internal_nodes
        expected_internal_nodes = [
            n for n in self.test_network if self.test_network.out_degree(n) > 0
        ]
        self.assertEqual(len(obs_internal_nodes), len(expected_internal_nodes))
        for n in obs_internal_nodes:
            self.assertIn(n, expected_internal_nodes)

        obs_nodes = tree.nodes
        expected_nodes = [n for n in self.test_network]
        self.assertEqual(len(obs_nodes), len(expected_nodes))
        for n in obs_nodes:
            self.assertIn(n, expected_nodes)


if __name__ == "__main__":
    unittest.main()
