"""
Tests for the CassiopeiaTree object in the data module.
"""
import unittest

import ete3
import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.data.CassiopeiaTree import CassiopeiaTreeError


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

        # this should obey PP for easy checking of ancestral states
        self.character_matrix = pd.DataFrame.from_dict(
            {
                "node3": [1, 0, 0, 0, 0, 0, 0, 0],
                "node7": [1, 1, 0, 0, 0, 0, 0, 0],
                "node9": [1, 1, 1, 0, 0, 0, 0, 0],
                "node11": [1, 1, 1, 1, 0, 0, 0, 0],
                "node13": [1, 1, 1, 1, 1, 0, 0, 0],
                "node15": [1, 1, 1, 1, 1, 1, 0, 0],
                "node17": [1, 1, 1, 1, 1, 1, 1, 0],
                "node18": [1, 1, 1, 1, 1, 1, 1, 1],
                "node5": [2, 0, 0, 0, 0, 0, 0, 0],
                "node6": [2, 2, 0, 0, 0, 0, 0, 0],
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
        self.assertEqual(tree.n_character, 8)

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
        self.assertEqual(tree.n_character, 8)

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

    def test_get_children(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        obs_children = tree.children("node14")
        expected_children = ["node15", "node16"]
        self.assertCountEqual(obs_children, expected_children)

        obs_children = tree.children("node5")
        self.assertEqual(len(obs_children), 0)

    def test_character_state_assignments_at_leaves(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_nwk
        )

        obs_states = tree.get_character_states("node5")
        expected_states = self.character_matrix.loc["node5"].to_list()
        self.assertCountEqual(obs_states, expected_states)

        obs_state = tree.get_character_states("node3")[0]
        self.assertEqual(obs_state, 1)

        obs_states = tree.get_character_states("node0")
        self.assertCountEqual(obs_states, [])

    def test_root_and_leaf_indicators(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        self.assertTrue(tree.is_root("node0"))
        self.assertFalse(tree.is_root("node5"))

        self.assertTrue(tree.is_leaf("node5"))
        self.assertFalse(tree.is_leaf("node10"))

    def test_depth_first_traversal(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        obs_ordering = tree.depth_first_traverse_nodes(
            source="node0", postorder=True
        )
        expected_ordering = [
            "node3",
            "node7",
            "node9",
            "node11",
            "node13",
            "node15",
            "node17",
            "node18",
            "node16",
            "node14",
            "node12",
            "node10",
            "node8",
            "node4",
            "node1",
            "node5",
            "node6",
            "node2",
            "node0",
        ]
        self.assertCountEqual([n for n in obs_ordering], expected_ordering)

        obs_ordering = tree.depth_first_traverse_nodes(
            source="node14", postorder=True
        )
        expected_ordering = ["node15", "node17", "node18", "node16", "node14"]
        self.assertCountEqual([n for n in obs_ordering], expected_ordering)

        obs_ordering = tree.depth_first_traverse_nodes(
            source="node0", postorder=False
        )
        expected_ordering = [
            "node0",
            "node1",
            "node3",
            "node4",
            "node7",
            "node8",
            "node9",
            "node10",
            "node11",
            "node12",
            "node13",
            "node14",
            "node15",
            "node16",
            "node17",
            "node18",
            "node2",
            "node5",
            "node6",
        ]
        self.assertCountEqual([n for n in obs_ordering], expected_ordering)

    def test_depth_first_traversal_edges(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        obs_ordering = tree.depth_first_traverse_edges(source="node0")
        expected_ordering = [
            ("node0", "node1"),
            ("node1", "node3"),
            ("node1", "node4"),
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
            ("node0", "node2"),
            ("node2", "node5"),
            ("node2", "node6"),
        ]
        self.assertCountEqual(obs_ordering, expected_ordering)

    def test_get_leaves_in_subtree(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        obs_leaves = tree.leaves_in_subtree("node0")
        self.assertCountEqual(obs_leaves, tree.leaves)

        obs_leaves = tree.leaves_in_subtree("node14")
        expected_leaves = ["node15", "node17", "node18"]
        self.assertCountEqual(obs_leaves, expected_leaves)

    def test_reconstruct_ancestral_states(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        tree.reconstruct_ancestral_characters()

        self.assertCountEqual(tree.get_character_states("node0"), [0,0,0,0,0,0,0,0])
        self.assertCountEqual(tree.get_character_states("node2"), [2, 0, 0, 0, 0, 0, 0, 0])
        self.assertCountEqual(tree.get_character_states("node10"), [1, 1, 1, 1, 0, 0, 0, 0])

    def test_get_mutations_along_edge(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        tree.reconstruct_ancestral_characters()

        edge_of_interest = ('node4', 'node8')
        expected_mutations = [(3, 1)]
        observed_mutations = tree.get_mutations_along_edge('node4', 'node8')

        self.assertCountEqual(expected_mutations, observed_mutations)

        self.assertRaises(CassiopeiaTreeError, tree.get_mutations_along_edge, 'node4', 'node6')

    def test_depth_calculations_on_tree(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        mean_depth = tree.get_mean_depth_of_tree()
        self.assertEqual(mean_depth, 4.7)

        max_depth = tree.get_max_depth_of_tree()
        self.assertEqual(max_depth, 8)

    def test_relabel_nodes_in_network(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        relabel_map = {"node0": 'root', 'node1': 'child1', 'node2': 'child2'}
        tree.relabel_nodes(relabel_map)

        self.assertIn('root', tree.nodes)
        self.assertNotIn('node0', tree.nodes)

        expected_children = ['child1', 'child2']
        observed_children = tree.children('root')
        self.assertCountEqual(expected_children, observed_children)

        self.assertIn("node8", tree.nodes)

if __name__ == "__main__":
    unittest.main()
