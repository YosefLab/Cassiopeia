"""
Tests for the CassiopeiaTree object in the data module.
"""
import unittest
from typing import Dict, Optional

import ete3
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.data.CassiopeiaTree import (
    CassiopeiaTree,
    CassiopeiaTreeError,
    CassiopeiaTreeWarning,
)


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

        # A simple balanced binary tree to test lineage pruning
        complete_binary = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        str_names = dict(
            zip(
                list(complete_binary.nodes),
                ["node" + str(i) for i in complete_binary.nodes],
            )
        )
        self.simple_complete_binary_tree = nx.relabel_nodes(
            complete_binary, str_names
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

    def test_construction_without_character_matrix(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        test_edges = tree.edges
        expected_edges = [(u, v) for (u, v) in self.test_network.edges()]
        for e in test_edges:
            self.assertIn(e, expected_edges)

        self.assertEqual(tree.n_cell, 10)
        self.assertEqual(tree.n_character, 0)

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

        self.assertCountEqual(
            tree.get_character_states("node0"), [0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertCountEqual(
            tree.get_character_states("node2"), [2, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertCountEqual(
            tree.get_character_states("node10"), [1, 1, 1, 1, 0, 0, 0, 0]
        )

    def test_get_mutations_along_edge(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        tree.reconstruct_ancestral_characters()

        edge_of_interest = ("node4", "node8")
        expected_mutations = [(2, 1)]
        observed_mutations = tree.get_mutations_along_edge("node4", "node8")

        self.assertCountEqual(expected_mutations, observed_mutations)

        self.assertRaises(
            CassiopeiaTreeError, tree.get_mutations_along_edge, "node4", "node6"
        )

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

        relabel_map = {"node0": "root", "node1": "child1", "node2": "child2"}
        tree.relabel_nodes(relabel_map)

        self.assertIn("root", tree.nodes)
        self.assertNotIn("node0", tree.nodes)

        expected_children = ["child1", "child2"]
        observed_children = tree.children("root")
        self.assertCountEqual(expected_children, observed_children)

        self.assertIn("node8", tree.nodes)

    def test_change_time_of_node(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        self.assertEqual(tree.get_time("node16"), 7)

        self.assertRaises(CassiopeiaTreeError, tree.set_time, "node16", 20)

        tree.set_time("node16", 7.5)
        self.assertEqual(tree.get_time("node16"), 7.5)
        self.assertEqual(tree.get_time("node17"), 8)
        self.assertEqual(tree.get_time("node18"), 8)

        # make sure edges are adjusted accordingly
        self.assertEqual(tree.get_branch_length("node14", "node16"), 1.5)
        self.assertEqual(tree.get_branch_length("node16", "node17"), 0.5)

        self.assertRaises(CassiopeiaTreeError, tree.set_time, "node14", 1)

    def test_change_branch_length(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        self.assertEqual(tree.get_branch_length("node12", "node14"), 1)

        tree.set_branch_length("node12", "node14", 0.5)

        # make sure nodes affected have adjusted their edges
        node_to_time = {
            "node14": 5.5,
            "node15": 6.5,
            "node16": 6.5,
            "node17": 7.5,
            "node18": 7.5,
        }
        for n in node_to_time:
            self.assertEqual(node_to_time[n], tree.get_time(n))

        self.assertRaises(
            CassiopeiaTreeError, tree.set_branch_length, "node14", "node6", 10.0
        )
        self.assertRaises(
            CassiopeiaTreeError, tree.set_branch_length, "node12", "node14", -1
        )

    def test_set_states_with_pandas_dataframe(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        self.assertRaises(
            CassiopeiaTreeError,
            tree.set_character_states,
            "node5",
            [2, 0, 3, 0, 0, 0, 0, 0],
        )

        self.assertRaises(
            CassiopeiaTreeError,
            tree.initialize_character_states_at_leaves,
            {"node5": [2, 0, 3, 0, 0, 0, 0, 0]},
        )

        tree.initialize_character_states_at_leaves(self.character_matrix)

        self.assertCountEqual(
            tree.get_character_states("node5"), [2, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertCountEqual(tree.get_character_states("node0"), [])
        tree.set_character_states("node0", [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertCountEqual(
            tree.get_character_states("node0"), [0, 0, 0, 0, 0, 0, 0, 0]
        )

        observed_character_matrix = tree.get_original_character_matrix()
        pd.testing.assert_frame_equal(
            observed_character_matrix, self.character_matrix
        )

        tree.set_character_states("node5", [2, 0, 3, 0, 0, 0, 0, 0])
        self.assertCountEqual(
            tree.get_character_states("node5"), [2, 0, 3, 0, 0, 0, 0, 0]
        )

        observed_character_matrix = tree.get_original_character_matrix()
        pd.testing.assert_frame_equal(
            observed_character_matrix, self.character_matrix
        )

        observed_character_matrix = tree.get_current_character_matrix()
        expected_character_matrix = self.character_matrix.copy()
        expected_character_matrix.loc["node5"] = [2, 0, 3, 0, 0, 0, 0, 0]
        pd.testing.assert_frame_equal(
            observed_character_matrix, expected_character_matrix
        )

    def test_set_states_with_dictionary(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        character_dictionary = {}
        for ind in self.character_matrix.index:
            character_dictionary[ind] = self.character_matrix.loc[ind].tolist()

        tree.initialize_character_states_at_leaves(character_dictionary)
        self.assertCountEqual(
            tree.get_character_states("node5"), [2, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertCountEqual(tree.get_character_states("node0"), [])

    def test_set_all_states(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        self.assertRaises(
            CassiopeiaTreeError,
            tree.initialize_all_character_states,
            {"node0": [0, 0, 0, 0, 0, 0, 0, 0]},
        )

        character_mapping = {}
        for node in tree.nodes:
            if tree.is_leaf(node):
                character_mapping[node] = [1]
            else:
                character_mapping[node] = [0]

        tree.initialize_all_character_states(character_mapping)

        self.assertCountEqual(tree.get_character_states("node0"), [0])
        self.assertCountEqual(tree.get_character_states("node5"), [1])

    def test_clear_cache(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        self.assertEqual(tree.root, "node0")

        new_network = nx.DiGraph()
        new_network.add_edges_from([("a", "b"), ("b", "c")])
        tree.populate_tree(new_network)
        self.assertFalse(tree.root == "node0")

        self.assertEqual(tree.root, "a")
        self.assertCountEqual(tree.leaves, ["c"])

    def test_check_internal_node(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        self.assertTrue(tree.is_internal_node("node10"))
        self.assertTrue(tree.is_internal_node("node0"))
        self.assertFalse(tree.is_internal_node("node5"))

    def test_uninitialized_tree_raises_error(self):
        r"""
        Methods of the CassiopeiaTree that operate on the tree (as opposed to,
        say, just the character matrix) require that the tree has been
        initialized, and should raise an error otherwise. Here we make sure
        that one such method raises the error. It is a very minimal test.
        """
        tree = cas.data.CassiopeiaTree(character_matrix=self.character_matrix)
        with self.assertRaises(CassiopeiaTreeError):
            tree.root

    def test_cache_internals_not_exposed(self):
        r"""
        The CassiopeiaTree should not expose the lists in the cache. It should
        only return copies.
        """
        tree = cas.data.CassiopeiaTree(tree=self.test_network)
        for method in [
            CassiopeiaTree.leaves,
            CassiopeiaTree.internal_nodes,
            CassiopeiaTree.nodes,
            CassiopeiaTree.edges,
        ]:
            res = method.fget(tree)  # Should return a COPY of the list
            res.clear()  # Thus, this should NOT clear the cache's internal list
            assert len(method.fget(tree)) > 0

    def test_tree_topology_not_exposed(self):
        r"""
        When the CassiopeiaTree is created with a networkx.DiGraph object, a
        COPY of the networkx.DiGraph object should be stored internally, to
        avoid aliasing issues.
        """
        tree1 = cas.data.CassiopeiaTree(tree=self.test_network)
        tree2 = cas.data.CassiopeiaTree(tree=self.test_network)
        a_leaf = tree1.leaves[0]
        tree1.set_time(a_leaf, 1234)  # Should NOT modify tree2!
        assert tree2.get_time(a_leaf) != 1234

    def test_set_dissimilarity_map(self):

        dissimilarity_map = pd.DataFrame.from_dict(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [0, 0, 2, 3, 4],
                "c": [0, 0, 0, 4, 4],
                "d": [0, 0, 0, 0, 4],
                "e": [0, 0, 0, 0, 0],
            },
            orient="index",
        )

        tree = cas.data.CassiopeiaTree(dissimilarity_map=dissimilarity_map)

        observed_dissimilarity_map = tree.get_dissimilarity_map()
        pd.testing.assert_frame_equal(
            observed_dissimilarity_map, dissimilarity_map
        )

        def delta_fn(
            x: np.array,
            y: np.array,
            missing_state: int,
            priors: Optional[Dict[int, Dict[int, float]]],
        ):
            d = 0
            for i in range(len(x)):
                if x[i] != y[i]:
                    d += 1
            return d

        tree = cas.data.CassiopeiaTree(self.character_matrix)
        tree.compute_dissimilarity_map(delta_fn)
        observed_dissimilarity_map = tree.get_dissimilarity_map()

        expected_dissimilarity_map = pd.DataFrame.from_dict(
            {
                "node3": [0, 1, 2, 3, 4, 5, 6, 7, 1, 2],
                "node7": [1, 0, 1, 2, 3, 4, 5, 6, 2, 2],
                "node9": [2, 1, 0, 1, 2, 3, 4, 5, 3, 3],
                "node11": [3, 2, 1, 0, 1, 2, 3, 4, 4, 4],
                "node13": [4, 3, 2, 1, 0, 1, 2, 3, 5, 5],
                "node15": [5, 4, 3, 2, 1, 0, 1, 2, 6, 6],
                "node17": [6, 5, 4, 3, 2, 1, 0, 1, 7, 7],
                "node18": [7, 6, 5, 4, 3, 2, 1, 0, 8, 8],
                "node5": [1, 2, 3, 4, 5, 6, 7, 8, 0, 1],
                "node6": [2, 2, 3, 4, 5, 6, 7, 8, 1, 0],
            },
            orient="index",
            columns=[
                "node3",
                "node7",
                "node9",
                "node11",
                "node13",
                "node15",
                "node17",
                "node18",
                "node5",
                "node6",
            ],
            dtype=np.float64,
        )

        pd.testing.assert_frame_equal(
            observed_dissimilarity_map, expected_dissimilarity_map
        )

        tree = cas.data.CassiopeiaTree(self.character_matrix)
        self.assertWarns(
            CassiopeiaTreeWarning, tree.set_dissimilarity_map, dissimilarity_map
        )

    def test_remove_and_prune_lineage_all(self):
        """Tests the case where all lineages are removed and pruned."""
        cas_tree = cas.data.CassiopeiaTree(
            tree=self.simple_complete_binary_tree
        )
        for i in cas_tree.leaves:
            cas_tree.remove_and_prune_lineage(i)

        self.assertEqual(cas_tree.nodes, ["node0"])
        self.assertEqual(cas_tree.edges, [])

    def test_remove_and_prune_lineage_some(self):
        """Tests a case where some lineages are removed"""
        cas_tree = cas.data.CassiopeiaTree(
            tree=self.simple_complete_binary_tree
        )
        cas_tree.remove_and_prune_lineage("node11")
        cas_tree.remove_and_prune_lineage("node13")
        cas_tree.remove_and_prune_lineage("node14")

        expected_edges = [
            ("node0", "node1"),
            ("node0", "node2"),
            ("node1", "node3"),
            ("node1", "node4"),
            ("node2", "node5"),
            ("node3", "node7"),
            ("node3", "node8"),
            ("node4", "node9"),
            ("node4", "node10"),
            ("node5", "node12"),
        ]
        self.assertEqual(cas_tree.edges, expected_edges)

    def test_remove_and_prune_lineage_one_side(self):
        """Tests a case where the entire one side of a tree is removed."""
        cas_tree = cas.data.CassiopeiaTree(
            tree=self.simple_complete_binary_tree
        )
        for i in range(7, 11):
            cas_tree.remove_and_prune_lineage("node" + str(i))

        expected_edges = [
            ("node0", "node2"),
            ("node2", "node5"),
            ("node2", "node6"),
            ("node5", "node11"),
            ("node5", "node12"),
            ("node6", "node13"),
            ("node6", "node14"),
        ]
        self.assertEqual(cas_tree.edges, expected_edges)

    def test_collapse_unifurcations_source(self):
        """Tests a case where a non-root source is provided."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(6)))
        tree.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
        str_names = dict(
            zip(list(tree.nodes), ["node" + str(i) for i in tree.nodes])
        )
        tree = nx.relabel_nodes(tree, str_names)
        cas_tree = cas.data.CassiopeiaTree(tree=tree)
        for u, v in cas_tree.edges:
            cas_tree.set_branch_length(u, v, 1.5)

        cas_tree.collapse_unifurcations(source="node1")

        expected_edges = {
            ("node0", "node1"): 1.5,
            ("node1", "node4"): 3.0,
            ("node1", "node5"): 4.5,
        }
        for u, v in cas_tree.edges:
            self.assertEqual(
                cas_tree.get_branch_length(u, v), expected_edges[(u, v)]
            )

    def test_collapse_unifurcations(self):
        """Tests a general case with unifurcations throughout the tree."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(10)))
        tree.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (2, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (2, 9),
            ]
        )
        str_names = dict(
            zip(list(tree.nodes), ["node" + str(i) for i in tree.nodes])
        )
        tree = nx.relabel_nodes(tree, str_names)
        cas_tree = cas.data.CassiopeiaTree(tree=tree)
        for u, v in cas_tree.edges:
            cas_tree.set_branch_length(u, v, 1.5)

        cas_tree.collapse_unifurcations()
        expected_edges = {
            ("node0", "node1"): 1.5,
            ("node0", "node2"): 1.5,
            ("node2", "node9"): 1.5,
            ("node2", "node4"): 3.0,
            ("node2", "node6"): 3.0,
            ("node6", "node7"): 1.5,
            ("node6", "node8"): 1.5,
        }
        for u, v in cas_tree.edges:
            self.assertEqual(
                cas_tree.get_branch_length(u, v), expected_edges[(u, v)]
            )

    def test_collapse_unifurcations_long_root_unifurcation(self):
        """Tests a case where there is a long chain at the root."""
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(15)))
        tree.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 6),
                (6, 7),
                (6, 8),
                (5, 9),
                (5, 10),
                (5, 11),
                (10, 12),
                (12, 13),
                (13, 14),
            ]
        )
        str_names = dict(
            zip(list(tree.nodes), ["node" + str(i) for i in tree.nodes])
        )
        tree = nx.relabel_nodes(tree, str_names)
        cas_tree = cas.data.CassiopeiaTree(tree=tree)
        for u, v in cas_tree.edges:
            cas_tree.set_branch_length(u, v, 1.5)

        cas_tree.collapse_unifurcations()

        expected_edges = {
            ("node0", "node5"): 6.0,
            ("node0", "node6"): 7.5,
            ("node5", "node9"): 1.5,
            ("node5", "node11"): 1.5,
            ("node5", "node14"): 6.0,
            ("node6", "node7"): 1.5,
            ("node6", "node8"): 1.5,
        }
        for u, v in cas_tree.edges:
            self.assertEqual(
                cas_tree.get_branch_length(u, v), expected_edges[(u, v)]
            )

    def test_set_and_add_attribute(self):

        tree = cas.data.CassiopeiaTree(
            character_matrix=self.character_matrix, tree=self.test_network
        )

        tree.set_attribute("node3", "test_attribute", 5)
        tree.set_attribute("node5", "test_attribute", 10)

        self.assertEqual(5, tree.get_attribute("node3", "test_attribute"))
        self.assertEqual(10, tree.get_attribute("node5", "test_attribute"))

        self.assertRaises(CassiopeiaTreeError, tree.get_attribute, "node10", "test_attribute")

    def test_filter_nodes(self):

        tree = cas.data.CassiopeiaTree(tree=self.test_network)

        for n in tree.depth_first_traverse_nodes(postorder=False):
            if tree.is_root(n):
                tree.set_attribute(n, "depth", 0)
                continue
            tree.set_attribute(n, "depth", tree.get_attribute(tree.parent(n), "depth")+1)
        
        nodes = tree.filter_nodes(lambda x: tree.get_attribute(x, "depth") == 2)
        expected_nodes = ["node5", "node6", "node3", "node4"]

        self.assertCountEqual(nodes, expected_nodes)



if __name__ == "__main__":
    unittest.main()
