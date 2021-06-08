import unittest

import networkx as nx
import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import LeafSubsamplerError
from cassiopeia.simulator.UniformLeafSubsampler import UniformLeafSubsampler

import cassiopeia.data.utilities as utilities


class UniformLeafSubsamplerTest(unittest.TestCase):
    def test_bad_parameters(self):
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler(
                ratio=0.5, number_of_leaves=400
            )
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler()

    def test_bad_number_of_samples(self):
        balanced_tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
        tree = CassiopeiaTree(tree=balanced_tree)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler(number_of_leaves=0)
            uniform_sampler.subsample_leaves(tree)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler(ratio=0.0001)
            uniform_sampler.subsample_leaves(tree)

    def test_subsample_balanced_tree(self):
        balanced_tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
        balanced_tree = nx.relabel_nodes(
            balanced_tree,
            dict([(i, "node" + str(i)) for i in balanced_tree.nodes]),
        )
        balanced_tree.add_node("node15")
        balanced_tree.add_edge("node15", "node0")
        tree = CassiopeiaTree(tree=balanced_tree)

        np.random.seed(10)
        uni = UniformLeafSubsampler(number_of_leaves=3)
        res = uni.subsample_leaves(tree=tree)
        expected_edges = [
            ("node15", "node8"),
            ("node15", "node5"),
            ("node5", "node11"),
            ("node5", "node12"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

        np.random.seed(10)
        uni = UniformLeafSubsampler(ratio=0.65)
        res = uni.subsample_leaves(tree=tree)
        expected_edges = [
            ("node15", "node2"),
            ("node15", "node3"),
            ("node2", "node14"),
            ("node2", "node5"),
            ("node5", "node11"),
            ("node5", "node12"),
            ("node3", "node7"),
            ("node3", "node8"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

    def test_subsample_custom_tree(self):
        custom_tree = nx.DiGraph()
        custom_tree.add_nodes_from(["node" + str(i) for i in range(17)])
        custom_tree.add_edges_from(
            [
                ("node16", "node0"),
                ("node0", "node1"),
                ("node0", "node2"),
                ("node1", "node3"),
                ("node1", "node4"),
                ("node2", "node5"),
                ("node2", "node6"),
                ("node4", "node7"),
                ("node4", "node8"),
                ("node6", "node9"),
                ("node6", "node10"),
                ("node7", "node11"),
                ("node11", "node12"),
                ("node11", "node13"),
                ("node9", "node14"),
                ("node9", "node15"),
            ]
        )
        tree = CassiopeiaTree(tree=custom_tree)
        for u, v in tree.edges:
            tree.set_branch_length(u, v, 1.5)

        np.random.seed(10)
        uni = UniformLeafSubsampler(ratio=0.5)
        res = uni.subsample_leaves(tree=tree, collapse_source="node0")

        expected_edges = {
            ("node16", "node0"): 1.5,
            ("node0", "node1"): 1.5,
            ("node0", "node5"): 3.0,
            ("node1", "node3"): 1.5,
            ("node1", "node11"): 4.5,
            ("node11", "node12"): 1.5,
            ("node11", "node13"): 1.5,
        }
        self.assertEqual(set(res.edges), set(expected_edges.keys()))
        for u, v in res.edges:
            self.assertEqual(
                res.get_branch_length(u, v), expected_edges[(u, v)]
            )

        expected_times = {
            "node16": 0.0,
            "node0": 1.5,
            "node1": 3.0,
            "node5": 4.5,
            "node3": 4.5,
            "node11": 7.5,
            "node12": 9.0,
            "node13": 9.0,
        }
        for u in res.nodes:
            self.assertEqual(res.get_time(u), expected_times[u])

        np.random.seed(11)
        uni = UniformLeafSubsampler(number_of_leaves=6)
        res = uni.subsample_leaves(tree=tree, collapse_source="node0")

        expected_edges = [
            ("node16", "node0"),
            ("node0", "node1"),
            ("node0", "node2"),
            ("node1", "node3"),
            ("node1", "node11"),
            ("node11", "node12"),
            ("node11", "node13"),
            ("node2", "node5"),
            ("node2", "node6"),
            ("node6", "node10"),
            ("node6", "node15"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))


if __name__ == "__main__":
    unittest.main()
