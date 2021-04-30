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
            uniform_sampler = UniformLeafSubsampler(ratio = 0.5, number_of_leaves = 400)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler()

    def test_bad_number_of_samples(self):
        balanced_tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
        tree = CassiopeiaTree(tree=balanced_tree)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler(number_of_leaves = 0)
            uniform_sampler.subsample_leaves(tree)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = UniformLeafSubsampler(ratio = 0.0001)
            uniform_sampler.subsample_leaves(tree)

    def test_subsample_balanced_tree(self):
        balanced_tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
        balanced_tree = nx.relabel_nodes(balanced_tree, dict([(i, "node" + str(i)) for i in balanced_tree.nodes]))
        balanced_tree.add_node('node15')
        balanced_tree.add_edge('node15', 'node0')
        tree = CassiopeiaTree(tree=balanced_tree)

        np.random.seed(10)
        uni = UniformLeafSubsampler(number_of_leaves=3)
        res = uni.subsample_leaves(tree=tree, collapse_source = "node0")
        expected_edges = [
            ("node15", "node0"),
            ("node0", "node8"),
            ("node0", "node5"),
            ("node5", "node11"),
            ("node5", "node12"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

        np.random.seed(10)
        uni = UniformLeafSubsampler(ratio=0.65)
        res = uni.subsample_leaves(tree=tree, collapse_source = "node0")
        expected_edges = [
            ("node15", "node0"),
            ("node0", "node2"),
            ("node0", "node3"),
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
        custom_tree.add_edges_from([
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
        ])
        tree = CassiopeiaTree(tree=custom_tree)

        np.random.seed(10)
        uni = UniformLeafSubsampler(ratio = 0.5)
        res = uni.subsample_leaves(tree=tree, collapse_source = "node0")

        expected_edges = [
            ("node16", "node0"),
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node11"),
            ("node11", "node12"),
            ("node11", "node13"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

        np.random.seed(11)
        uni = UniformLeafSubsampler(number_of_leaves = 6)
        res = uni.subsample_leaves(tree=tree, collapse_source = "node0")

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

