import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree, CassiopeiaTreeError
from cassiopeia.simulator.LeafSubsampler import LeafSubsamplerError
from cassiopeia.simulator.SupercellularSampler import SupercellularSampler

import cassiopeia.data.utilities as utilities

class SupercellularSamplerTest(unittest.TestCase):
    def setUp(self):
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

    def test_bad_parameters(self):
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = SupercellularSampler(ratio=0.5, number_of_merges=400)
        with self.assertRaises(LeafSubsamplerError):
            uniform_sampler = SupercellularSampler()

    def test_bad_number_of_samples(self):
        tree = CassiopeiaTree(tree=self.test_network, character_matrix=self.character_matrix)
        tree_no_character_matrix = CassiopeiaTree(tree=self.test_network)
        with self.assertRaises(LeafSubsamplerError):
            sampler = SupercellularSampler(number_of_merges=10)
            sampler.subsample_leaves(tree)
        with self.assertRaises(LeafSubsamplerError):
            sampler = SupercellularSampler(number_of_merges=0)
            sampler.subsample_leaves(tree)
        with self.assertRaises(CassiopeiaTreeError):
            sampler = SupercellularSampler(number_of_merges=2)
            sampler.subsample_leaves(tree_no_character_matrix)

    def test_subsample_balanced_tree(self):
        tree = CassiopeiaTree(tree=self.test_network, character_matrix=self.character_matrix)

        np.random.seed(10)
        sampler = SupercellularSampler(number_of_merges=2)
        res = sampler.subsample_leaves(tree=tree)
        cm = res.get_current_character_matrix()
        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "node7": [1, 1, 0, 0, 0, 0, 0, 0],
                "node9": [1, 1, 1, 0, 0, 0, 0, 0],
                "node11": [1, 1, 1, 1, 0, 0, 0, 0],
                "node13": [1, 1, 1, 1, 1, 0, 0, 0],
                "node17": [1, 1, 1, 1, 1, 1, 1, 0],
                "node6": [2, 2, 0, 0, 0, 0, 0, 0],
                "node18-node15": [(1,), (1,), (1,), (1,), (1,), (1,), (0, 1), (0, 1)],
                "node3-node5": [(1, 2), (0,), (0,), (0,), (0,), (0,), (0,), (0,)],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(expected_character_matrix, cm)
        expected_edges = [
            ("node0", "node3-node5"),
            ("node0", "node4"),
            ("node0", "node6"),
            ("node4", "node7"),
            ("node4", "node8"),
            ("node8", "node10"),
            ("node8", "node9"),
            ("node10", "node11"),
            ("node10", "node12"),
            ("node12", "node13"),
            ("node14", "node17"),
            ("node12", "node14"),
            ("node14", "node18-node15"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

        np.random.seed(10)
        res = sampler.subsample_leaves(tree=tree, collapse_duplicates=False)
        cm = res.get_current_character_matrix()
        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "node7": [1, 1, 0, 0, 0, 0, 0, 0],
                "node9": [1, 1, 1, 0, 0, 0, 0, 0],
                "node11": [1, 1, 1, 1, 0, 0, 0, 0],
                "node13": [1, 1, 1, 1, 1, 0, 0, 0],
                "node17": [1, 1, 1, 1, 1, 1, 1, 0],
                "node6": [2, 2, 0, 0, 0, 0, 0, 0],
                "node18-node15": [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 0), (1, 0)],
                "node3-node5": [(1, 2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(expected_character_matrix, cm)
        expected_edges = [
            ("node4", "node7"),
            ("node4", "node8"),
            ("node8", "node9"),
            ("node12", "node13"),
            ("node14", "node17"),
            ("node12", "node14"),
            ("node14", "node18-node15"),
            ('node0', 'node6'),
            ('node10', 'node12'),
            ('node10', 'node11'),
            ('node8', 'node10'),
            ('node0', 'node4'),
            ('node0', 'node3-node5'),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))


if __name__ == "__main__":
    unittest.main()
