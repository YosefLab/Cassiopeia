"""
Tests the functionality of cassiopeia.simulator.SpatialLeafSubsampler.
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import LeafSubsamplerError, LeafSubsamplerWarning
from cassiopeia.simulator.SpatialLeafSubsampler import SpatialLeafSubsampler


class SpatialLeafSubsamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        #tree
        balanced_tree = nx.balanced_tree(2, 2, create_using=nx.DiGraph)
        balanced_tree = nx.relabel_nodes(
            balanced_tree,
            dict([(i, "node" + str(i)) for i in balanced_tree.nodes]),
        )
        tree = CassiopeiaTree(tree=balanced_tree)
        self.tree = tree.copy()
        tree.set_attribute("node3", "spatial", (2,5,5))
        tree.set_attribute("node4", "spatial", (2.5,5,5))
        tree.set_attribute("node5", "spatial", (3,5,5))
        tree.set_attribute("node6", "spatial", (8,5,5))
        #3D tree
        self.tree_3d = tree.copy()
        #2D tree
        for node in tree.leaves:
            tree.set_attribute(node, "spatial",
                tree.get_attribute(node, "spatial")[:2])
        self.tree_2d = tree
        #3D regions
        self.bounding_box_3d = [(0,5),(0,10),(0,10)]
        self.space_3d = np.zeros((10,10,10))
        self.space_3d[0:5,:,:] = 1
        self.space_3d = self.space_3d.astype(bool)
        #2D regions
        self.bounding_box_2d = [(0,5),(0,10)]
        self.space_2d = np.zeros((10,10))
        self.space_2d[0:5,:] = 1
        self.space_2d = self.space_2d.astype(bool)

    def test_bad_init_parameters(self):
        # both ratio and number_of_leaves provided
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(
                ratio=0.5, number_of_leaves=1, space=self.space_2d)
        # neither space nor bounding_box provided
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler()
        # both space and bounding_box provided
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(space=self.space_2d, 
                bounding_box=self.bounding_box_2d)
        # negative number of leaves
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(number_of_leaves=-1, 
                space=self.space_2d)
        # ratio > 1
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(ratio=2, space=self.space_2d)
        # merge cells without space
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(bounding_box=self.bounding_box_3d,
                        merge_cells=True)
        
    def test_bad_subsample_parameters(self):
        # tree without spatial attributes
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(space=self.space_2d)
            sampler.subsample_leaves(self.tree)
        # incompatible space dimensions
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(space=self.space_3d)
            sampler.subsample_leaves(self.tree_2d)
        # incompatible bounding box dimensions
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(bounding_box=self.bounding_box_3d)
            sampler.subsample_leaves(self.tree_2d)
        # leaf coordinates outside space
        with self.assertRaises(LeafSubsamplerError):
            tree = self.tree_2d.copy()
            tree.set_attribute("node3", "spatial", (100,10))
            sampler = SpatialLeafSubsampler(space=self.space_2d)
            sampler.subsample_leaves(tree)
        # no leaves in region
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(bounding_box=[(0,0),(0,0)])
            sampler.subsample_leaves(self.tree_2d)
        # test number_of_leaves too large
        with self.assertRaises(LeafSubsamplerError):
            sampler = SpatialLeafSubsampler(number_of_leaves=100, 
                space=self.space_3d)
            sampler.subsample_leaves(self.tree_3d)

    def test_bad_scale(self):
        #tree
        balanced_tree = nx.balanced_tree(2, 1, create_using=nx.DiGraph)
        balanced_tree = nx.relabel_nodes(
            balanced_tree,
            dict([(i, "node" + str(i)) for i in balanced_tree.nodes]),
        )
        tree = CassiopeiaTree(tree=balanced_tree)
        tree.set_attribute("node1", "spatial", (.001,0))
        tree.set_attribute("node2", "spatial", (0,.001))
        sampler = SpatialLeafSubsampler(space=self.space_2d)
        with self.assertWarns(LeafSubsamplerWarning):
            res = sampler.subsample_leaves(tree)

    def test_subsample_3d_tree(self):
        # test subsampling using bounding box
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(bounding_box=self.bounding_box_3d)
        res = sampler.subsample_leaves(self.tree_3d)
        expected_edges = [
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]
        self.assertEqual(set(res.edges), set(expected_edges))
        # test subsampling using space
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(space=self.space_3d)
        res = sampler.subsample_leaves(self.tree_3d)
        expected_edges = [
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

    def test_subsample_2d_tree(self):
        # test subsampling using bounding box
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(bounding_box=self.bounding_box_2d)
        res = sampler.subsample_leaves(self.tree_2d)
        expected_edges = [
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]
        self.assertEqual(set(res.edges), set(expected_edges))
        # test subsampling using space
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(space=self.space_2d)
        res = sampler.subsample_leaves(self.tree_2d)
        expected_edges = [
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]

    def test_keep_singular_root_edge(self):
        #tree
        balanced_tree = nx.balanced_tree(2, 2, create_using=nx.DiGraph)
        balanced_tree = nx.relabel_nodes(
            balanced_tree,
            dict([(i, "node" + str(i)) for i in balanced_tree.nodes]),
        )
        balanced_tree.add_node("root")
        balanced_tree.add_edge("root", "node0")
        tree = CassiopeiaTree(tree=balanced_tree)
        tree.set_attribute("node3", "spatial", (2,5,5))
        tree.set_attribute("node4", "spatial", (2.5,5,5))
        tree.set_attribute("node5", "spatial", (3,5,5))
        tree.set_attribute("node6", "spatial", (8,5,5))
        sampler = SpatialLeafSubsampler(bounding_box=self.bounding_box_3d)
        # test keep_singular_root_edge=True
        res = sampler.subsample_leaves(tree,keep_singular_root_edge=True)
        expected_edges = [
            ("root", "node0"),
            ("node0", "node1"),
            ("node0", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]
        self.assertEqual(set(res.edges), set(expected_edges))
        # test keep_singular_root_edge=False
        res = sampler.subsample_leaves(tree,keep_singular_root_edge=False)
        expected_edges = [
            ("root", "node1"),
            ("root", "node5"),
            ("node1", "node3"),
            ("node1", "node4")
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

    def test_subsample_with_ratio(self):
        # test subsampling using ratio
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(ratio=0.7, space=self.space_3d)
        res = sampler.subsample_leaves(self.tree_3d)
        expected_edges = [
            ("node0", "node3"),
            ("node0", "node5"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

    def test_subsample_with_number(self):
        # test subsampling using number_of_leaves
        np.random.seed(10)
        sampler = SpatialLeafSubsampler(number_of_leaves=2, space=self.space_3d)
        res = sampler.subsample_leaves(self.tree_3d)
        expected_edges = [
            ("node0", "node3"),
            ("node0", "node5"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))

    def test_merge_cells(self):
        # no character matrix
        sampler = SpatialLeafSubsampler(space=self.space_3d,merge_cells=True)
        res = sampler.subsample_leaves(self.tree_3d)
        expected_edges = [
            ("node0", "node3-node4"),
            ("node0", "node5"),
        ]
        self.assertEqual(set(res.edges), set(expected_edges))
        self.assertEqual(res.get_attribute("node3-node4","spatial"),(2,5,5))
        self.assertEqual(res.get_attribute("node5","spatial"),(3,5,5))
        # with character matrix
        tree = self.tree_3d.copy()
        character_matrix = pd.DataFrame(
            data = np.array([[0,1,],[1,1],[1,1],[2,2]]),
            index = ["node3","node4","node5","node6"],
        )
        tree.character_matrix = character_matrix
        tree.set_character_states_at_leaves(character_matrix)
        res = sampler.subsample_leaves(tree)
        self.assertEqual(res.get_attribute("node3-node4","character_states"),
                         [(0, 1), (1,)])
        self.assertEqual(res.get_attribute("node5","character_states"),
                         [(1),(1)])
        # without collapsing duplicates
        res = sampler.subsample_leaves(tree,collapse_duplicates=False)
        self.assertEqual(res.get_attribute("node3-node4","character_states"),
                            [(0, 1), (1, 1)])
        self.assertEqual(res.get_attribute("node5","character_states"),
                            [(1), (1)])

if __name__ == "__main__":
    unittest.main()
