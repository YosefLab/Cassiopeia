import unittest

import networkx as nx
import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import LeafSubsamplerError
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
        for node in ["node3", "node4","node5"]:
            tree.set_attribute(node, "spatial", (2,5,5))
        for node in ["node6"]:
            tree.set_attribute(node, "spatial", (8,5,5))
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
            sampler.subsample_leaves(self.tree_2d)

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

if __name__ == "__main__":
    unittest.main()