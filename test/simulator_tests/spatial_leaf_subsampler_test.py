import unittest

import networkx as nx
import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import LeafSubsamplerError
from cassiopeia.simulator.SpatialLeafSubsampler import SpatialLeafSubsampler


class SpatialLeafSubsamplerTest(unittest.TestCase):
    def setUp(self):
        # create tree
        balanced_tree = nx.balanced_tree(2, 2, create_using=nx.DiGraph)
        self.tree = CassiopeiaTree(tree=balanced_tree)

        # add spatial information
        self.tree.set_attribute("3","spatial", (.25,.25))
        self.tree.set_attribute("4","spatial", (.75,.25))
        self.tree.set_attribute("5","spatial", (.25,.75))
        self.tree.set_attribute("6","spatial", (.75,.75))

        #add additional attributes
        self.tree.set_attribute("3","test","test")

    def test_bad_parameters(self):
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler()

    def test_bad_number_of_samples(self):
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler(number_of_leaves=0)
            spatial_sampler.subsample_leaves(self.tree)
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler(number_of_leaves=1000)
            spatial_sampler.subsample_leaves(self.tree)

    def test_bad_bounding_box(self):
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1)])
            spatial_sampler.subsample_leaves(self.tree)
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1),(0,1),(0,1)])
            spatial_sampler.subsample_leaves(self.tree)

    def test_bad_attrbute_key(self):
        with self.assertRaises(LeafSubsamplerError):
            spatial_sampler = SpatialLeafSubsampler(attribute_key="bad_key")
            spatial_sampler.subsample_leaves(self.tree)

    def test_subsample_bounding_box(self):
        # all leaves in bounding box
        spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1),(0,1)])
        subsampled_tree = spatial_sampler.subsample_leaves(self.tree)
        self.assertEqual(len(subsampled_tree.leaves), 4)
        # edges (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)
        expected_edges = [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6")
        ]
        self.assertEqual(set(subsampled_tree.edges), set(expected_edges))
        # 2 leaves in bounding box
        spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,0.5),(0,1)])
        subsampled_tree = spatial_sampler.subsample_leaves(self.tree)
        self.assertEqual(len(subsampled_tree.leaves), 2)
        expected_edges = [
            ("0","3"),
            ("0","6")
        ]

    def test_subsample_number_of_leaves(self):
        # all leaves in bounding box
        spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1),(0,1)],number_of_leaves=2)
        subsampled_tree = spatial_sampler.subsample_leaves(self.tree)
        self.assertEqual(len(subsampled_tree.leaves),2)
        # 2 leaves in bounding box
        spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1),(0,1)],number_of_leaves=1)
        subsampled_tree = spatial_sampler.subsample_leaves(self.tree)
        self.assertEqual(len(subsampled_tree.leaves), 1)

    def test_attribute_copy(self):
        spatial_sampler = SpatialLeafSubsampler(bounding_box=[(0,1),(0,1)],number_of_leaves=4)
        subsampled_tree = spatial_sampler.subsample_leaves(self.tree)
        self.assertEqual(subsampled_tree.get_attribute("3","test"),"test")


if __name__ == "__main__":
    unittest.main()
