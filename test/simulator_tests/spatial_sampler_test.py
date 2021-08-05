import unittest

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import mapping, Point, Polygon

import cassiopeia as cas
from cassiopeia.mixins import LeafSubsamplerError
from cassiopeia.simulator import SpatialSampler


class TestSpatialSampler(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        topology = nx.DiGraph()
        topology.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "6"),
                ("3", "7"),
                ("3", "8"),
                ("4", "9"),
                ("4", "10"),
                ("5", "11"),
                ("5", "12"),
                ("6", "13"),
                ("6", "14"),
            ]
        )

        tree = cas.data.CassiopeiaTree(tree=topology)
        tree.set_times(
            {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 2,
                "4": 2,
                "5": 2,
                "6": 2,
                "7": 3,
                "8": 3,
                "9": 3,
                "10": 3,
                "11": 3,
                "12": 3,
                "13": 3,
                "14": 3,
            }
        )
        spatial = {
            "7": (0.0, 0.0),
            "8": (0.5, 0.5),
            "9": (0.5, 1.5),
            "10": (0.5, 1.5),
            "11": (1.5, 0.5),
            "12": (1.5, 0.5),
            "13": (1.5, 1.5),
            "14": (1.5, 1.5),
        }
        for leaf, location in spatial.items():
            tree.set_attribute(leaf, "spatial", location)
        tree.character_matrix = pd.DataFrame.from_dict(
            {
                "7": [1, 0, 0, 0, 0, 0, 0, 0],
                "8": [1, 1, 0, 0, 0, 0, 0, 0],
                "9": [1, 1, 1, 0, 0, 0, 0, 0],
                "10": [1, 1, 1, 1, 0, 0, 0, 0],
                "11": [1, 1, 1, 1, 1, 0, 0, 0],
                "12": [1, 1, 1, 1, 1, 1, 0, 0],
                "13": [1, 1, 1, 1, 1, 1, 1, 0],
                "14": [1, 1, 1, 1, 1, 1, 1, 1],
            },
            orient="index",
        )
        tree.set_character_states_at_leaves(tree.character_matrix)
        self.basic_tree = tree

    def test_create_spots(self):
        sampler = SpatialSampler(spot_size=1)
        spots = sampler.create_spots((0, 2), (0, 2))
        expected_mappings = [
            {
                "type": "Polygon",
                "coordinates": (
                    (
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0),
                        (0.0, 0.0),
                    ),
                ),
            },
            {
                "type": "Polygon",
                "coordinates": (
                    (
                        (0.0, 1.0),
                        (1.0, 1.0),
                        (1.0, 2.0),
                        (0.0, 2.0),
                        (0.0, 1.0),
                    ),
                ),
            },
            {
                "type": "Polygon",
                "coordinates": (
                    (
                        (1.0, 0.0),
                        (2.0, 0.0),
                        (2.0, 1.0),
                        (1.0, 1.0),
                        (1.0, 0.0),
                    ),
                ),
            },
            {
                "type": "Polygon",
                "coordinates": (
                    (
                        (1.0, 1.0),
                        (2.0, 1.0),
                        (2.0, 2.0),
                        (1.0, 2.0),
                        (1.0, 1.0),
                    ),
                ),
            },
        ]
        self.assertEqual([mapping(spot) for spot in spots], expected_mappings)

        with self.assertRaises(LeafSubsamplerError):
            sampler.create_spots((0, 0), (0, 2))
        with self.assertRaises(LeafSubsamplerError):
            sampler.create_spots((0, 2), (0, 0))

    def test_create_cell(self):
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        self.assertEqual(
            mapping(sampler.create_cell(0, 0)),
            {"type": "Point", "coordinates": (0.0, 0.0)},
        )

        sampler = SpatialSampler(spot_size=1, cell_size=1.0)
        self.assertEqual(
            mapping(sampler.create_cell(0, 0)),
            {
                "type": "Polygon",
                "coordinates": (
                    (
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0),
                        (0.0, 0.0),
                    ),
                ),
            },
        )

    def test_capture_cells(self):
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        spot = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        cells = {
            "1": sampler.create_cell(0, 0),
            "2": sampler.create_cell(0.5, 0.5),
            "3": sampler.create_cell(2, 2),
        }
        captured = sampler.capture_cells(spot, cells)
        expected_captured = {"1": 1.0, "2": 1.0}
        self.assertEqual(captured, expected_captured)

        sampler = SpatialSampler(spot_size=1, cell_size=1.0)
        spot = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        cells = {
            "1": sampler.create_cell(0, 0),
            "2": sampler.create_cell(0.5, 0.5),
            "3": sampler.create_cell(2, 2),
        }
        captured = sampler.capture_cells(spot, cells)
        expected_captured = {"1": 1.0, "2": 0.25}
        self.assertEqual(captured, expected_captured)

    def test_sample_states(self):
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        sampled_states = sampler.sample_states(
            self.basic_tree, {"7": 1.0, "8": 0.5}
        )
        expected_sampled_states = [1, (0, 1), 0, 0, 0, 0, (0, 0), (0, 0)]
        self.assertEqual(sampled_states, expected_sampled_states)

    def test_subsample_leaves(self):
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        sampled_tree = sampler.subsample_leaves(
            self.basic_tree, collapse_duplicates=False
        )
        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "s0": [
                    (1, 1),
                    (0, 1),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ],
                "s1": [
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (0, 1),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ],
                "s2": [
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (0, 1),
                    (0, 0),
                    (0, 0),
                ],
                "s3": [
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (0, 1),
                ],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(
            sampled_tree.character_matrix, expected_character_matrix
        )

        expected_cell_meta = pd.DataFrame.from_dict(
            {
                "s0": [0.5, 0.5],
                "s1": [0.5, 1.5],
                "s2": [1.5, 0.5],
                "s3": [1.5, 1.5],
            },
            orient="index",
            columns=["spatial_0", "spatial_1"],
        )
        pd.testing.assert_frame_equal(
            sampled_tree.cell_meta, expected_cell_meta
        )

    def test_subsample_leaves_raises_on_ambiguous(self):
        self.basic_tree.set_character_states("7", [(1, 1), 0, 0, 0, 0, 0, 0, 0])
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        with self.assertRaises(LeafSubsamplerError):
            sampler.subsample_leaves(self.basic_tree)

    def test_subsample_leaves_raises_on_non2D(self):
        self.basic_tree.set_attribute("7", "spatial", (1, 1, 1))
        sampler = SpatialSampler(spot_size=1, cell_size=0.0)
        with self.assertRaises(LeafSubsamplerError):
            sampler.subsample_leaves(self.basic_tree)
