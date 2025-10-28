import unittest

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cassiopeia as cas
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator import ClonalSpatialDataSimulator


class TestClonalSpatialDataSimulator(unittest.TestCase):
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
        self.basic_tree = tree

        self.cell_meta = pd.DataFrame(
            [
                ["a"],
                ["b"],
                ["c"],
                ["d"],
                ["e"],
                ["f"],
                ["g"],
                ["h"],
            ],
            columns=["existing_column"],
            index=["7", "8", "9", "10", "11", "12", "13", "14"],
        )
        self.tree_with_cell_meta = cas.data.CassiopeiaTree(tree=topology, cell_meta=self.cell_meta)

    @pytest.mark.spatial
    def test_init(self):
        with self.assertRaises(DataSimulatorError):
            ClonalSpatialDataSimulator()

        simulator = ClonalSpatialDataSimulator((3, 3))
        self.assertEqual(simulator.dim, 2)
        expected = np.zeros((3, 3), dtype=bool)
        expected[0, 1] = True
        expected[1] = True
        expected[2, 1] = True
        np.testing.assert_array_equal(expected, simulator.space)

        simulator = ClonalSpatialDataSimulator((10, 10, 10))
        self.assertEqual(simulator.dim, 3)
        np.testing.assert_array_equal(np.ones((10, 10, 10), dtype=bool), simulator.space)

    @pytest.mark.spatial
    def test_overlay_data(self):
        simulator = ClonalSpatialDataSimulator((100, 100))
        simulator.overlay_data(self.basic_tree)

        expected_coordinates = {
            "0": (49.313065, 50.858765),
            "1": (70.08444, 56.516514),
            "2": (28.541695, 45.20101),
            "3": (63.72982, 72.93363),
            "4": (76.43906, 40.099403),
            "5": (36.568497, 69.24371),
            "6": (20.514889, 21.158302),
            "7": (72.57829, 74.34833),
            "8": (54.88135, 71.518936),
            "9": (68.418655, 24.157845),
            "10": (84.45945, 56.04096),
            "11": (41.12374, 40.823444),
            "12": (32.013256, 97.66399),
            "13": (11.028472, 30.138132),
            "14": (30.001307, 12.178472),
        }
        for node in self.basic_tree.nodes:
            np.testing.assert_allclose(
                self.basic_tree.get_attribute(node, "spatial"),
                expected_coordinates[node],
            )
        expected_cell_meta = pd.DataFrame(
            data=[expected_coordinates[leaf] for leaf in self.basic_tree.leaves],
            index=self.basic_tree.leaves,
            columns=["spatial_0", "spatial_1"],
        )
        pd.testing.assert_frame_equal(self.basic_tree.cell_meta, expected_cell_meta)

    @pytest.mark.spatial
    def test_overlay_data_with_space(self):
        simulator = ClonalSpatialDataSimulator(space=np.ones((100, 100), dtype=bool))
        simulator.overlay_data(self.basic_tree)

        expected_coordinates = {
            "0": (36.012184, 44.996532),
            "1": (40.75788, 48.254433),
            "2": (31.266485, 41.73863),
            "3": (8.0896225, 49.16102),
            "4": (73.42613, 47.34785),
            "5": (26.970448, 56.9763),
            "6": (35.562523, 26.500957),
            "7": (2.3998058, 96.62743),
            "8": (13.779439, 1.6946061),
            "9": (96.97778, 70.33626),
            "10": (49.874485, 24.359444),
            "11": (39.2209, 63.14556),
            "12": (14.719994, 50.807037),
            "13": (41.12374, 40.823444),
            "14": (30.001307, 12.178472),
        }
        for node in self.basic_tree.nodes:
            np.testing.assert_allclose(
                self.basic_tree.get_attribute(node, "spatial"),
                expected_coordinates[node],
            )
        expected_cell_meta = pd.DataFrame(
            data=[expected_coordinates[leaf] for leaf in self.basic_tree.leaves],
            index=self.basic_tree.leaves,
            columns=["spatial_0", "spatial_1"],
        )
        pd.testing.assert_frame_equal(self.basic_tree.cell_meta, expected_cell_meta)

    @pytest.mark.spatial
    def test_overlay_data_with_existing_cell_meta(self):
        simulator = ClonalSpatialDataSimulator((100, 100))
        simulator.overlay_data(self.tree_with_cell_meta)

        expected_coordinates = {
            "0": (49.313065, 50.858765),
            "1": (70.08444, 56.516514),
            "2": (28.541695, 45.20101),
            "3": (63.72982, 72.93363),
            "4": (76.43906, 40.099403),
            "5": (36.568497, 69.24371),
            "6": (20.514889, 21.158302),
            "7": (72.57829, 74.34833),
            "8": (54.88135, 71.518936),
            "9": (68.418655, 24.157845),
            "10": (84.45945, 56.04096),
            "11": (41.12374, 40.823444),
            "12": (32.013256, 97.66399),
            "13": (11.028472, 30.138132),
            "14": (30.001307, 12.178472),
        }
        for node in self.tree_with_cell_meta.nodes:
            np.testing.assert_allclose(
                self.tree_with_cell_meta.get_attribute(node, "spatial"),
                expected_coordinates[node],
            )
        expected_spatial_cell_meta = pd.DataFrame(
            data=[expected_coordinates[leaf] for leaf in self.tree_with_cell_meta.leaves],
            index=self.tree_with_cell_meta.leaves,
            columns=["spatial_0", "spatial_1"],
        )
        expected_cell_meta = pd.concat((self.cell_meta, expected_spatial_cell_meta), axis=1)
        pd.testing.assert_frame_equal(self.tree_with_cell_meta.cell_meta, expected_cell_meta)
