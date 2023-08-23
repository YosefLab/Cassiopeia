import unittest
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.plotting import local_3d


class TestLocal3DPlotting(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        simulator = cas.sim.CompleteBinarySimulator(num_cells=8)
        self.tree = simulator.simulate_tree()

        spatial_simulator = cas.sim.ClonalSpatialDataSimulator((10, 10))
        spatial_simulator.overlay_data(self.tree)

        self.labels = local_3d.labels_from_coordinates(self.tree)

    def test_interpolate_branch(self):
        parent = (0, 0, 0)
        child = (1, 1, 1)
        np.testing.assert_array_equal(
            [[0, 0, 0], [1, 1, 0], [1, 1, 1]],
            local_3d.interpolate_branch(parent, child),
        )

    def test_polyline_from_points(self):
        points = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
            ]
        )
        poly = local_3d.polyline_from_points(points)
        np.testing.assert_array_equal(points, poly.points)

    def test_average_mixing(self):
        c1 = (0, 0, 0)
        c2 = (0.1, 0.2, 0.3)
        c3 = (0.5, 0.7, 0.0)
        np.testing.assert_allclose(
            (0.2, 0.3, 0.1), local_3d.average_mixing(c1, c2, c3)
        )

    def test_highlight(self):
        c = (0.8, 0.2, 0.0)
        np.testing.assert_allclose((1.0, 0.25, 0.0), local_3d.highlight(c))

    def test_lowlight(self):
        c = (0.8, 0.2, 0.0)
        np.testing.assert_allclose((0.3, 0.075, 0.0), local_3d.lowlight(c))

    def test_labels_from_coordinates(self):
        for leaf in self.tree.leaves:
            x, y = self.tree.get_attribute(leaf, "spatial")
            self.assertEqual(
                self.labels[int(x), int(y)],
                self.tree.cell_meta["spatial_label"][leaf],
            )

    def test_Tree3D(self):
        # There isn't a good way to test this, other than making sure there
        # are no errors on initialization.
        tree3d = local_3d.Tree3D(self.tree, self.labels)
        tree3d.plot(show=False)

if __name__ == "__main__":
    unittest.main()
