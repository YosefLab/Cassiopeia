import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.plotting import utilities


class TestPlottingUtilities(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(123412334)

        self.indel_priors = pd.DataFrame.from_dict(
            {"i": 0.8, "j": 0.1, "k": 0.01, "m": 0.5, "n": 0.5},
            orient="index",
            columns=["freq"],
        )

        tree = nx.DiGraph()
        tree.add_nodes_from(["B", "C", "D", "E", "F"])
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)
        self.basic_tree = cas.data.CassiopeiaTree(tree=tree)
        self.basic_tree.set_attribute("B", "other", 4)
        self.basic_tree.set_attribute("C", "other", 3)
        self.basic_tree.set_attribute("D", "other", 2)
        self.basic_tree.set_attribute("E", "other", 1)
        self.basic_tree.set_attribute("F", "other", 0)

    def test_degrees_to_radians(self):
        self.assertAlmostEqual(np.pi, utilities.degrees_to_radians(180))
        self.assertAlmostEqual(np.pi / 2, utilities.degrees_to_radians(90))

    def test_polar_to_cartesian(self):
        x, y = utilities.polar_to_cartesian(0, 1)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)

        x, y = utilities.polar_to_cartesian(90, 1)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)

    def test_generate_random_color(self):
        color_itervals = [0.5, 1, 0, 0.5, 0, 1]

        rgb = utilities.generate_random_color(
            color_itervals[:2],
            color_itervals[2:4],
            color_itervals[4:6],
            random_state=self.random_state,
        )
        self.assertAlmostEqual(rgb[0], 0.87, delta=0.01)
        self.assertAlmostEqual(rgb[1], 0.02, delta=0.01)
        self.assertAlmostEqual(rgb[2], 0.69, delta=0.01)

    def test_place_tree(self):
        expected_node_coords = {
            "B": (0.0, 0.0),
            "C": (1.0, 0.0),
            "D": (2.0, 0.0),
            "E": (1.5, 2.0),
            "F": (0.75, 3.0),
        }
        expected_branch_coords = {
            ("E", "C"): ([1.5, 1.0, 1.0], [2.0, 2.0, 0.0]),
            ("E", "D"): ([1.5, 2.0, 2.0], [2.0, 2.0, 0.0]),
            ("F", "B"): ([0.75, 0.0, 0.0], [3.0, 3.0, 0.0]),
            ("F", "E"): ([0.75, 1.5, 1.5], [3.0, 3.0, 2.0]),
        }
        node_coords, branch_coords = utilities.place_tree(self.basic_tree)
        for node, coords in expected_node_coords.items():
            np.testing.assert_allclose(coords, expected_node_coords[node])
        for edge, coords in expected_branch_coords.items():
            np.testing.assert_allclose(coords, expected_branch_coords[edge])

        expected_node_coords = {
            "B": (0.0, -0.2),
            "C": (1.0, -0.8),
            "D": (2.0, -0.9),
            "E": (1.5, -0.5),
            "F": (0.75, 0.0),
        }
        expected_branch_coords = {
            ("E", "C"): ([1.5, 1.0, 1.0], [-0.5, -0.5, -0.8]),
            ("E", "D"): ([1.5, 2.0, 2.0], [-0.5, -0.5, -0.9]),
            ("F", "B"): ([0.75, 0.0, 0.0], [0.0, 0.0, -0.2]),
            ("F", "E"): ([0.75, 1.5, 1.5], [0.0, 0.0, -0.5]),
        }
        node_coords, branch_coords = utilities.place_tree(
            self.basic_tree, extend_branches=False
        )
        for node, coords in expected_node_coords.items():
            np.testing.assert_allclose(coords, expected_node_coords[node])
        for edge, coords in expected_branch_coords.items():
            np.testing.assert_allclose(coords, expected_branch_coords[edge])

        expected_node_coords = {
            "B": (0.0, -4.0),
            "C": (1.0, -3.0),
            "D": (2.0, -2.0),
            "E": (1.5, -1.0),
            "F": (0.75, 0.0),
        }
        expected_branch_coords = {
            ("E", "C"): ([1.5, 1.0, 1.0], [-1.0, -1.0, -3.0]),
            ("E", "D"): ([1.5, 2.0, 2.0], [-1.0, -1.0, -2.0]),
            ("F", "B"): ([0.75, 0.0, 0.0], [0.0, 0.0, 4.0]),
            ("F", "E"): ([0.75, 1.5, 1.5], [0.0, 0.0, 1.0]),
        }
        node_coords, branch_coords = utilities.place_tree(
            self.basic_tree, depth_key="other", extend_branches=False
        )
        for node, coords in expected_node_coords.items():
            np.testing.assert_allclose(coords, expected_node_coords[node])
        for edge, coords in expected_branch_coords.items():
            np.testing.assert_allclose(coords, expected_branch_coords[edge])

        expected_node_coords = {
            "B": (0.0, -4.0),
            "C": (1.0, -4.0),
            "D": (2.0, -4.0),
            "E": (1.5, -1.0),
            "F": (0.75, 0.0),
        }
        expected_branch_coords = {
            ("E", "C"): ([1.5, 1.0, 1.0], [-1.0, -1.0, -4.0]),
            ("E", "D"): ([1.5, 2.0, 2.0], [-1.0, -1.0, -4.0]),
            ("F", "B"): ([0.75, 0.0, 0.0], [0.0, 0.0, -4.0]),
            ("F", "E"): ([0.75, 1.5, 1.5], [0.0, 0.0, -1.0]),
        }
        node_coords, branch_coords = utilities.place_tree(
            self.basic_tree, depth_key="other"
        )
        for node, coords in expected_node_coords.items():
            np.testing.assert_allclose(coords, expected_node_coords[node])
        for edge, coords in expected_branch_coords.items():
            np.testing.assert_allclose(coords, expected_branch_coords[edge])

        expected_node_coords = {
            "B": (90, 3),
            "C": (180, 3),
            "D": (270, 3),
            "E": (225, 1),
            "F": (157, 0),
        }
        expected_branch_coords = {
            ("E", "C"): ([225, 180, 180], [1, 1, 3]),
            ("E", "D"): ([225, 270, 270], [1, 1, 3]),
            ("F", "B"): ([157.5, 90, 90], [0, 0, 3]),
            ("F", "E"): ([157.5, 225, 225], [0, 0, 1]),
        }
        node_coords, branch_coords = utilities.place_tree(
            self.basic_tree, orient=0, polar_interpolation_threshold=np.inf
        )
        for node, coords in expected_node_coords.items():
            np.testing.assert_allclose(coords, expected_node_coords[node])
        for edge, coords in expected_branch_coords.items():
            np.testing.assert_allclose(coords, expected_branch_coords[edge])

    def test_place_colorstrip(self):
        expected = ({"0": ([4, 1, 1, 4, 4], [1, 1, -1, -1, 1])}, {"0": (4, 0)})
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, spacing=1, loc="right"
            ),
            expected,
        )

        expected = (
            {"0": ([-1, -4, -4, -1, -1], [1, 1, -1, -1, 1])},
            {"0": (-4, 0)},
        )
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, spacing=1, loc="left"
            ),
            expected,
        )

        expected = ({"0": ([1, -1, -1, 1, 1], [4, 4, 1, 1, 4])}, {"0": (0, 4)})
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, spacing=1, loc="up"
            ),
            expected,
        )

        expected = (
            {"0": ([1, -1, -1, 1, 1], [-1, -1, -4, -4, -1])},
            {"0": (0, -4)},
        )
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, spacing=1, loc="down"
            ),
            expected,
        )

        expected = ({"0": ([1, -1, -1, 1, 1], [4, 4, 1, 1, 4])}, {"0": (0, 4)})
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, spacing=1, loc="polar"
            ),
            expected,
        )

    def test_generate_indel_colors_from_priors(self):

        indel_to_color = utilities.get_indel_colors(
            self.indel_priors, self.random_state
        )

        expected_values = {
            "i": [0.75, 0, 0.5],
            "j": [0.05, 0.88, 0.94],
            "k": [0.69, 1.0, 1.0],
            "m": [0.798, 0.37, 0.68],
            "n": [0.56, 0.37, 0.68],
        }

        for indel in indel_to_color.index:
            self.assertIn(indel, expected_values.keys())

            observed_color = indel_to_color.loc[indel, "color"]
            for i in range(len(observed_color)):
                self.assertAlmostEqual(
                    observed_color[i], expected_values[indel][i], delta=0.01
                )

    def test_color_converters(self):

        # convert hex to rgb
        _hex = "#000000"
        rgb = utilities.hex_to_rgb(_hex)
        self.assertEqual(rgb, (0, 0, 0))

        _hex = "#812dd3"
        rgb = utilities.hex_to_rgb(_hex)
        self.assertEqual(rgb, (129, 45, 211))

        rgb = (129, 45, 211)
        _hex = utilities.rgb_to_hex(rgb)
        self.assertEqual(_hex, "#812dd3")

    def test_generate_random_indel_colors(self):

        lineage_profile = pd.DataFrame.from_dict(
            {"cellA": ["i", "j", "k", "i"], "cellB": ["i", "m", "n", "none"]},
            orient="index",
            columns=["site1", "site2", "site3", "site4"],
        )

        indel_to_color = utilities.get_random_indel_colors(
            lineage_profile, self.random_state
        )

        expected_values = {
            "i": [0.53, 0.97, 0.89],
            "j": [0.73, 0.74, 0.69],
            "k": [0.26, 0.35, 0.56],
            "m": [0.28, 0.55, 0.80],
            "n": [0.9, 0.84, 0.76],
            "none": [0, 0, 0.75],
        }
        for indel in indel_to_color.index:
            self.assertIn(indel, expected_values.keys())

            observed_color = indel_to_color.loc[indel, "color"]
            for i in range(len(observed_color)):
                self.assertAlmostEqual(
                    observed_color[i], expected_values[indel][i], delta=0.01
                )
