import unittest

import networkx as nx
import numpy as np

import cassiopeia as cas
from cassiopeia.plotting import utilities


class TestPlottingUtilities(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(123412334)

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

        x, y = utilities.polar_to_cartesian(np.pi / 2, 1)
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
        expected = {"0": ([3, 0, 0, 3, 3], [1, 1, -1, -1, 1])}
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, loc="right"
            ),
            expected,
        )

        expected = {"0": ([0, -3, -3, 0, 0], [1, 1, -1, -1, 1])}
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, loc="left"
            ),
            expected,
        )

        expected = {"0": ([1, -1, -1, 1, 1], [3, 3, 0, 0, 3])}
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, loc="up"
            ),
            expected,
        )

        expected = {"0": ([1, -1, -1, 1, 1], [0, 0, -3, -3, 0])}
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, loc="down"
            ),
            expected,
        )

        expected = {"0": ([1, -1, -1, 1, 1], [3, 3, 0, 0, 3])}
        self.assertEqual(
            utilities.place_colorstrip(
                {"0": (0, 0)}, width=3, height=2, loc="polar"
            ),
            expected,
        )
