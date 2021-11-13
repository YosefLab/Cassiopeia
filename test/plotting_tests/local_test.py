import unittest
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.plotting import local


class TestLocalPlotting(unittest.TestCase):
    def setUp(self):
        self.allele_table = pd.DataFrame.from_dict(
            {
                1: ["2", "A", 10, "i", "j", "k"],
                3: ["3", "A", 10, "i", "m", "n"],
                4: ["5", "A", 10, "i", "j", "k"],
                6: ["6", "A", 10, "i", "j", "m"],
            },
            orient="index",
            columns=["cellBC", "intBC", "UMI", "r1", "r2", "r3"],
        )

        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("root", "1"),
                ("1", "2"),
                ("1", "3"),
                ("root", "4"),
                ("4", "5"),
                ("4", "6"),
            ]
        )

        # cell meta
        cell_meta = pd.DataFrame(index=["2", "3", "5", "6"])
        cell_meta["nUMI"] = [1, 3, 5, 7]
        cell_meta["cluster"] = ["a", "a", "b", "b"]

        self.tree = cas.data.CassiopeiaTree(tree=graph, cell_meta=cell_meta)

    def test_compute_colorstrip_size(self):
        expected = (0.05, 1.0)
        size = local.compute_colorstrip_size(
            {"0": (0, 0), "1": (0, 1), "2": (1, 1)},
            {"1": (0, 1), "2": (1, 1)},
            loc="up",
        )
        np.testing.assert_allclose(size, expected)

        size = local.compute_colorstrip_size(
            {"0": (0, 0), "1": (0, 1), "2": (1, 1)},
            {"1": (0, 1), "2": (1, 1)},
            loc="up",
        )
        np.testing.assert_allclose(size, expected)

        size = local.compute_colorstrip_size(
            {"0": (0, 0), "1": (1, 0), "2": (1, 1)},
            {"1": (1, 0), "2": (1, 1)},
            loc="right",
        )
        np.testing.assert_allclose(size, expected)

        size = local.compute_colorstrip_size(
            {"0": (0, 0), "1": (-1, 0), "2": (-1, 1)},
            {"1": (-1, 0), "2": (-1, 1)},
            loc="left",
        )
        np.testing.assert_allclose(size, expected)

        size = local.compute_colorstrip_size(
            {"0": (0, 0), "1": (0, 1), "2": (1, 1)},
            {"1": (0, 1), "2": (1, 1)},
            loc="polar",
        )
        np.testing.assert_allclose(size, expected)

    def test_create_categorical_colorstrip(self):
        expected_colorstrip = {
            "1": ([1, -1, -1, 1, 1], [2, 2, 1, 1, 2], mock.ANY, mock.ANY),
            "2": ([2, 0, 0, 2, 2], [2, 2, 1, 1, 2], mock.ANY, mock.ANY),
        }
        expected_next_anchor_coords = {"1": (0, 2), "2": (1, 2)}
        colorstrip, next_anchor_coords = local.create_categorical_colorstrip(
            {"1": "a", "2": "b"}, {"1": (0, 0), "2": (1, 0)}, 1, 2, 1, "up"
        )
        self.assertEqual(colorstrip, expected_colorstrip)
        self.assertEqual(next_anchor_coords, expected_next_anchor_coords)

    def test_create_continous_colorstrip(self):
        expected_colorstrip = {
            "1": ([1, -1, -1, 1, 1], [2, 2, 1, 1, 2], mock.ANY, mock.ANY),
            "2": ([2, 0, 0, 2, 2], [2, 2, 1, 1, 2], mock.ANY, mock.ANY),
        }
        expected_next_anchor_coords = {"1": (0, 2), "2": (1, 2)}
        colorstrip, next_anchor_coords = local.create_continuous_colorstrip(
            {"1": -10, "2": 10}, {"1": (0, 0), "2": (1, 0)}, 1, 2, 1, "up"
        )
        self.assertEqual(colorstrip, expected_colorstrip)
        self.assertEqual(next_anchor_coords, expected_next_anchor_coords)

    def test_create_indel_heatmap(self):
        indel_colors = {
            "i": [0.75, 0, 0.5],
            "j": [0.05, 0.88, 0.94],
            "k": [0.69, 1.0, 1.0],
            "m": [0.798, 0.37, 0.68],
            "n": [0.56, 0.37, 0.68],
        }

        indel_color_df = pd.DataFrame(columns=["color"])
        for indel in indel_colors:
            indel_color_df.loc[indel, "color"] = indel_colors[indel]

        expected_anchor_coords = {
            "2": (0, 9),
            "3": (1, 9),
            "5": (2, 9),
            "6": (3, 9),
        }
        heatmap, anchor_coords = local.create_indel_heatmap(
            self.allele_table,
            {"2": (0, 0), "3": (1, 0), "5": (2, 0), "6": (3, 0)},
            2,
            1,
            1,
            "up",
            indel_colors=indel_color_df,
        )
        self.assertEqual(anchor_coords, expected_anchor_coords)
        self.assertEqual(len(heatmap), 3)
        self.assertEqual(
            heatmap[0]["6"],
            (
                [3.5, 2.5, 2.5, 3.5, 3.5],
                [3, 3, 1, 1, 3],
                (0.5, 0.5, 0.5),
                mock.ANY,
            ),
        )

    def test_plot_matplotlib(self):
        local.plot_matplotlib(self.tree)
        local.plot_matplotlib(self.tree, add_root=True)
        local.plot_matplotlib(self.tree, meta_data=["nUMI", "cluster"])
