"""
Tests for the coupling estimators implemented in cassiopeia/tools/coupling.py
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import CassiopeiaError


class TestDataUtilities(unittest.TestCase):
    def setUp(self) -> None:

        tree = nx.DiGraph()
        tree.add_edges_from(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("B", "E"),
                ("B", "F"),
                ("E", "G"),
                ("E", "H"),
                ("C", "I"),
                ("C", "J"),
            ]
        )

        meta_data = pd.DataFrame.from_dict(
            {
                "D": ["TypeB", 10],
                "F": ["TypeA", 5],
                "G": ["TypeA", 3],
                "H": ["TypeB", 22],
                "I": ["TypeC", 2],
                "J": ["TypeC", 11],
            },
            orient="index",
            columns=["CellType", "nUMI"],
        )

        self.tree = CassiopeiaTree(tree=tree, cell_meta=meta_data)

    def test_evolutionary_coupling_basic(self):

        random_state = np.random.RandomState(1231234)

        evolutionary_coupling = cas.tl.compute_evolutionary_coupling(
            self.tree,
            meta_variable="CellType",
            random_state=random_state,
            minimum_proportion=0.0,
            number_of_shuffles=10,
        )

        inter_cluster_distances = data_utilities.compute_inter_cluster_distances(
            self.tree, meta_item="CellType"
        )

        # background computed with random seed set above and 10 shuffles
        # (state1, state2): (mean, sd)
        expected_summary_stats = {
            ("TypeA", "TypeA"): (1.7, 0.6000000000000001),
            ("TypeA", "TypeB"): (3.55, 0.4716990566028302),
            ("TypeA", "TypeC"): (3.55, 0.4716990566028302),
            ("TypeB", "TypeA"): (3.55, 0.4716990566028302),
            ("TypeB", "TypeB"): (2.0, 0.5),
            ("TypeB", "TypeC"): (3.65, 0.45),
            ("TypeC", "TypeA"): (3.55, 0.4716990566028302),
            ("TypeC", "TypeB"): (3.65, 0.45),
            ("TypeC", "TypeC"): (1.8, 0.5567764362830022),
        }

        expected_coupling = inter_cluster_distances.copy()
        for s1 in expected_coupling.index:
            for s2 in expected_coupling.columns:
                mean = expected_summary_stats[(s1, s2)][0]
                sd = expected_summary_stats[(s1, s2)][1]

                expected_coupling.loc[s1, s2] = (
                    inter_cluster_distances.loc[s1, s2] - mean
                ) / sd

        pd.testing.assert_frame_equal(
            expected_coupling, evolutionary_coupling, atol=0.001
        )

        # make sure errors are raised for numerical data
        self.assertRaises(
            CassiopeiaError,
            cas.tl.compute_evolutionary_coupling,
            self.tree,
            "nUMI",
        )

    def test_evolutionary_coupling_custom_dissimilarity_map(self):

        weight_matrix = pd.DataFrame.from_dict(
            {
                "D": [0.0, 0.5, 1.2, 0.4, 0.5, 0.6],
                "F": [0.5, 0.0, 3.0, 1.1, 3.0, 0.1],
                "G": [1.2, 3.0, 0.0, 0.8, 0.2, 0.8],
                "H": [0.4, 1.1, 0.8, 0.0, 2.0, 2.1],
                "I": [0.5, 3.0, 0.2, 2.0, 0.0, 0.1],
                "J": [0.6, 0.1, 1.8, 2.1, 0.1, 0.0],
            },
            orient="index",
            columns=["D", "F", "G", "H", "I", "J"],
        )

        random_state = np.random.RandomState(1231234)

        evolutionary_coupling = cas.tl.compute_evolutionary_coupling(
            self.tree,
            meta_variable="CellType",
            random_state=random_state,
            minimum_proportion=0.0,
            number_of_shuffles=10,
            dissimilarity_map=weight_matrix,
        )

        inter_cluster_distances = data_utilities.compute_inter_cluster_distances(
            self.tree, meta_item="CellType", dissimilarity_map=weight_matrix
        )

        # background computed with random seed set above and 10 shuffles
        # (state1, state2): (mean, sd)
        expected_summary_stats = {
            ("TypeB", "TypeB"): (0.695, 0.5456418239101545),
            ("TypeB", "TypeA"): (1.0000000000000002, 0.281291663580704),
            ("TypeB", "TypeC"): (1.0925, 0.44763964301656745),
            ("TypeA", "TypeB"): (1.0000000000000002, 0.3148412298286232),
            ("TypeA", "TypeA"): (0.63, 0.4550824101193101),
            ("TypeA", "TypeC"): (1.2349999999999999, 0.391503512117069),
            ("TypeC", "TypeB"): (1.0675000000000001, 0.4493119740225047),
            ("TypeC", "TypeA"): (1.26, 0.41791147387933725),
            ("TypeC", "TypeC"): (0.4699999999999999, 0.41424630354415953),
        }

        expected_coupling = inter_cluster_distances.copy()
        for s1 in expected_coupling.index:
            for s2 in expected_coupling.columns:
                mean = expected_summary_stats[(s1, s2)][0]
                sd = expected_summary_stats[(s1, s2)][1]

                expected_coupling.loc[s1, s2] = (
                    inter_cluster_distances.loc[s1, s2] - mean
                ) / sd

        pd.testing.assert_frame_equal(
            expected_coupling, evolutionary_coupling, atol=0.001
        )

    def test_evolutionary_coupling_minimum_proportion(self):

        self.tree.cell_meta.loc["J", "CellType"] = "TypeD"

        random_state = np.random.RandomState(1231234)

        evolutionary_coupling = cas.tl.compute_evolutionary_coupling(
            self.tree,
            meta_variable="CellType",
            random_state=random_state,
            minimum_proportion=1 / 6, # This will drop types C and D
            number_of_shuffles=10,
        )

        # make sure TypeC and TypeD are not in the evolutionary coupling matrix
        expected_types = ["TypeA", "TypeB"]
        self.assertCountEqual(expected_types, evolutionary_coupling.index)
        self.assertCountEqual(expected_types, evolutionary_coupling.columns)

        # make sure couplings are correct
        inter_cluster_distances = data_utilities.compute_inter_cluster_distances(
            self.tree, meta_item="CellType"
        )

        inter_cluster_distances = inter_cluster_distances.loc[
            expected_types, expected_types
        ]

        expected_summary_stats = {
            ("TypeB", "TypeB"): (1.4, 0.19999999999999998),
            ("TypeB", "TypeA"): (2.6, 0.19999999999999998),
            ("TypeA", "TypeB"): (2.6, 0.19999999999999998),
            ("TypeA", "TypeA"): (1.4, 0.19999999999999998),
        }

        expected_coupling = inter_cluster_distances.copy()
        for s1 in expected_coupling.index:
            for s2 in expected_coupling.columns:
                mean = expected_summary_stats[(s1, s2)][0]
                sd = expected_summary_stats[(s1, s2)][1]

                expected_coupling.loc[s1, s2] = (
                    inter_cluster_distances.loc[s1, s2] - mean
                ) / sd

        evolutionary_coupling = evolutionary_coupling.loc[
            expected_types, expected_types
        ]
        pd.testing.assert_frame_equal(
            expected_coupling, evolutionary_coupling, atol=0.001
        )


if __name__ == "__main__":
    unittest.main()
