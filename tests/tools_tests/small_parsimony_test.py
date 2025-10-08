"""
Tests for cassiopeia/tools/small_parsimony.py
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.mixins import (
    CassiopeiaError,
    CassiopeiaTreeError,
    FitchCountError,
)
from cassiopeia.tools.small_parsimony import (
    fitch_hartigan_bottom_up,
    fitch_hartigan_top_down,
)


class TestSmallParsimony(unittest.TestCase):
    def setUp(self):
        binary_tree = nx.DiGraph()
        binary_tree.add_edges_from(
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
        basic_meta = pd.DataFrame.from_dict(
            {
                "7": ["A", 10],
                "8": ["G", 2],
                "9": ["A", 2],
                "10": ["A", 12],
                "11": ["C", 1],
                "12": ["A", 5],
                "13": ["G", 8],
                "14": ["G", 9],
            },
            orient="index",
            columns=["nucleotide", "quality"],
        )

        self.binary_tree = cas.data.CassiopeiaTree(tree=binary_tree, cell_meta=basic_meta)

        general_tree = nx.DiGraph()
        general_tree.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "14"),
                ("4", "8"),
                ("4", "9"),
                ("4", "10"),
                ("5", "11"),
                ("5", "12"),
                ("5", "13"),
                ("1", "15"),
                ("4", "16"),
            ]
        )

        general_meta = pd.DataFrame.from_dict(
            {
                "3": ["A", 10],
                "8": ["G", 2],
                "9": ["G", 2],
                "10": ["A", 12],
                "11": ["C", 1],
                "12": ["A", 5],
                "13": ["G", 8],
                "14": ["G", 9],
                "15": ["G", 22],
                "16": ["A", 5],
            },
            orient="index",
            columns=["nucleotide", "quality"],
        )

        self.general_tree = cas.data.CassiopeiaTree(tree=general_tree, cell_meta=general_meta)

    def test_fitch_hartigan_bottom_up(self):
        self.assertRaises(
            CassiopeiaError,
            fitch_hartigan_bottom_up,
            self.binary_tree,
            "quality",
        )
        self.assertRaises(
            CassiopeiaError,
            fitch_hartigan_bottom_up,
            self.binary_tree,
            "imaginary_column",
        )

        fitch_tree = fitch_hartigan_bottom_up(self.binary_tree, "nucleotide", copy=True)

        expected_sets = {
            "0": ["A"],
            "1": ["A"],
            "2": ["C", "A", "G"],
            "3": ["A", "G"],
            "4": ["A"],
            "5": ["C", "A"],
            "6": ["G"],
            "7": ["A"],
            "8": ["G"],
            "9": ["A"],
            "10": ["A"],
            "11": ["C"],
            "12": ["A"],
            "13": ["G"],
            "14": ["G"],
        }

        for n in fitch_tree.depth_first_traverse_nodes():
            node_states = fitch_tree.get_attribute(n, "S1")
            self.assertCountEqual(node_states, expected_sets[n])

        fitch_hartigan_bottom_up(self.binary_tree, "nucleotide", add_key="possible_states")
        for n in self.binary_tree.depth_first_traverse_nodes():
            self.assertRaises(CassiopeiaTreeError, self.binary_tree.get_attribute, n, "S1")
            node_states = self.binary_tree.get_attribute(n, "possible_states")
            self.assertCountEqual(node_states, expected_sets[n])

    def test_fitch_hartigan_top_down(self):
        np.random.seed(1234)

        fitch_tree = fitch_hartigan_bottom_up(self.binary_tree, "nucleotide", copy=True)
        fitch_hartigan_top_down(fitch_tree)

        expected_labels = {
            "0": "A",
            "1": "A",
            "2": "A",
            "3": "A",
            "4": "A",
            "5": "A",
            "6": "G",
            "7": "A",
            "8": "G",
            "9": "A",
            "10": "A",
            "11": "C",
            "12": "A",
            "13": "G",
            "14": "G",
        }

        for n in fitch_tree.depth_first_traverse_nodes():
            node_states = fitch_tree.get_attribute(n, "label")
            self.assertCountEqual(node_states, expected_labels[n])

        fitch_hartigan_bottom_up(self.binary_tree, "nucleotide")
        fitch_hartigan_top_down(self.binary_tree, label_key="nucleotide_assignment")

        for n in self.binary_tree.depth_first_traverse_nodes():
            self.assertRaises(CassiopeiaTreeError, self.binary_tree.get_attribute, n, "label")
            node_states = self.binary_tree.get_attribute(n, "nucleotide_assignment")
            self.assertCountEqual(node_states, expected_labels[n])

    def test_fitch_hartigan(self):
        np.random.seed(1234)

        cas.tl.fitch_hartigan(self.binary_tree, "nucleotide")
        expected_labels = {
            "0": "A",
            "1": "A",
            "2": "A",
            "3": "A",
            "4": "A",
            "5": "A",
            "6": "G",
            "7": "A",
            "8": "G",
            "9": "A",
            "10": "A",
            "11": "C",
            "12": "A",
            "13": "G",
            "14": "G",
        }

        for n in self.binary_tree.depth_first_traverse_nodes():
            node_states = self.binary_tree.get_attribute(n, "label")
            self.assertCountEqual(node_states, expected_labels[n])

    def test_score_parsimony(self):
        self.assertRaises(
            CassiopeiaError,
            cas.tl.score_small_parsimony,
            self.binary_tree,
            "nucleotide",
            None,
            False,
            "label",
        )

        parsimony = cas.tl.score_small_parsimony(self.binary_tree, "nucleotide", infer_ancestral_states=True)
        self.assertEqual(parsimony, 3)

        fitch_tree = cas.tl.fitch_hartigan(
            self.binary_tree,
            "nucleotide",
            label_key="inferred_nucleotide",
            copy=True,
        )
        parsimony = cas.tl.score_small_parsimony(
            fitch_tree,
            "nucleotide",
            infer_ancestral_states=False,
            label_key="inferred_nucleotide",
        )
        self.assertEqual(parsimony, 3)

    def test_general_tree_fitch_bottom_up(self):
        fitch_hartigan_bottom_up(self.general_tree, "nucleotide")

        expected_sets = {
            "0": ["G"],
            "1": ["G", "A"],
            "2": ["G"],
            "3": ["A"],
            "4": ["G", "A"],
            "5": ["C", "A", "G"],
            "8": ["G"],
            "9": ["G"],
            "10": ["A"],
            "11": ["C"],
            "12": ["A"],
            "13": ["G"],
            "14": ["G"],
            "15": ["G"],
            "16": ["A"],
        }

        for n in self.general_tree.depth_first_traverse_nodes():
            node_states = self.general_tree.get_attribute(n, "S1")
            self.assertCountEqual(node_states, expected_sets[n])

    def test_general_tree_fitch_hartigan(self):
        np.random.seed(1234)

        cas.tl.fitch_hartigan(self.general_tree, "nucleotide")
        expected_labels = {
            "0": "G",
            "1": "G",
            "2": "G",
            "3": "A",
            "4": "G",
            "5": "G",
            "8": "G",
            "9": "G",
            "10": "A",
            "11": "C",
            "12": "A",
            "13": "G",
            "14": "G",
            "15": "G",
            "16": "A",
        }

        for n in self.general_tree.depth_first_traverse_nodes():
            node_states = self.general_tree.get_attribute(n, "label")
            self.assertCountEqual(node_states, expected_labels[n])

    def test_general_tree_parsimony(self):
        parsimony = cas.tl.score_small_parsimony(self.general_tree, "nucleotide", infer_ancestral_states=True)
        self.assertEqual(parsimony, 5)

    def test_fitch_count_basic_binary(self):
        fitch_matrix = cas.tl.fitch_count(self.binary_tree, "nucleotide")

        num_nucleotides = self.binary_tree.cell_meta["nucleotide"].nunique()
        self.assertEqual(fitch_matrix.shape, (num_nucleotides, num_nucleotides))

        expected_matrix = pd.DataFrame.from_dict(
            {"A": [9, 2, 1], "G": [0, 2, 0], "C": [0, 0, 0]},
            orient="index",
            columns=["A", "G", "C"],
        ).astype(float)

        pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)

        # test if ancestral states are already assigned
        fitch_hartigan_bottom_up(self.binary_tree, "nucleotide", add_key="nucleotide_sets")
        fitch_matrix_no_ancestral_state_inferred = cas.tl.fitch_count(
            self.binary_tree,
            "nucleotide",
            infer_ancestral_states=False,
            state_key="nucleotide_sets",
        )

        pd.testing.assert_frame_equal(expected_matrix, fitch_matrix_no_ancestral_state_inferred)

    def test_fitch_count_basic_binary_custom_state_space(self):
        fitch_matrix = cas.tl.fitch_count(self.binary_tree, "nucleotide", unique_states=["A", "G", "C", "N"])

        expected_matrix = pd.DataFrame.from_dict(
            {"A": [9, 2, 1, 0], "G": [0, 2, 0, 0], "C": [0, 0, 0, 0], "N": [0, 0, 0, 0]},
            orient="index",
            columns=["A", "G", "C", "N"],
        ).astype(float)

        pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)

        with self.assertRaises(FitchCountError):
            fitch_matrix = cas.tl.fitch_count(self.binary_tree, "nucleotide", unique_states=["A", "G"])

    def test_fitch_count_basic_binary_internal_node(self):
        fitch_matrix = cas.tl.fitch_count(self.binary_tree, "nucleotide", root="5")

        num_nucleotides = self.binary_tree.cell_meta["nucleotide"].nunique()
        self.assertEqual(fitch_matrix.shape, (num_nucleotides, num_nucleotides))

        expected_matrix = pd.DataFrame.from_dict(
            {"A": [1, 0, 1], "G": [0, 0, 0], "C": [1, 0, 1]},
            orient="index",
            columns=["A", "G", "C"],
        ).astype(float)

        pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)

    def test_fitch_count_general_tree(self):
        fitch_matrix = cas.tl.fitch_count(self.general_tree, "nucleotide")

        num_nucleotides = self.general_tree.cell_meta["nucleotide"].nunique()
        self.assertEqual(fitch_matrix.shape, (num_nucleotides, num_nucleotides))

        expected_matrix = pd.DataFrame.from_dict(
            {"A": [0, 0, 0], "G": [4, 9, 1], "C": [0, 0, 0]},
            orient="index",
            columns=["A", "G", "C"],
        ).astype(float)

        pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)


if __name__ == "__main__":
    unittest.main()
