"""
Test suite for the autocorrelation functions in
cassiopeia/tools/autocorrelation.py
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.mixins.errors import AutocorrelationError
from cassiopeia.tools.autocorrelation import compute_morans_i


class TestAutocorrelation(unittest.TestCase):
    def setUp(self) -> None:

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        self.basic_tree = cas.data.CassiopeiaTree(tree=tree)

        example_obs = pd.DataFrame.from_dict(
            {
                "nUMI": [10, 10, 3, 3],
                "GeneX": [3, 5, 10, 2],
                "GeneY": [30, 30, 1, 1],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        ).T

        self.X = example_obs

    def test_simple_moran_single_variable(self):
        """
        Tests Moran's I, comparing values gotten from the function implemented
        in Chaligne et al, Nat Genetics 2021
        """

        I = cas.tl.compute_morans_i(
            self.basic_tree, X=pd.DataFrame(self.X["nUMI"])
        )

        self.assertAlmostEqual(I, 0.084456, delta=0.001)

    def test_moran_bivariate(self):
        """
        Statistics compared to the function implemented in Chaligne et al,
        Nat Gen 2021
        """
        I = cas.tl.compute_morans_i(self.basic_tree, X=self.X)

        expected_correlations = pd.DataFrame.from_dict(
            {
                "nUMI": [0.08445, -0.00874, 0.08446],
                "GeneX": [-0.00874, -0.31810, -0.00874],
                "GeneY": [0.08446, -0.00874, 0.08446],
            },
            orient="index",
            columns=["nUMI", "GeneX", "GeneY"],
        )

        pd.testing.assert_frame_equal(
            I, expected_correlations, check_exact=False, atol=0.001
        )

    def test_moran_custom_weights(self):

        W = pd.DataFrame.from_dict(
            {
                "A": [0, 1 / 2, 1 / 3, 1 / 3],
                "B": [1 / 2, 0, 1 / 3, 1 / 3],
                "C": [1 / 3, 1 / 3, 0, 1 / 2],
                "D": [1 / 3, 1 / 3, 1 / 2, 0],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        )

        I = cas.tl.compute_morans_i(
            self.basic_tree, X=pd.DataFrame(self.X["nUMI"]), W=W
        )

        self.assertAlmostEqual(I, -0.1428571, delta=0.0001)

    def test_moran_exceptions(self):

        # check typing
        string_type_meta = pd.DataFrame(
            ["type1", "type2", "type1", "type3"],
            index=["A", "B", "C", "D"],
            columns=["CellType"],
        )

        X = pd.concat([self.X, string_type_meta])

        self.assertRaises(
            AutocorrelationError,
            cas.tl.compute_morans_i,
            self.basic_tree,
            None,
            X,
        )

        # check all leaves are accounted for
        new_row = pd.DataFrame.from_dict(
            {"E": [5, 5, 5]}, orient="index", columns=["nUMI", "GeneX", "GeneY"]
        )

        X = pd.concat([self.X, new_row], axis=1)

        self.assertRaises(
            AutocorrelationError,
            cas.tl.compute_morans_i,
            self.basic_tree,
            None,
            X,
        )

        # make sure some data is passed in
        self.assertRaises(
            AutocorrelationError,
            cas.tl.compute_morans_i,
            self.basic_tree,
            None,
            None,
        )
        
        # make sure weight matrix has the right leaves
        W = pd.DataFrame.from_dict(
            {
                "A": [0, 1 / 2, 1 / 3],
                "B": [1 / 2, 0, 1 / 3],
                "C": [1 / 3, 1 / 3, 0],
            },
            orient="index",
            columns=["A", "B", "C"],
        )
        self.assertRaises(
            AutocorrelationError,
            cas.tl.compute_morans_i,
            self.basic_tree,
            None,
            self.X,
            W
        )

if __name__ == "__main__":
    unittest.main()
