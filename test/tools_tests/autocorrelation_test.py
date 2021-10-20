"""
Test suite for the autocorrelation functions in
cassiopeia/tools/autocorrelation.py
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas


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

    def test_simple_autocorrelation_single_variable(self):
        """
        Tests Moran's I, comparing values gotten from the function implemented
        in Chaligne et al, Nat Genetics 2021
        """

        I = cas.tl.compute_morans_i(
            self.basic_tree, X=pd.DataFrame(self.X["nUMI"])
        )

        self.assertAlmostEqual(I, 0.084456, delta=0.001)

    def test_autocorrelation_bivariate(self):

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


if __name__ == "__main__":
    unittest.main()
