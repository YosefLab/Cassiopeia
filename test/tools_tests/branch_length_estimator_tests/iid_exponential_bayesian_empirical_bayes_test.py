"""
Test CrossValidatedBLE in cassiopeia.tools.
"""
import unittest

import networkx as nx

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import \
    IIDExponentialBayesianEmpiricalBayes


class TestIIDExponentialBayesianEmpiricalBayes(unittest.TestCase):
    def test_smoke(self):
        """
        Just test that IIDExponentialBayesianEmpiricalBayes runs without errors.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree.add_edge("1", "3")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0],
                "1": [0, 0, 0, 0],
                "2": [0, 0, 0, 0],
                "3": [1, 1, 1, 1],
            }
        )
        model = IIDExponentialBayesianEmpiricalBayes(
            n_hyperparams=1,
            n_parallel_hyperparams=1,
            random_seed=0,
            verbose=True,
        )
        model.estimate_branch_lengths(tree)
        assert model.best_cv_metric > -1e8
