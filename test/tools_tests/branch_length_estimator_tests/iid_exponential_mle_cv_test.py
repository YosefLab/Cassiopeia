"""
Test CrossValidatedBLE in cassiopeia.tools.
"""
import unittest

import networkx as nx
import pytest

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import \
    IIDExponentialMLECrossValidated


class TestIIDExponentialMLECrossValidated(unittest.TestCase):
    @pytest.mark.slow
    def test_smoke(self):
        """
        Just test that IIDExponentialMLECrossValidated runs without errors.
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
        model = IIDExponentialMLECrossValidated(
            n_parallel_hyperparams=1,
            n_folds=2,
            n_parallel_folds=2,
            verbose=True,
        )
        model.estimate_branch_lengths(tree)
        assert model.best_cv_metric > -1e8
        # The lowest level of regularization should have been chosen, because
        # the CV metric is train log-likelihood, which is maximized when the
        # train penalized_log_likelihood is as similar to the log-likelihood,
        # which is when there is least regularization.
        self.assertAlmostEqual(model.pseudo_mutations_per_edge, 0.015625)
