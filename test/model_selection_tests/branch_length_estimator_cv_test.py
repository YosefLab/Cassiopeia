"""
Test CrossValidatedBLE in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.model_selection import (
    IIDExponentialBayesianCrossValidated,
    IIDExponentialBayesianEmpiricalBayes,
    IIDExponentialMLECrossValidated,
)


class TestIIDExponentialMLE(unittest.TestCase):
    @parameterized.expand(
        [
            ("MLE_CV", IIDExponentialMLECrossValidated),
            ("Bayesian_CV", IIDExponentialBayesianCrossValidated),
            ("Bayesian_EmpiricalBayes", IIDExponentialBayesianEmpiricalBayes),
        ]
    )
    def test_smoke(self, name, ble_class):
        """
        TODO
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree.add_edge("1", "3")
        tree = CassiopeiaTree(tree=tree)
        np.random.seed(0)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0],
                "1": [0, 0, 0, 0],
                "2": [0, 0, 0, 0],
                "3": [1, 1, 1, 1],
            }
        )
        if name == "Bayesian_EmpiricalBayes":
            model = ble_class(
                n_hyperparams=1,
                n_parallel_hyperparams=1,
                verbose=True,
            )
        else:
            model = ble_class(
                n_hyperparams=2,
                n_parallel_hyperparams=2,
                n_folds=2,
                n_parallel_folds=2,
                verbose=True,
            )
        model.estimate_branch_lengths(tree)
        assert model.best_cv_metric > -1e8
