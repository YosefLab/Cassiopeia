"""
Tests for cassiopeia/tools/parameter_estimators.py
"""

import unittest

import cassiopeia as cas
import networkx as nx
import numpy as np
import pandas as pd
from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning
from cassiopeia.tools import parameter_estimators


class TestCassiopeiaTree(unittest.TestCase):
    def setUp(self):
        # A small network to test likelihood calculation
        small_net = nx.DiGraph()
        small_net.add_edges_from(
            [
                ("node5", "node0"),
                ("node5", "node1"),
                ("node6", "node2"),
                ("node6", "node3"),
                ("node6", "node4"),
                ("node7", "node5"),
                ("node7", "node6"),
            ]
        )
        self.small_net = small_net

        cm1 = pd.DataFrame.from_dict(
            {
                "node0": [0, -1, -1],
                "node1": [1, 1, -1],
                "node2": [1, -1, -1],
                "node3": [1, -1, -1],
                "node4": [1, -1, -1],
            },
            orient="index",
        )
        priors1 = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
        self.discrete_tree = cas.data.CassiopeiaTree(
            tree=small_net, character_matrix=cm1, priors=priors1
        )

        cm2 = pd.DataFrame.from_dict(
            {
                "node0": [1, 0],
                "node1": [1, 1],
                "node2": [2, 3],
                "node3": [-1, 2],
                "node4": [-1, 1],
            },
            orient="index",
        )
        priors2 = {
            0: {1: 0.2, 2: 0.7, 3: 0.1},
            1: {1: 0.2, 2: 0.7, 3: 0.1},
            2: {1: 0.2, 2: 0.7, 3: 0.1},
        }
        self.continuous_tree = cas.data.CassiopeiaTree(
            tree=small_net, character_matrix=cm2, priors=priors2
        )
        self.continuous_tree.set_branch_length("node5", "node0", 1.5)
        self.continuous_tree.set_branch_length("node6", "node3", 2)

    def test_proportions(self):
        prop_mut = parameter_estimators.get_proportion_of_mutation(
            self.discrete_tree
        )
        prop_missing = parameter_estimators.get_proportion_of_missing_data(
            self.discrete_tree
        )

        self.assertEqual(prop_mut, 5 / 6)
        self.assertEqual(prop_missing, 0.6)

        prop_mut = parameter_estimators.get_proportion_of_mutation(
            self.continuous_tree
        )
        prop_missing = parameter_estimators.get_proportion_of_missing_data(
            self.continuous_tree
        )

        self.assertEqual(prop_mut, 7 / 8)
        self.assertEqual(prop_missing, 0.2)

    def test_estimate_mutation_rate(self):
        mut_rate = parameter_estimators.estimate_mutation_rate(
            self.discrete_tree, continuous=False
        )
        self.assertTrue(np.isclose(mut_rate, 0.44967879185089554))

        mut_rate = parameter_estimators.estimate_mutation_rate(
            self.discrete_tree,
            continuous=False,
            assume_root_implicit_branch=False,
        )
        self.assertTrue(np.isclose(mut_rate, 0.5917517095361371))

        mut_rate = parameter_estimators.estimate_mutation_rate(
            self.continuous_tree, continuous=True
        )
        self.assertTrue(np.isclose(mut_rate, 0.5917110077950752))

        mut_rate = parameter_estimators.estimate_mutation_rate(
            self.continuous_tree,
            continuous=True,
            assume_root_implicit_branch=False,
        )
        self.assertTrue(np.isclose(mut_rate, 0.9041050181216678))

    def test_estimate_missing_data_bad_cases(self):
        with self.assertRaises(ParameterEstimateError):
            parameter_estimators.estimate_missing_data_rates(
                self.discrete_tree, continuous=False
            )

        with self.assertRaises(ParameterEstimateError):
            parameter_estimators.estimate_missing_data_rates(
                self.discrete_tree,
                continuous=False,
                heritable_missing_rate=0.25,
                stochastic_missing_probability=0.2,
            )

        with self.assertRaises(ParameterEstimateError):
            self.discrete_tree.parameters["heritable_missing_rate"] = 0.25
            self.discrete_tree.parameters[
                "stochastic_missing_probability"
            ] = 0.2
            parameter_estimators.estimate_missing_data_rates(
                self.discrete_tree, continuous=False
            )

        with self.assertRaises(ParameterEstimateWarning):
            self.discrete_tree.reset_parameters()
            self.discrete_tree.parameters["heritable_missing_rate"] = 0.5
            parameter_estimators.estimate_missing_data_rates(
                self.discrete_tree, continuous=False
            )

        with self.assertRaises(ParameterEstimateWarning):
            self.continuous_tree.parameters[
                "stochastic_missing_probability"
            ] = 0.9
            parameter_estimators.estimate_missing_data_rates(
                self.continuous_tree, continuous=True
            )

    def test_estimate_stochastic_missing_data_probability(self):
        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree, continuous=False, heritable_missing_rate=0.25
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 0.0518518518518518))

        self.discrete_tree.parameters["heritable_missing_rate"] = 0.25
        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree, continuous=False
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 0.0518518518518518))

        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree,
            continuous=False,
            assume_root_implicit_branch=False,
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 13 / 45))

        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree, continuous=True, heritable_missing_rate=0.05
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 0.046322071416968195))

        self.continuous_tree.parameters["heritable_missing_rate"] = 0.05
        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree, continuous=True
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 0.046322071416968195))

        s_missing_prob = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree,
            continuous=True,
            assume_root_implicit_branch=False,
        )[0]
        self.assertTrue(np.isclose(s_missing_prob, 0.10250124994244929))

    def test_estimate_heritable_missing_data_rate(self):
        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree,
            continuous=False,
            stochastic_missing_probability=0.12,
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.23111904017137075))

        self.discrete_tree.parameters["stochastic_missing_probability"] = 0.2
        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree, continuous=False
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.2062994740159002))

        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.discrete_tree,
            continuous=False,
            assume_root_implicit_branch=False,
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.2928932188134524))

        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree,
            continuous=True,
            stochastic_missing_probability=0.04,
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.05188011778689765))

        self.continuous_tree.parameters["stochastic_missing_probability"] = 0.1
        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree, continuous=True
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.0335154979510034))

        h_missing_rate = parameter_estimators.estimate_missing_data_rates(
            self.continuous_tree,
            continuous=True,
            assume_root_implicit_branch=False,
        )[1]
        self.assertTrue(np.isclose(h_missing_rate, 0.05121001550277538))


if __name__ == "__main__":
    unittest.main()
