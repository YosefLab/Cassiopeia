"""
Tests for cassiopeia/tools/tree_metrics.py
"""
import itertools
import unittest

import cassiopeia as cas
import networkx as nx
import numpy as np
import pandas as pd
from cassiopeia.mixins import TreeMetricError
from cassiopeia.tools import tree_metrics


class TestCassiopeiaTree(unittest.TestCase):
    def setUp(self):
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

        parsimony_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )

        self.parsimony_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=parsimony_cm
        )

    def test_parsimony_bad_cases(self):
        small_tree = cas.data.CassiopeiaTree(tree=self.small_net)
        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_parsimony(
                small_tree, infer_ancestral_characters=False
            )

        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_parsimony(
                self.parsimony_tree, infer_ancestral_characters=False
            )

    def test_parsimony_reconstruct_internal_states(self):
        p = tree_metrics.calculate_parsimony(
            self.parsimony_tree, infer_ancestral_characters=True
        )
        self.assertEqual(p, 8)
        p = tree_metrics.calculate_parsimony(
            self.parsimony_tree,
            infer_ancestral_characters=True,
            treat_missing_as_mutation=True,
        )
        self.assertEqual(p, 12)

    def test_parsimony_specify_internal_states(self):

        self.parsimony_tree.set_character_states("node7", [0, 0, 0])
        self.parsimony_tree.set_character_states("node5", [0, 0, 0])
        self.parsimony_tree.set_character_states("node6", [0, 0, 2])

        p = tree_metrics.calculate_parsimony(
            self.parsimony_tree, infer_ancestral_characters=False
        )
        self.assertEqual(p, 9)
        p = tree_metrics.calculate_parsimony(
            self.parsimony_tree,
            infer_ancestral_characters=False,
            treat_missing_as_mutation=True,
        )
        self.assertEqual(p, 14)

    def test_log_transition_probability(self):
        priors = {0: {1: 0.2, 2: 0.7, 3: 0.1}, 1: {1: 0.2, 2: 0.6, 3: 0.2}}
        small_tree = cas.data.CassiopeiaTree(tree=self.small_net, priors=priors)
        mutation_probability_function_of_time = lambda t: t * 0.2
        missing_probability_function_of_time = lambda t: t * 0.1

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            -1,
            -1,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, np.log(1))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            1,
            -1,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.1)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            0,
            -1,
            2,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.2)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            2,
            -1,
            3,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.3)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            -1,
            "&",
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, -1e16)

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            0,
            "&",
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, np.log(0.9))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            1,
            "&",
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, np.log(0.9))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            0,
            0,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.72)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            0,
            0,
            2,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.48)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            -1,
            0,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, -1e16)

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            1,
            0,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, -1e16)

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            -1,
            2,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertEqual(p, -1e16)

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            1,
            1,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.9)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            2,
            2,
            3,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.7)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            0,
            0,
            2,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.2 * 0.9 * 0.7)))

        p = tree_metrics.log_transition_probability(
            small_tree,
            1,
            0,
            2,
            1,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
        )
        self.assertTrue(np.isclose(p, np.log(0.2 * 0.9 * 0.6)))

    def test_log_likelihood_of_character(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [0, -1, -1],
                "node1": [1, 1, -1],
                "node2": [1, -1, -1],
                "node3": [1, -1, -1],
                "node4": [1, -1, -1],
            },
            orient="index",
        )
        priors = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        stochastic_missing_probability = 0.3
        mutation_probability_function_of_time = lambda t: 0.44967879185089554
        missing_probability_function_of_time = lambda t: 0.17017346663375654

        L = tree_metrics.log_likelihood_of_character(
            small_tree,
            0,
            False,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
            stochastic_missing_probability,
            1,
        )

        self.assertTrue(np.isclose(L, np.log(0.0014153576307335343)))

        L = tree_metrics.log_likelihood_of_character(
            small_tree,
            1,
            False,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
            stochastic_missing_probability,
            1,
        )

        self.assertTrue(np.isclose(L, np.log(0.03230988091167525)))

        L = tree_metrics.log_likelihood_of_character(
            small_tree,
            2,
            False,
            mutation_probability_function_of_time,
            missing_probability_function_of_time,
            stochastic_missing_probability,
            1,
        )
        self.assertTrue(np.isclose(L, np.log(0.23080700775778995)))

    def test_bad_lineage_tracing_parameters(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )

        with self.assertRaises(TreeMetricError):
            small_tree.parameters["mutation_rate"] = -1
            tree_metrics.calculate_likelihood_continuous(small_tree)
        with self.assertRaises(TreeMetricError):
            small_tree.parameters["mutation_rate"] = -1
            tree_metrics.calculate_likelihood_discrete(small_tree)
        with self.assertRaises(TreeMetricError):
            small_tree.parameters["heritable_missing_rate"] = -1
            tree_metrics.calculate_likelihood_continuous(small_tree)
        with self.assertRaises(TreeMetricError):
            small_tree.parameters["heritable_missing_rate"] = 1.5
            tree_metrics.calculate_likelihood_discrete(small_tree)
        with self.assertRaises(TreeMetricError):
            small_tree.parameters["stochastic_missing_probability"] = -1
            tree_metrics.calculate_likelihood_continuous(small_tree)
        with self.assertRaises(TreeMetricError):
            small_tree.parameters["stochastic_missing_probability"] = 1.5
            tree_metrics.calculate_likelihood_continuous(small_tree)

    def test_get_lineage_tracing_parameters(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [0, -1, -1],
                "node1": [1, 1, -1],
                "node2": [1, -1, -1],
                "node3": [1, -1, -1],
                "node4": [1, -1, -1],
            },
            orient="index",
        )
        priors = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )

        small_tree.parameters["stochastic_missing_probability"] = 0.3
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=True
        )
        true_params = (0.44967879185089554, 0.17017346663375654, 0.3)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=False
        )
        true_params = (0.5917517095361371, 0.2440710539815455, 0.3)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        small_tree.reset_parameters()
        small_tree.parameters["heritable_missing_rate"] = 0.25
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=True
        )
        true_params = (0.44967879185089554, 0.25, 0.0518518518518518)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=False
        )
        true_params = (0.5917517095361371, 0.25, 0.28888888888888886)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)


        small_tree.reset_parameters()
        small_tree.parameters["stochastic_missing_probability"] = 0.3
        small_tree.parameters["heritable_missing_rate"] = 0.25
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=True
        )
        true_params = (0.44967879185089554, 0.25, 0.3)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)


        small_tree.parameters["mutation_rate"] = 0.25
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=False, assume_root_implicit_branch=True
        )
        self.assertEqual(params, (0.25, 0.25, 0.3))

        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, 0],
                "node1": [1, 1],
                "node2": [2, 3],
                "node3": [-1, 2],
                "node4": [-1, 1],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.2, 2: 0.7, 3: 0.1},
            1: {1: 0.2, 2: 0.7, 3: 0.1},
            2: {1: 0.2, 2: 0.7, 3: 0.1},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.set_branch_length("node5", "node0", 1.5)
        small_tree.set_branch_length("node6", "node3", 2)

        small_tree.parameters["stochastic_missing_probability"] = 0.1
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=True
        )
        true_params = (0.5917110077950752, 0.033515497951003406, 0.1)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=False
        )
        true_params = (0.90410501812166781, 0.05121001550277539, 0.1)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        small_tree.reset_parameters()
        small_tree.parameters["heritable_missing_rate"] = 0.05
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=True
        )
        true_params = (0.5917110077950752, 0.05, 0.046322071416968195)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=False
        )
        true_params = (0.9041050181216678, 0.05, 0.10250124994244929)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        small_tree.reset_parameters()
        small_tree.parameters["stochastic_missing_probability"] = 0.3
        small_tree.parameters["heritable_missing_rate"] = 0.25
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=True
        )
        true_params = (0.5917110077950752, 0.25, 0.3)
        for i in range(len(params)):
            self.assertAlmostEqual(params[i], true_params[i], delta=1e-6)

        small_tree.parameters["mutation_rate"] = 0.25
        params = tree_metrics.get_lineage_tracing_parameters(
            small_tree, continuous=True, assume_root_implicit_branch=True
        )
        self.assertEqual(params, (0.25, 0.25, 0.3))

    def test_likelihood_bad_cases(self):
        small_tree = cas.data.CassiopeiaTree(tree=self.small_net)
        small_tree.parameters["stochastic_missing_probability"] = 0.2
        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_likelihood_discrete(small_tree)

        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )
        small_tree.parameters["stochastic_missing_probability"] = 0.2
        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_likelihood_discrete(small_tree)

        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.parameters["stochastic_missing_probability"] = 0.2

        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_likelihood_discrete(
                small_tree, use_internal_character_states=True
            )

        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])

        with self.assertRaises(TreeMetricError):
            tree_metrics.calculate_likelihood_discrete(
                small_tree,
                use_internal_character_states=True,
            )

        small_tree.set_character_states("node6", [0, 0, 1])
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
            use_internal_character_states=True,
        )
        self.assertEqual(-np.inf, L)

    def test_likelihood_simple_mostly_missing(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [0, -1, -1],
                "node1": [1, 1, -1],
                "node2": [1, -1, -1],
                "node3": [1, -1, -1],
                "node4": [1, -1, -1],
            },
            orient="index",
        )
        priors = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.parameters["stochastic_missing_probability"] = 0.3
        L = tree_metrics.calculate_likelihood_discrete(small_tree)
        self.assertTrue(np.isclose(L, -11.458928604116634))

        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["stochastic_missing_probability"] = 0.2
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
        )
        self.assertTrue(np.isclose(L, -11.09716890609409))

        small_tree.parameters.pop("stochastic_missing_probability")
        small_tree.parameters["heritable_missing_rate"] = 0.25
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
        )
        self.assertTrue(np.isclose(L, -10.685658651089808))

        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
        )
        self.assertTrue(np.isclose(L, -10.549534744691526))

    def test_likelihood_more_complex_case(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1, 1],
                "node1": [2, 1, -1, 1],
                "node2": [2, -1, -1, -1],
                "node3": [1, 2, 2, -1],
                "node4": [1, 1, 2, 1],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )

        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tree_metrics.calculate_likelihood_discrete(small_tree)
        self.assertTrue(np.isclose(L, -33.11623901010781))

    def test_likelihood_set_internal_states(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        small_tree.reconstruct_ancestral_characters()
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
            use_internal_character_states=True,
        )
        self.assertTrue(np.isclose(L, -24.57491637086155))

        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])
        small_tree.set_character_states("node6", [0, 0, 2])
        L = tree_metrics.calculate_likelihood_discrete(
            small_tree,
            use_internal_character_states=True,
        )
        self.assertTrue(np.isclose(L, -28.68500929005179))

    def test_likelihood_time(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, 0],
                "node1": [1, 1],
                "node2": [2, 3],
                "node3": [-1, 2],
                "node4": [-1, 1],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.2, 2: 0.7, 3: 0.1},
            1: {1: 0.2, 2: 0.7, 3: 0.1},
            2: {1: 0.2, 2: 0.7, 3: 0.1},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.set_branch_length("node5", "node0", 1.5)
        small_tree.set_branch_length("node6", "node3", 2)

        small_tree.parameters["stochastic_missing_probability"] = 0.1
        L = tree_metrics.calculate_likelihood_continuous(small_tree)
        self.assertTrue(np.isclose(L, -20.5238276768878))

        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["stochastic_missing_probability"] = 0.1
        L = tree_metrics.calculate_likelihood_continuous(small_tree)
        self.assertTrue(np.isclose(L, -20.67410206503938))

        small_tree.parameters.pop("stochastic_missing_probability")
        small_tree.parameters["heritable_missing_rate"] = 0.05
        L = tree_metrics.calculate_likelihood_continuous(small_tree)
        self.assertTrue(np.isclose(L, -20.959879404598198))

        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tree_metrics.calculate_likelihood_continuous(small_tree)
        self.assertTrue(np.isclose(L, -21.943439525312456))

        small_tree.parameters["stochastic_missing_probability"] = 0.2
        L = tree_metrics.calculate_likelihood_continuous(small_tree)
        self.assertTrue(np.isclose(L, -22.926786566275887))

    def test_likelihood_sum_to_one(self):
        priors = {0: {1: 0.2, 2: 0.8}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.2, 2: 0.8}}
        ls_branch = []
        ls_no_branch = []
        for (
            a,
            b,
        ) in itertools.product([0, 1, -1, 2], repeat=2):
            for a_, b_ in itertools.product([0, 1, -1, 2], repeat=2):
                small_net = nx.DiGraph()
                small_net.add_edges_from(
                    [("node2", "node0"), ("node2", "node1"), ("node3", "node2")]
                )
                small_cm = pd.DataFrame.from_dict(
                    {
                        "node0": [a, a_],
                        "node1": [b, b_],
                    },
                    orient="index",
                )
                small_tree = cas.data.CassiopeiaTree(
                    tree=small_net, character_matrix=small_cm, priors=priors
                )
                small_tree.parameters["mutation_rate"] = 0.5
                small_tree.parameters["heritable_missing_rate"] = 0.25
                small_tree.parameters["stochastic_missing_probability"] = 0.25
                L_no_branch = tree_metrics.calculate_likelihood_discrete(
                    small_tree,
                    use_internal_character_states=False,
                )
                L_branch = tree_metrics.calculate_likelihood_continuous(
                    small_tree,
                    use_internal_character_states=False,
                )
                ls_no_branch.append(np.exp(L_no_branch))
                ls_branch.append(np.exp(L_branch))
        self.assertTrue(np.isclose(sum(ls_branch), 1.0))
        self.assertTrue(np.isclose(sum(ls_no_branch), 1.0))


if __name__ == "__main__":
    unittest.main()
