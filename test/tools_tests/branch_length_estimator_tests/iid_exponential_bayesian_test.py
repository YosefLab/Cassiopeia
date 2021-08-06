"""
Test IIDExponentialMLE in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized
from scipy.special import logsumexp

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import Cas9LineageTracingDataSimulator
from cassiopeia.tools import IIDExponentialBayesian


class TestIIDExponentialBayesian(unittest.TestCase):
    @parameterized.expand(
        [
            ("1", 1.0, 0.8, False, 500, 3),
            ("2", 1.0, 0.8, True, 500, 3),
            ("3", 0.1, 5.0, False, 500, 2),
            ("4", 0.1, 5.0, True, 500, 2),
            ("5", 0.3, 4.0, False, 500, 3),
            ("6", 0.3, 4.0, True, 500, 3),
        ]
    )
    def test_against_closed_form_solution(
        self,
        name,
        sampling_probability,
        birth_rate,
        many_characters,
        discretization_level,
        significance,
    ):
        r"""
        For a small tree with only one internal node, the likelihood of the data,
        and the posterior age of the internal node, can be computed easily in
        closed form. We check the theoretical values against those obtained from
        our model.
        """

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        if many_characters:
            tree.set_all_character_states(
                {
                    "0": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "1": [0, 1, 0, 0, 0, 0, 1, -1, 0],
                    "2": [0, -1, 0, 1, 1, 0, 1, -1, -1],
                    "3": [0, -1, -1, 1, 0, 0, -1, -1, -1],
                },
            )
        else:
            tree.set_all_character_states(
                {"0": [0], "1": [1], "2": [-1], "3": [1]},
            )
        tree.impute_unambiguous_missing_states()

        mutation_rate = 0.3
        model = IIDExponentialBayesian(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )

        model.estimate_branch_lengths(tree)

        # Test the model log likelihood vs its computation from the joint of the
        # age of vertex 1.
        model_log_joints = model.log_joints("1")
        model_log_likelihood_2 = logsumexp(model_log_joints)
        print(f"{model_log_likelihood_2} = {model_log_likelihood_2}")
        np.testing.assert_approx_equal(
            model.log_likelihood,
            model_log_likelihood_2,
            significant=significance,
        )

        # Test the model log likelihood vs its computation from the leaf nodes.
        for leaf in ["2", "3"]:
            model_log_likelihood_up = (
                model._up(
                    leaf,
                    discretization_level,
                    model._get_number_of_mutated_characters_in_node(tree, leaf),
                )
                - np.log(birth_rate * 1.0 / discretization_level)
                + np.log(sampling_probability)
            )
            print(f"{model_log_likelihood_up} = model_log_likelihood_up")
            np.testing.assert_approx_equal(
                model.log_likelihood,
                model_log_likelihood_up,
                significant=significance,
            )

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialBayesian.numerical_log_likelihood(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        print(f"{numerical_log_likelihood} = numerical_log_likelihood")
        np.testing.assert_approx_equal(
            model.log_likelihood,
            numerical_log_likelihood,
            significant=significance,
        )

        # Test the _whole_ array of log joints P(t_v = t, X, T) against its
        # numerical computation
        numerical_log_joints = IIDExponentialBayesian.numerical_log_joints(
            tree=tree,
            node="1",
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        np.testing.assert_array_almost_equal(
            model.log_joints("1")[50:-50],
            numerical_log_joints[50:-50],
            decimal=1,
        )

        # Test the model posterior against its numerical posterior
        numerical_posterior = IIDExponentialBayesian.numerical_posterior_time(
            tree=tree,
            node="1",
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        # import matplotlib.pyplot as plt
        # plt.plot(model.posterior_time("1"))
        # plt.show()
        # plt.plot(numerical_posterior)
        # plt.show()
        total_variation = np.sum(
            np.abs(model.posterior_time("1") - numerical_posterior)
        )
        assert total_variation < 0.03

        # Test the posterior mean against the numerical posterior mean.
        numerical_posterior_mean = np.sum(
            numerical_posterior
            * np.array(range(discretization_level + 1))
            / discretization_level
        )
        posterior_mean = tree.get_time("1")
        np.testing.assert_approx_equal(
            posterior_mean, numerical_posterior_mean, significant=2
        )

    def test_IIDExponentialPosteriorMeanBLE_2(self):
        r"""
        We run the Bayesian estimator on a small tree with all different leaves,
        and then check that:
        - The likelihood of the data, computed from all of the leaves, is the
            same.
        - The posteriors of the internal node ages matches their numerical
            counterpart.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0],
             "1": [0, 0],
             "2": [1, 0],
             "3": [0, 0],
             "4": [0, 1],
             "5": [1, 0],
             "6": [1, 1]},
        )

        mutation_rate = 0.625
        birth_rate = 0.75
        sampling_probability = 0.1
        discretization_level = 100
        model = IIDExponentialBayesian(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )

        model.estimate_branch_lengths(tree)
        print(model.log_likelihood)

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialBayesian.numerical_log_likelihood(
                tree=tree, mutation_rate=mutation_rate, birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # Check that the likelihood computed from each leaf node is correct.
        for leaf in tree.leaves:
            model_log_likelihood_up = model._up(
                leaf, discretization_level, model._get_number_of_mutated_characters_in_node(tree, leaf)
            ) - np.log(birth_rate * 1.0 / discretization_level)\
                + np.log(sampling_probability)
            print(model_log_likelihood_up)
            np.testing.assert_approx_equal(
                model.log_likelihood, model_log_likelihood_up, significant=3
            )

            model_log_likelihood_up_wrong = model._up(
                leaf, discretization_level, (model._get_number_of_mutated_characters_in_node(tree, leaf) + 1) % 2
            )
            with self.assertRaises(AssertionError):
                np.testing.assert_approx_equal(
                    model.log_likelihood,
                    model_log_likelihood_up_wrong,
                    significant=3,
                )

        # Check that the posterior ages of the nodes are correct.
        for node in model._non_root_internal_nodes(tree):
            numerical_log_joints = (
                IIDExponentialBayesian.numerical_log_joints(
                    tree=tree,
                    node=node,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                    discretization_level=discretization_level,
                )
            )
            np.testing.assert_array_almost_equal(
                model.log_joints(node)[25:-25],
                numerical_log_joints[25:-25],
                decimal=1,
            )

            # Test the model posterior against its numerical posterior.
            numerical_posterior = np.exp(
                numerical_log_joints - numerical_log_joints.max()
            )
            numerical_posterior /= numerical_posterior.sum()
            # import matplotlib.pyplot as plt
            # plt.plot(model.posteriors[node])
            # plt.show()
            # plt.plot(analytical_posterior)
            # plt.show()
            total_variation = np.sum(
                np.abs(model.posterior_time(node) - numerical_posterior)
            )
            assert total_variation < 0.03
