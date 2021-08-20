"""
Test IIDExponentialBayesian in cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np
import pytest
from parameterized import parameterized
from scipy.special import logsumexp

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import IIDExponentialBayesian


def relative_error(x: float, y: float) -> float:
    """
    Relative error between x and y.
    """
    assert x > 0 and y > 0
    return max(np.abs(y / x - 1), np.abs(x / y - 1))


class TestIIDExponentialBayesian(unittest.TestCase):
    @parameterized.expand(
        [
            ("1", 1.0, 0.8, False, 500),
            ("2", 1.0, 0.8, True, 500),
            ("3", 0.1, 5.0, False, 500),
            ("4", 0.1, 5.0, True, 500),
            ("5", 0.3, 4.0, False, 500),
            ("6", 0.3, 4.0, True, 500),
        ]
    )
    def test_against_closed_form_solution_small(
        self,
        name,
        sampling_probability,
        birth_rate,
        many_characters,
        discretization_level,
    ):
        """
        For a small tree with only one internal node, the likelihood of the
        data, and the posterior age of the internal node, can be computed
        easily in closed form. We check the theoretical values against those
        obtained from our model. We try different settings of the model
        hyperparameters, particularly the birth rate and sampling probability.
        """
        # First, we create the ground truth tree and character states
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

        # Estimate branch lengths
        mutation_rate = 0.3  # This is kind of arbitrary; not super relevant.
        model = IIDExponentialBayesian(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        model.estimate_branch_lengths(tree)

        # Test the model log likelihood vs its computation from the joint of the
        # age of vertex 1.
        re = relative_error(
            -model.log_likelihood, -logsumexp(model.log_joints("1"))
        )
        self.assertLessEqual(re, 0.01)

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialBayesian.numerical_log_likelihood(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        re = relative_error(-model.log_likelihood, -numerical_log_likelihood)
        self.assertLessEqual(re, 0.01)

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

        # Test the model posterior times against its numerical posterior
        numerical_posterior = IIDExponentialBayesian.numerical_posterior_time(
            tree=tree,
            node="1",
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        # # For debugging; these two plots should look very similar.
        # import matplotlib.pyplot as plt
        # plt.plot(model.posterior_time("1"))
        # plt.show()
        # plt.plot(numerical_posterior)
        # plt.show()
        total_variation = np.sum(
            np.abs(model.posterior_time("1") - numerical_posterior)
        )
        self.assertLessEqual(total_variation, 0.03)

        # Test the posterior mean against the numerical posterior mean.
        numerical_posterior_mean = np.sum(
            numerical_posterior
            * np.array(range(discretization_level + 1))
            / discretization_level
        )
        posterior_mean = tree.get_time("1")
        re = relative_error(posterior_mean, numerical_posterior_mean)
        self.assertLessEqual(re, 0.01)

    @pytest.mark.slow
    def test_against_closed_form_solution_medium(self):
        r"""
        Same as test_against_closed_form_solution_small but with a tree having
        two internal nodes instead of 1. This makes the test slow, but more
        interesting. Because this test is slow, we only try one sensible
        setting of the model hyperparameters.
        """
        # First, we create the ground truth tree and character states
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5"]),
        tree.add_edges_from(
            [
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("2", "4"),
                ("1", "5"),
            ]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0],
                "1": [0, 0],
                "2": [1, 0],
                "3": [1, 0],
                "4": [1, 1],
                "5": [0, 1],
            },
        )

        # Estimate branch lengths
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

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialBayesian.numerical_log_likelihood(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        re = relative_error(-model.log_likelihood, -numerical_log_likelihood)
        self.assertLessEqual(re, 0.01)

        # Check that the posterior ages of the nodes are correct.
        for node in tree.internal_nodes:
            if node == tree.root:
                continue
            numerical_log_joints = IIDExponentialBayesian.numerical_log_joints(
                tree=tree,
                node=node,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
                discretization_level=discretization_level,
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
            # # For debugging; these two plots should look very similar.
            # import matplotlib.pyplot as plt
            # plt.plot(model.posterior_time(node))
            # plt.show()
            # plt.plot(numerical_posterior)
            # plt.show()
            total_variation = np.sum(
                np.abs(model.posterior_time(node) - numerical_posterior)
            )
            self.assertLessEqual(total_variation, 0.03)

    def test_small_discretization_level_raises_error(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [-1], "3": [1]},
        )

        model = IIDExponentialBayesian(
            mutation_rate=1.0,
            birth_rate=1.0,
            sampling_probability=1.0,
            discretization_level=2,
        )
        with self.assertRaises(ValueError):
            model.estimate_branch_lengths(tree)

    def test_invalid_tree_topology_raises_error(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edges_from([("0", "1"), ("0", "2")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [0]},
        )

        model = IIDExponentialBayesian(
            mutation_rate=1.0,
            birth_rate=1.0,
            sampling_probability=1.0,
            discretization_level=500,
        )
        with self.assertRaises(ValueError):
            model.estimate_branch_lengths(tree)

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edges_from([("0", "1"), ("1", "2")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [1]},
        )

        model = IIDExponentialBayesian(
            mutation_rate=1.0,
            birth_rate=1.0,
            sampling_probability=1.0,
            discretization_level=500,
        )
        with self.assertRaises(ValueError):
            model.estimate_branch_lengths(tree)

    def test_invalid_sampling_probability_raises_error(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [-1], "3": [1]},
        )

        for sampling_probability in [-1.0, 2.0]:
            with self.assertRaises(ValueError):
                IIDExponentialBayesian(
                    mutation_rate=1.0,
                    birth_rate=1.0,
                    sampling_probability=sampling_probability,
                    discretization_level=500,
                )
