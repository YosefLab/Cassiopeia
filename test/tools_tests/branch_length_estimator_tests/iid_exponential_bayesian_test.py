"""
Test IIDExponentialBayesian in cassiopeia.tools.
"""
import unittest
from copy import deepcopy
from typing import List

import networkx as nx
import numpy as np
import pytest
from parameterized import parameterized
from scipy import integrate
from scipy.special import binom, logsumexp

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import IIDExponentialBayesian


def _non_root_internal_nodes(tree: CassiopeiaTree) -> List[str]:
    """Internal nodes of the tree, excluding the root.

    Returns:
        The internal nodes of the tree that are not the root (i.e. all
            nodes not at the leaves, and not the root)
    """
    return list(set(tree.internal_nodes) - set(tree.root))


def calc_exact_log_full_joint(
    tree: CassiopeiaTree,
    mutation_rate: float,
    birth_rate: float,
    sampling_probability: float,
) -> float:
    """
    Exact log full joint probability computation.

    This method is used for testing the implementation of the model.

    The log full joint probability density of the observed tree topology,
    state vectors, and branch lengths. In other words:
    log P(branch lengths, character states, tree topology)
    Intergrating this function allows computing the marginals and hence
    the posteriors of the times of any internal node in the tree.

    Note that this method is only fast enough for small trees. It's
    run time scales exponentially with the number of internal nodes of the
    tree.

    Args:
        tree: The CassiopeiaTree containing the tree topology and all
            character states.
        node: An internal node of the tree, for which to compute the
            posterior log joint.
        mutation_rate: The mutation rate of the model.
        birth_rate: The birth rate of the model.
        sampling_probability: The sampling probability of the model.

    Returns:
        log P(branch lengths, character states, tree topology)
    """
    tree = deepcopy(tree)
    ll = 0.0
    lam = birth_rate
    r = mutation_rate
    p = sampling_probability
    q_inv = (1.0 - p) / p
    lg = np.log
    e = np.exp
    b = binom
    T = tree.get_max_depth_of_tree()
    for (p, c) in tree.edges:
        t = tree.get_branch_length(p, c)
        # Birth process with subsampling likelihood
        h = T - tree.get_time(p) + tree.get_time(tree.root)
        h_tilde = T - tree.get_time(c) + tree.get_time(tree.root)
        if c in tree.leaves:
            # "Easy" case
            assert h_tilde == 0
            ll += (
                2.0 * lg(q_inv + 1.0)
                + lam * h
                - 2.0 * lg(q_inv + e(lam * h))
                + lg(sampling_probability)
            )
        else:
            ll += (
                lg(lam)
                + lam * h
                - 2.0 * lg(q_inv + e(lam * h))
                + 2.0 * lg(q_inv + e(lam * h_tilde))
                - lam * h_tilde
            )
        # Mutation process likelihood
        cuts = len(
            tree.get_mutations_along_edge(
                p, c, treat_missing_as_mutations=False
            )
        )
        uncuts = tree.get_character_states(c).count(0)
        # Care must be taken here, we might get a nan
        if np.isnan(lg(1 - e(-t * r)) * cuts):
            return -np.inf
        ll += (
            (-t * r) * uncuts
            + lg(1 - e(-t * r)) * cuts
            + lg(b(cuts + uncuts, cuts))
        )
    return ll


def calc_numerical_log_likelihood(
    tree: CassiopeiaTree,
    mutation_rate: float,
    birth_rate: float,
    sampling_probability: float,
    epsrel: float = 0.01,
) -> np.array:
    """
    Numerical log likelihood of the observed data.

    This method is used for testing the implementation of log_likelihood of
    the model.

    The log likelihood of the observed tree topology and state vectors,
    i.e.:
    log P(character states, tree topology)

    Note that this method is only fast enough for small trees. Its
    run time scales exponentially with the number of internal nodes of the
    tree.

    Args:
        tree: The CassiopeiaTree containing the tree topology and all
            character states.
        mutation_rate: The mutation rate of the model.
        birth_rate: The birth rate of the model.
        sampling_probability: The sampling probability of the model.
        epsrel: The degree of tolerance for the numerical integrals
            performed.

    Returns:
        log P(character states, tree topology)
    """
    tree = deepcopy(tree)

    def f(*args):
        times_list = args
        times = {}
        for node, t in list(zip(_non_root_internal_nodes(tree), times_list)):
            times[node] = t
        times[tree.root] = 0
        for leaf in tree.leaves:
            times[leaf] = 1.0
        for (p, c) in tree.edges:
            if times[p] >= times[c]:
                return 0.0
        tree.set_times(times)
        return np.exp(
            calc_exact_log_full_joint(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )

    res = np.log(
        integrate.nquad(
            f,
            [[0, 1]] * len(_non_root_internal_nodes(tree)),
            opts={"epsrel": epsrel},
        )[0]
    )
    assert not np.isnan(res)
    return res


def calc_numerical_log_joints(
    tree: CassiopeiaTree,
    node: str,
    mutation_rate: float,
    birth_rate: float,
    sampling_probability: float,
    discretization_level: int,
    epsrel: float = 0.01,
) -> np.array:
    """
    Numerical log joint probability computation.

    This method is used for testing the implementation of log_joints of the
    model.

    The log joint probability density of the observed tree topology, state
    vectors, and all possible times of a node in the tree. In other words:
    log P(node time = t, character states, tree topology) for t in [0, T]
    where T is the discretization_level.

    Note that this method is only fast enough for small trees. It's
    run time scales exponentially with the number of internal nodes of the
    tree.

    Args:
        tree: The CassiopeiaTree containing the tree topology and all
            character states.
        node: An internal node of the tree, for which to compute the
            posterior log joint.
        mutation_rate: The mutation rate of the model.
        birth_rate: The birth rate of the model.
        sampling_probability: The sampling probability of the model.
        discretization_level: The number of timesteps used to discretize
            time. The output thus is a vector of length
            discretization_level + 1.
        epsrel: The degree of tolerance for the numerical integrals
            performed.

    Returns:
        log P(node time = t, character states, tree topology) for t in
            [0, T], where T is the discretization_level.
    """
    res = np.zeros(shape=(discretization_level + 1,))
    other_nodes = [n for n in _non_root_internal_nodes(tree) if n != node]
    node_time = -1

    tree = deepcopy(tree)

    def f(*args):
        times_list = args
        times = {}
        times[node] = node_time
        assert len(other_nodes) == len(times_list)
        for other_node, t in list(zip(other_nodes, times_list)):
            times[other_node] = t
        times[tree.root] = 0
        for leaf in tree.leaves:
            times[leaf] = 1.0
        for (p, c) in tree.edges:
            if times[p] >= times[c]:
                return 0.0
        tree.set_times(times)
        return np.exp(
            calc_exact_log_full_joint(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )

    for i in range(discretization_level + 1):
        node_time = i / discretization_level
        if len(other_nodes) == 0:
            # There is nothing to integrate over.
            times = {}
            times[tree.root] = 0
            for leaf in tree.leaves:
                times[leaf] = 1.0
            times[node] = node_time
            tree.set_times(times)
            res[i] = calc_exact_log_full_joint(
                tree=tree,
                mutation_rate=mutation_rate,
                birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
            res[i] -= np.log(discretization_level)
        else:
            res[i] = (
                np.log(
                    integrate.nquad(
                        f,
                        [[0, 1]] * (len(_non_root_internal_nodes(tree)) - 1),
                        opts={"epsrel": epsrel},
                    )[0]
                )
                - np.log(discretization_level)
            )
            assert not np.isnan(res[i])

    return res


def numerical_posterior_time(
    tree: CassiopeiaTree,
    node: str,
    mutation_rate: float,
    birth_rate: float,
    sampling_probability: float,
    discretization_level: int,
    epsrel: float = 0.01,
) -> np.array:
    """
    Numerical posterior time inference under the model.

    This method is used for testing the implementation of posterior_time of
    the model.

    The posterior time distribution of a node, numerically computed, i.e.:
    P(node time = t | character states, tree topology) for t in [0, T]
    where T is the discretization_level.

    Note that this method is only fast enough for small trees. It's
    run time scales exponentially with the number of internal nodes of the
    tree.

    Args:
        tree: The CassiopeiaTree containing the tree topology and all
            character states.
        node: An internal node of the tree, for which to compute the
            posterior time distribution.
        mutation_rate: The mutation rate of the model.
        birth_rate: The birth rate of the model.
        sampling_probability: The sampling probability of the model.
        discretization_level: The number of timesteps used to discretize
            time. The output thus is a vector of length
            discretization_level + 1.
        epsrel: The degree of tolerance for the numerical integrals
            performed.

    Returns:
        P(node time = t | character states, tree topology) for t in [0, T]
            where T is the discretization_level.
    """
    numerical_log_joints = calc_numerical_log_joints(
        tree=tree,
        node=node,
        mutation_rate=mutation_rate,
        birth_rate=birth_rate,
        sampling_probability=sampling_probability,
        discretization_level=discretization_level,
        epsrel=epsrel,
    )
    numerical_posterior = np.exp(
        numerical_log_joints - numerical_log_joints.max()
    )
    numerical_posterior /= numerical_posterior.sum()
    return numerical_posterior.copy()


def relative_error(x: float, y: float) -> float:
    """
    Relative error between x and y.
    """
    assert x > 0 and y > 0
    return max(np.abs(y / x - 1), np.abs(x / y - 1))


class TestIIDExponentialBayesian(unittest.TestCase):
    @parameterized.expand(
        [
            ("1", 1.0, 0.8, False, 200),
            ("2", 1.0, 0.8, True, 200),
            ("3", 0.1, 5.0, False, 200),
            ("4", 0.1, 5.0, True, 200),
            ("5", 0.3, 4.0, False, 200),
            ("6", 0.3, 4.0, True, 200),
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
        numerical_log_likelihood = calc_numerical_log_likelihood(
            tree=tree,
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
        )
        re = relative_error(-model.log_likelihood, -numerical_log_likelihood)
        self.assertLessEqual(re, 0.01)

        # Test the _whole_ array of log joints P(t_v = t, X, T) against its
        # numerical computation
        numerical_log_joints = calc_numerical_log_joints(
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
        numerical_posterior = numerical_posterior_time(
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
        numerical_log_likelihood = calc_numerical_log_likelihood(
            tree=tree,
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
        )
        re = relative_error(-model.log_likelihood, -numerical_log_likelihood)
        self.assertLessEqual(re, 0.01)

        # Check that the posterior ages of the nodes are correct.
        for node in tree.internal_nodes:
            if node == tree.root:
                continue
            numerical_log_joints = calc_numerical_log_joints(
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
