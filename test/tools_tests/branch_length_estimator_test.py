import itertools
import multiprocessing
import unittest
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import (BirthProcess, BLEMultifurcationWrapper,
                              IIDExponentialBLE,
                              IIDExponentialBLEGridSearchCV,
                              IIDExponentialLineageTracer,
                              IIDExponentialPosteriorMeanBLE,
                              IIDExponentialPosteriorMeanBLEGridSearchCV)


class TestIIDExponentialBLE(unittest.TestCase):
    def test_no_mutations(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one unmutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '0']
        This is thus the simplest possible example of no mutations, and the MLE
        branch length should be 0
        """
        tree = nx.DiGraph()
        tree.add_node("0"), tree.add_node("1")
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.0)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(tree.get_time("1"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, 0.0)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_saturation(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one mutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '1']
        This is thus the simplest possible example of saturation, and the MLE
        branch length should be infinity (>15 for all practical purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert tree.get_branch_length("0", "1") > 15.0
        assert tree.get_time("1") > 15.0
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=5)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_1(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one mutated character and one unmutated character, i.e.:
            root [state = '00']
            |
            v
            child [state = '01']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(2) ~ 0.693
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0],
            "1": [0, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_2(self):
        r"""
        Tree topology is just a branch 0->1.
        There are two mutated characters and one unmutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '011']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + 2 * log(1 - exp(-r * t0))
        The solution is r * t0 = ln(3) ~ 1.098
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 1, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(3), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(3), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_3(self):
        r"""
        Tree topology is just a branch 0->1.
        There are two unmutated characters and one mutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '001']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(1.5), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(1.5), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_no_mutations(self):
        r"""
        Perfect binary tree with no mutations: Should give edges of length 0
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0, 0, 0],
            "1": [0, 0, 0, 0],
            "2": [0, 0, 0, 0],
            "3": [0, 0, 0, 0],
            "4": [0, 0, 0, 0],
            "5": [0, 0, 0, 0],
            "6": [0, 0, 0, 0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        for edge in tree.edges:
            np.testing.assert_almost_equal(
                tree.get_branch_length(*edge), 0, decimal=3
            )
        np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_one_mutation(self):
        r"""
        Perfect binary tree with one mutation at a node 6: Should give very short
        edges 1->3,1->4,0->2 and very long edges 0->1,2->5,2->6.
        The problem can be solved by hand: it trivially reduces to a 1-dimensional
        problem:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [0],
            "2": [0],
            "3": [0],
            "4": [0],
            "5": [0],
            "6": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.405, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 0.405, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 0.405, decimal=3)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_saturation(self):
        r"""
        Perfect binary tree with saturation. The edges which saturate should thus
        have length infinity (>15 for all practical purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [0],
            "2": [1],
            "3": [1],
            "4": [1],
            "5": [1],
            "6": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert tree.get_branch_length("0", "2") > 15.0
        assert tree.get_branch_length("1", "3") > 15.0
        assert tree.get_branch_length("1", "4") > 15.0
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_regression(self):
        r"""
        Regression test. Cannot be solved by hand. We just check that this solution
        never changes.
        """
        # Perfect binary tree with normal amount of mutations on each edge
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "2": [0, 0, 0, 0, 0, 6, 0, 0, 0],
            "3": [1, 2, 0, 0, 0, 0, 0, 0, 0],
            "4": [1, 0, 3, 0, 0, 0, 0, 0, 0],
            "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],
            "6": [0, 0, 0, 4, 0, 6, 0, 8, 9]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.203, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.082, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.175, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 0.175, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 0.295, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 0.295, decimal=3)
        np.testing.assert_almost_equal(log_likelihood, -22.689, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_symmetric_tree(self):
        r"""
        Symmetric tree should have equal length edges.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0, 0],
            "1": [1, 0, 0],
            "2": [1, 0, 0],
            "3": [1, 1, 0],
            "4": [1, 1, 0],
            "5": [1, 1, 0],
            "6": [1, 1, 0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), tree.get_branch_length("0", "2")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("1", "3"), tree.get_branch_length("1", "4")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("1", "4"), tree.get_branch_length("2", "5")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("2", "5"), tree.get_branch_length("2", "6")
        )
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_infinite_legs(self):
        r"""
        Perfect binary tree with saturated leaves. The first level of the tree
        should be normal (can be solved by hand, solution is log(2)),
        the branches for the leaves should be infinity (>15 for all practical
        purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0],
            "1": [1, 0],
            "2": [1, 0],
            "3": [1, 1],
            "4": [1, 1],
            "5": [1, 1],
            "6": [1, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.693, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.693, decimal=3)
        assert tree.get_branch_length("1", "3") > 15
        assert tree.get_branch_length("1", "4") > 15
        assert tree.get_branch_length("2", "5") > 15
        assert tree.get_branch_length("2", "6") > 15
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_on_simulated_data(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {"0": 0,
            "1": 0.1,
            "2": 0.9,
            "3": 1.0,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0}
        )
        np.random.seed(1)
        IIDExponentialLineageTracer(
            mutation_rate=1.0, num_characters=100
        ).overlay_lineage_tracing_data(tree)
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert 0.05 < tree.get_time("1") < 0.15
        assert 0.8 < tree.get_time("2") < 1.0
        assert 0.9 < tree.get_time("3") < 1.1
        assert 0.9 < tree.get_time("4") < 1.1
        assert 0.9 < tree.get_time("5") < 1.1
        assert 0.9 < tree.get_time("6") < 1.1
        np.testing.assert_almost_equal(tree.get_time("0"), 0)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_subtree_collapses_when_no_mutations(self):
        r"""
        A subtree with no mutations should collapse to 0. It reduces the problem to
        the same as in 'test_hand_solvable_problem_1'
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [1],
            "2": [1],
            "3": [1],
            "4": [0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_branch_length("1", "2"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.0, decimal=3)
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "4"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_minimum_branch_length(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("0", "3"), ("2", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_character_states_at_leaves(
            {"1": [1],
            "3": [1],
            "4": [1]}
        )
        tree.reconstruct_ancestral_characters(zero_the_root=True)
        # Too large a minimum_branch_length
        model = IIDExponentialBLE(minimum_branch_length=0.6)
        model.estimate_branch_lengths(tree)
        for node in tree.nodes:
            print(f"{node} = {tree.get_time(node)}")
        assert model.log_likelihood == -np.inf
        # An okay minimum_branch_length
        model = IIDExponentialBLE(minimum_branch_length=0.4)
        model.estimate_branch_lengths(tree)
        assert model.log_likelihood != -np.inf


class TestIIDExponentialBLEGridSearchCV(unittest.TestCase):
    def test_IIDExponentialBLEGridSearchCV_smoke(self):
        r"""
        Just want to see that it runs in both single and multiprocessor mode
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"]),
        tree.add_edges_from([("0", "1")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [1]},
        )
        for processes in [1, 2]:
            model = IIDExponentialBLEGridSearchCV(
                minimum_branch_lengths=(1.0,),
                l2_regularizations=(1.0,),
                verbose=True,
                processes=processes,
            )
            model.estimate_branch_lengths(tree)

    def test_IIDExponentialBLEGridSearchCV(self):
        r"""
        We make sure to test a tree for which no regularization produces
        a likelihood of 0.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7"]),
        tree.add_edges_from(
            [("0", "1"), ("1", "2"), ("1", "3"), ("2", "4"), ("2", "5"), ("3", "6"),
            ("3", "7")]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_character_states_at_leaves(
            {"4": [1, 1, 0],
            "5": [1, 1, 0],
            "6": [1, 0, 0],
            "7": [1, 0, 0]},
        )
        tree.reconstruct_ancestral_characters(zero_the_root=True)
        model = IIDExponentialBLEGridSearchCV(
            minimum_branch_lengths=(0, 0.2, 4.0),
            l2_regularizations=(0.0, 2.0, 4.0),
            verbose=True,
            processes=6,
        )
        model.estimate_branch_lengths(tree)
        print(model.grid)
        assert model.grid[0, 0] == -np.inf

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(
        #     model.grid,
        #     yticklabels=model.minimum_branch_lengths,
        #     xticklabels=model.l2_regularizations,
        #     mask=np.isneginf(model.grid),
        # )
        # plt.ylabel("minimum_branch_length")
        # plt.xlabel("l2_regularization")
        # plt.show()

        np.testing.assert_almost_equal(model.minimum_branch_length, 0.2)
        np.testing.assert_almost_equal(model.l2_regularization, 2.0)


def get_z_scores(
    repetition,
    birth_rate_true,
    mutation_rate_true,
    birth_rate_model,
    mutation_rate_model,
    num_characters,
):
    r"""
    This function is at the global scope because it needs to be pickled
    for parallelization.
    """
    np.random.seed(repetition)
    tree = BirthProcess(
        birth_rate=birth_rate_true, tree_depth=1.0
    ).simulate_lineage()
    tree_true = deepcopy(tree)
    IIDExponentialLineageTracer(
        mutation_rate=mutation_rate_true, num_characters=num_characters
    ).overlay_lineage_tracing_data(tree)
    discretization_level = 100
    model = IIDExponentialPosteriorMeanBLE(
        birth_rate=birth_rate_model,
        mutation_rate=mutation_rate_model,
        discretization_level=discretization_level,
        use_cpp_implementation=True
    )
    model.estimate_branch_lengths(tree)
    z_scores = []
    if len(tree.non_root_internal_nodes) > 0:
        for node in [np.random.choice(tree.non_root_internal_nodes)]:
            true_age = tree_true.get_time(node)
            z_score = model.posteriors[node][
                : int(true_age * discretization_level)
            ].sum()
            z_scores.append(z_score)
    return z_scores


def get_z_scores_under_true_model(repetition):
    r"""
    This function is at the global scope because it needs to be pickled
    for parallelization.
    """
    return get_z_scores(
        repetition,
        birth_rate_true=0.8,
        mutation_rate_true=1.2,
        birth_rate_model=0.8,
        mutation_rate_model=1.2,
        num_characters=3,
    )


def get_z_scores_under_misspecified_model(repetition):
    r"""
    This function is at the global scope because it needs to be pickled
    for parallelization.
    """
    return get_z_scores(
        repetition,
        birth_rate_true=0.4,
        mutation_rate_true=0.6,
        birth_rate_model=0.8,
        mutation_rate_model=1.2,
        num_characters=3,
    )


class TestIIDExponentialPosteriorMeanBLE(unittest.TestCase):
    @parameterized.expand(
        [
            ("cpp0", True, 1.0, 0.8, False, 500, 3),
            ("cpp1", True, 1.0, 0.8, True, 500, 3),
            ("cpp2", True, 0.1, 5.0, False, 500, 2),
            ("cpp3", True, 0.1, 5.0, True, 500, 3),
            ("cpp4", True, 0.3, 4.0, False, 500, 3),
            ("cpp5", True, 0.3, 4.0, True, 500, 3),
            ("no_cpp0", False, 1.0, 0.8, False, 500, 3),
            ("no_cpp1", False, 1.0, 0.8, True, 500, 3),
            ("no_cpp2", False, 0.1, 5.0, False, 500, 2),
            ("no_cpp3", False, 0.1, 5.0, True, 500, 3),
            ("no_cpp4", False, 0.3, 4.0, False, 500, 3),
            ("no_cpp5", False, 0.3, 4.0, True, 500, 3),
        ]
    )
    def test_IIDExponentialPosteriorMeanBLE(
            self,
            name,
            use_cpp_implementation,
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
        from scipy.special import logsumexp

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
        tree = CassiopeiaTree(tree=tree)
        if many_characters:
            tree.initialize_all_character_states(
                {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 "1": [0, 1, 0, 0, 0, 0, 1, 1, 0],
                 "2": [0, 1, 0, 1, 1, 0, 1, 1, 1],
                 "3": [0, 1, 1, 1, 0, 0, 1, 1, 1]},
            )
        else:
            tree.initialize_all_character_states(
                {"0": [0],
                 "1": [1],
                 "2": [1],
                 "3": [1]},
            )

        mutation_rate = 0.3
        model = IIDExponentialPosteriorMeanBLE(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            use_cpp_implementation=use_cpp_implementation
        )

        model.estimate_branch_lengths(tree)
        print(f"{model.log_likelihood} = model.log_likelihood")

        # Test the model log likelihood vs its computation from the joint of the
        # age of vertex 1.
        model_log_joints = model.log_joints[
            "1"
        ]  # log P(t_1 = t, X, T) where t_1 is the age of the first node.
        model_log_likelihood_2 = logsumexp(model_log_joints)
        print(f"{model_log_likelihood_2} = {model_log_likelihood_2}")
        np.testing.assert_approx_equal(
            model.log_likelihood, model_log_likelihood_2, significant=significance
        )

        # Test the model log likelihood vs its computation from the leaf nodes.
        for leaf in ["2", "3"]:
            model_log_likelihood_up = model.up(
                leaf, discretization_level, tree.get_number_of_mutated_characters_in_node(leaf)
            ) - np.log(birth_rate * 1.0 / discretization_level)\
                + np.log(sampling_probability)
            print(f"{model_log_likelihood_up} = model_log_likelihood_up")
            np.testing.assert_approx_equal(
                model.log_likelihood, model_log_likelihood_up, significant=significance
            )

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialPosteriorMeanBLE.numerical_log_likelihood(
                tree=tree, mutation_rate=mutation_rate, birth_rate=birth_rate,
                sampling_probability=sampling_probability
            )
        )
        print(f"{numerical_log_likelihood} = numerical_log_likelihood")
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # Test the _whole_ array of log joints P(t_v = t, X, T) against its
        # numerical computation
        numerical_log_joint = IIDExponentialPosteriorMeanBLE.numerical_log_joint(
            tree=tree,
            node="1",
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        np.testing.assert_array_almost_equal(
            model.log_joints["1"][50:-50], numerical_log_joint[50:-50], decimal=1
        )

        # Test the model posterior against its numerical posterior
        numerical_posterior = IIDExponentialPosteriorMeanBLE.numerical_posterior(
            tree=tree,
            node="1",
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        # import matplotlib.pyplot as plt
        # plt.plot(model.posteriors["1"])
        # plt.show()
        # plt.plot(numerical_posterior)
        # plt.show()
        total_variation = np.sum(np.abs(model.posteriors["1"] - numerical_posterior))
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

    @parameterized.expand([("cpp", True), ("no_cpp", False)])
    def test_IIDExponentialPosteriorMeanBLE_2(self, name, use_cpp_implementation):
        r"""
        We run the Bayesian estimator on a small tree with all different leaves,
        and then check that:
        - The likelihood of the data, computed from all of the leaves, is the same.
        - The posteriors of the internal node ages matches their numerical
            counterpart.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
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
        model = IIDExponentialPosteriorMeanBLE(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            use_cpp_implementation=use_cpp_implementation
        )

        model.estimate_branch_lengths(tree)
        print(model.log_likelihood)

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialPosteriorMeanBLE.numerical_log_likelihood(
                tree=tree, mutation_rate=mutation_rate, birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # Check that the likelihood computed from each leaf node is correct.
        for leaf in tree.leaves:
            model_log_likelihood_up = model.up(
                leaf, discretization_level, tree.get_number_of_mutated_characters_in_node(leaf)
            ) - np.log(birth_rate * 1.0 / discretization_level)\
                + np.log(sampling_probability)
            print(model_log_likelihood_up)
            np.testing.assert_approx_equal(
                model.log_likelihood, model_log_likelihood_up, significant=3
            )

            model_log_likelihood_up_wrong = model.up(
                leaf, discretization_level, (tree.get_number_of_mutated_characters_in_node(leaf) + 1) % 2
            )
            with pytest.raises(AssertionError):
                np.testing.assert_approx_equal(
                    model.log_likelihood,
                    model_log_likelihood_up_wrong,
                    significant=3,
                )

        # Check that the posterior ages of the nodes are correct.
        for node in tree.non_root_internal_nodes:
            numerical_log_joint = (
                IIDExponentialPosteriorMeanBLE.numerical_log_joint(
                    tree=tree,
                    node=node,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                    discretization_level=discretization_level,
                )
            )
            np.testing.assert_array_almost_equal(
                model.log_joints[node][25:-25],
                numerical_log_joint[25:-25],
                decimal=1,
            )

            # Test the model posterior against its numerical posterior.
            numerical_posterior = np.exp(
                numerical_log_joint - numerical_log_joint.max()
            )
            numerical_posterior /= numerical_posterior.sum()
            # import matplotlib.pyplot as plt
            # plt.plot(model.posteriors[node])
            # plt.show()
            # plt.plot(analytical_posterior)
            # plt.show()
            total_variation = np.sum(
                np.abs(model.posteriors[node] - numerical_posterior)
            )
            assert total_variation < 0.03

    @parameterized.expand([("cpp", True), ("no_cpp", False)])
    @pytest.mark.slow
    def test_IIDExponentialPosteriorMeanBLE_3(self, name, use_cpp_implementation):
        r"""
        Same as test_IIDExponentialPosteriorMeanBLE_2 but with a weirder topology.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7"]),
        tree.add_edges_from(
            [("0", "1"), ("1", "2"), ("1", "3"), ("2", "4"), ("2", "5"), ("2", "6"),
            ("0", "7")]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0],
            "1": [0, 0],
            "2": [1, 0],
            "3": [1, 1],
            "4": [1, 0],
            "5": [1, 0],
            "6": [1, 1],
            "7": [0, 0]},
        )

        mutation_rate = 0.625
        birth_rate = 0.75
        sampling_probability = 0.1
        discretization_level = 100
        model = IIDExponentialPosteriorMeanBLE(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            use_cpp_implementation=use_cpp_implementation
        )

        model.estimate_branch_lengths(tree)
        print(model.log_likelihood)

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialPosteriorMeanBLE.numerical_log_likelihood(
                tree=tree, mutation_rate=mutation_rate, birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # Check that the likelihood computed from each leaf node is correct.
        for leaf in tree.leaves:
            model_log_likelihood_up = model.up(
                leaf, discretization_level, tree.get_number_of_mutated_characters_in_node(leaf)
            ) - np.log(birth_rate * 1.0 / discretization_level)\
                + np.log(sampling_probability)
            print(model_log_likelihood_up)
            np.testing.assert_approx_equal(
                model.log_likelihood, model_log_likelihood_up, significant=2
            )

        # Check that the posterior ages of the nodes are correct.
        for node in tree.non_root_internal_nodes:
            numerical_log_joint = (
                IIDExponentialPosteriorMeanBLE.numerical_log_joint(
                    tree=tree,
                    node=node,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                    discretization_level=discretization_level,
                )
            )
            np.testing.assert_array_almost_equal(
                model.log_joints[node][25:-25],
                numerical_log_joint[25:-25],
                decimal=1,
            )

            # Test the model posterior against its numerical posterior.
            numerical_posterior = np.exp(
                numerical_log_joint - numerical_log_joint.max()
            )
            numerical_posterior /= numerical_posterior.sum()
            # import matplotlib.pyplot as plt
            # plt.plot(model.posteriors[node])
            # plt.show()
            # plt.plot(numerical_posterior)
            # plt.show()
            total_variation = np.sum(
                np.abs(model.posteriors[node] - numerical_posterior)
            )
            assert total_variation < 0.03

    @parameterized.expand([("cpp", True), ("no_cpp", False)])
    @pytest.mark.slow
    def test_IIDExponentialPosteriorMeanBLE_DREAM_subC1(self, name, use_cpp_implementation):
        r"""
        A tree from the DREAM subchallenge 1, verified analytically.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_character_states_at_leaves(
            {"3": [2, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            "4": [2, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            "5": [2, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            "6": [2, 0, 1, 1, 0, 1, 0, 1, 1, 1]},
        )
        tree.reconstruct_ancestral_characters(zero_the_root=True)

        mutation_rate = 0.6
        birth_rate = 0.8
        sampling_probability = 0.1
        discretization_level = 500
        model = IIDExponentialPosteriorMeanBLE(
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            use_cpp_implementation=use_cpp_implementation
        )

        model.estimate_branch_lengths(tree)
        print(model.log_likelihood)

        # Test the model log likelihood against its numerical computation
        numerical_log_likelihood = (
            IIDExponentialPosteriorMeanBLE.numerical_log_likelihood(
                tree=tree, mutation_rate=mutation_rate, birth_rate=birth_rate,
                sampling_probability=sampling_probability,
            )
        )
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # Check that the likelihood computed from each leaf node is correct.
        for leaf in tree.leaves:
            model_log_likelihood_up = model.up(
                leaf, discretization_level, tree.get_number_of_mutated_characters_in_node(leaf)
            ) - np.log(birth_rate * 1.0 / discretization_level)\
                + np.log(sampling_probability)
            print(model_log_likelihood_up)
            np.testing.assert_approx_equal(
                model.log_likelihood, model_log_likelihood_up, significant=3
            )

        # Check that the posterior ages of the nodes are correct.
        for node in tree.non_root_internal_nodes:
            numerical_log_joint = (
                IIDExponentialPosteriorMeanBLE.numerical_log_joint(
                    tree=tree,
                    node=node,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                    discretization_level=discretization_level,
                )
            )
            mean_error = np.mean(
                np.abs(model.log_joints[node][25:-25] - numerical_log_joint[25:-25])
                / np.abs(numerical_log_joint[25:-25])
            )
            assert mean_error < 0.03

            # Test the model posterior against its numerical posterior.
            numerical_posterior = np.exp(
                numerical_log_joint - numerical_log_joint.max()
            )
            numerical_posterior /= numerical_posterior.sum()
            # import matplotlib.pyplot as plt
            # plt.plot(model.posteriors[node])
            # plt.show()
            # plt.plot(numerical_posterior)
            # plt.show()
            total_variation = np.sum(
                np.abs(model.posteriors[node] - numerical_posterior)
            )
            assert total_variation < 0.05

    @pytest.mark.slow
    def test_IIDExponentialPosteriorMeanBLE_posterior_calibration(self):
        r"""
        Under the true model, the Z scores should be ~Unif[0, 1]
        Under the wrong model, the Z scores should not be ~Unif[0, 1]
        This test is slow because we need to make many repetitions to get
        enough statistical power for the test to be meaningful.
        We use p-values computed from the Hoeffding bound.
        TODO: There might be a more powerful test, e.g. Kolmogorov–Smirnov?
        (This would mean we need less repetitions and can make the test faster.)
        This test uses the c++ implementation to be faster.
        """
        repetitions = 1000

        # Under the true model, the Z scores should be ~Unif[0, 1]
        with multiprocessing.Pool(processes=6) as pool:
            z_scores = pool.map(get_z_scores_under_true_model, range(repetitions))
        z_scores = np.array(list(itertools.chain(*z_scores)))
        mean_z_score = z_scores.mean()
        p_value = 2 * np.exp(-2 * repetitions * (mean_z_score - 0.5) ** 2)
        print(f"p_value under true model = {p_value}")
        assert p_value > 0.01
        # import matplotlib.pyplot as plt
        # plt.hist(z_scores, bins=10)
        # plt.show()

        # Under the wrong model, the Z scores should not be ~Unif[0, 1]
        with multiprocessing.Pool(processes=6) as pool:
            z_scores = pool.map(
                get_z_scores_under_misspecified_model, range(repetitions)
            )
        z_scores = np.array(list(itertools.chain(*z_scores)))
        mean_z_score = z_scores.mean()
        p_value = 2 * np.exp(-2 * repetitions * (mean_z_score - 0.5) ** 2)
        print(f"p_value under misspecified model = {p_value}")
        assert p_value < 0.01
        # import matplotlib.pyplot as plt
        # plt.hist(z_scores, bins=10)
        # plt.show()


class TestIIDExponentialPosteriorMeanBLEGridSeachCV(unittest.TestCase):
    @parameterized.expand([("cpp", True), ("no_cpp", False)])
    def test_IIDExponentialPosteriorMeanBLEGridSeachCV_smoke(self, name, use_cpp_implementation):
        r"""
        Just want to see that it runs in both single and multiprocessor mode
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"]),
        tree.add_edges_from([("0", "1")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [1]}
        )
        for processes in [1, 2]:
            model = IIDExponentialPosteriorMeanBLEGridSearchCV(
                mutation_rates=(0.5,),
                birth_rates=(1.5,),
                discretization_level=5,
                verbose=True,
                use_cpp_implementation=use_cpp_implementation
            )
            model.estimate_branch_lengths(tree)

    @parameterized.expand([("cpp", True), ("no_cpp", False)])
    def test_IIDExponentialPosteriorMeanBLEGridSeachCV(self, name, use_cpp_implementation):
        r"""
        We just check that the grid search estimator does its job on a small grid.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0],
            "1": [1],
            "2": [1],
            "3": [1],
            "4": [0]}
        )

        discretization_level = 100
        mutation_rates = (0.625, 0.750, 0.875)
        birth_rates = (0.25, 0.50, 0.75)
        model = IIDExponentialPosteriorMeanBLEGridSearchCV(
            mutation_rates=mutation_rates,
            birth_rates=birth_rates,
            discretization_level=discretization_level,
            verbose=True,
            use_cpp_implementation=use_cpp_implementation
        )

        # Test the model log likelihood against its numerical computation
        model.estimate_branch_lengths(tree)
        numerical_log_likelihood = (
            IIDExponentialPosteriorMeanBLE.numerical_log_likelihood(
                tree=tree,
                mutation_rate=model.mutation_rate,
                birth_rate=model.birth_rate,
            )
        )
        np.testing.assert_approx_equal(
            model.log_likelihood, numerical_log_likelihood, significant=3
        )

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.heatmap(
        #     model.grid,
        #     yticklabels=mutation_rates,
        #     xticklabels=birth_rates
        # )
        # plt.ylabel('Mutation Rate')
        # plt.xlabel('Birth Rate')
        # plt.show()

        np.testing.assert_almost_equal(model.mutation_rate, 0.75)
        np.testing.assert_almost_equal(model.birth_rate, 0.5)
        np.testing.assert_almost_equal(model.posterior_means["1"], 0.6815, decimal=3)


class TestBLEMultifurcationWrapper(unittest.TestCase):
    def test_smoke(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("0", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.initialize_all_character_states(
            {"0": [0, 0],
             "1": [0, 1],
             "2": [0, 1],
             "3": [0, 1]}
        )
        model = BLEMultifurcationWrapper(IIDExponentialBLE())
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "2"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "3"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("2"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("3"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        # The tree topology that the estimator sees is different from the
        # one in the final phylogeny, thus the lik will be different!
        np.testing.assert_almost_equal(log_likelihood * 3, log_likelihood_2, decimal=3)
