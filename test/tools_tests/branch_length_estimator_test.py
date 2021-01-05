import networkx as nx
import numpy as np
import pytest

from cassiopeia.tools import (
    IIDExponentialBLE,
    IIDExponentialBLEGridSearchCV,
    IIDExponentialLineageTracer,
    IIDExponentialPosteriorMeanBLE,
    IIDExponentialPosteriorMeanBLEGridSearchCV,
    Tree,
)


def test_no_mutations():
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
    tree.add_node(0), tree.add_node(1)
    tree.add_edge(0, 1)
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "0"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(tree.get_edge_length(0, 1), 0.0)
    np.testing.assert_almost_equal(tree.get_age(0), 0.0)
    np.testing.assert_almost_equal(tree.get_age(1), 0.0)
    np.testing.assert_almost_equal(log_likelihood, 0.0)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_saturation():
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
    tree.add_nodes_from([0, 1])
    tree.add_edge(0, 1)
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "1"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    assert tree.get_edge_length(0, 1) > 15.0
    assert tree.get_age(0) > 15.0
    np.testing.assert_almost_equal(tree.get_age(1), 0.0)
    np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=5)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_hand_solvable_problem_1():
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
    tree.add_nodes_from([0, 1])
    tree.add_edge(0, 1)
    tree.nodes[0]["characters"] = "00"
    tree.nodes[1]["characters"] = "01"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 1), np.log(2), decimal=3
    )
    np.testing.assert_almost_equal(tree.get_age(0), np.log(2), decimal=3)
    np.testing.assert_almost_equal(tree.get_age(1), 0.0)
    np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_hand_solvable_problem_2():
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
    tree.add_nodes_from([0, 1])
    tree.add_edge(0, 1)
    tree.nodes[0]["characters"] = "000"
    tree.nodes[1]["characters"] = "011"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 1), np.log(3), decimal=3
    )
    np.testing.assert_almost_equal(tree.get_age(0), np.log(3), decimal=3)
    np.testing.assert_almost_equal(tree.get_age(1), 0.0)
    np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_hand_solvable_problem_3():
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
    tree.add_nodes_from([0, 1])
    tree.add_edge(0, 1)
    tree.nodes[0]["characters"] = "000"
    tree.nodes[1]["characters"] = "001"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 1), np.log(1.5), decimal=3
    )
    np.testing.assert_almost_equal(tree.get_age(0), np.log(1.5), decimal=3)
    np.testing.assert_almost_equal(tree.get_age(1), 0.0)
    np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_tree_with_no_mutations():
    r"""
    Perfect binary tree with no mutations: Should give edges of length 0
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "0000"
    tree.nodes[1]["characters"] = "0000"
    tree.nodes[2]["characters"] = "0000"
    tree.nodes[3]["characters"] = "0000"
    tree.nodes[4]["characters"] = "0000"
    tree.nodes[5]["characters"] = "0000"
    tree.nodes[6]["characters"] = "0000"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    for edge in tree.edges():
        np.testing.assert_almost_equal(
            tree.get_edge_length(*edge), 0, decimal=3
        )
    np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_tree_with_one_mutation():
    r"""
    Perfect binary tree with one mutation at a node 6: Should give very short
    edges 1->3,1->4,0->2 and very long edges 0->1,2->5,2->6.
    The problem can be solved by hand: it trivially reduces to a 1-dimensional
    problem:
        min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
    The solution is r * t0 = ln(1.5) ~ 0.405
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "0"
    tree.nodes[2]["characters"] = "0"
    tree.nodes[3]["characters"] = "0"
    tree.nodes[4]["characters"] = "0"
    tree.nodes[5]["characters"] = "0"
    tree.nodes[6]["characters"] = "1"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(tree.get_edge_length(0, 1), 0.405, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(0, 2), 0.0, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(1, 3), 0.0, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(1, 4), 0.0, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(2, 5), 0.405, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(2, 6), 0.405, decimal=3)
    np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_tree_with_saturation():
    r"""
    Perfect binary tree with saturation. The edges which saturate should thus
    have length infinity (>15 for all practical purposes)
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "0"
    tree.nodes[2]["characters"] = "1"
    tree.nodes[3]["characters"] = "1"
    tree.nodes[4]["characters"] = "1"
    tree.nodes[5]["characters"] = "1"
    tree.nodes[6]["characters"] = "1"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    assert tree.get_edge_length(0, 2) > 15.0
    assert tree.get_edge_length(1, 3) > 15.0
    assert tree.get_edge_length(1, 4) > 15.0
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_tree_regression():
    r"""
    Regression test. Cannot be solved by hand. We just check that this solution
    never changes.
    """
    # Perfect binary tree with normal amount of mutations on each edge
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "000000000"
    tree.nodes[1]["characters"] = "100000000"
    tree.nodes[2]["characters"] = "000006000"
    tree.nodes[3]["characters"] = "120000000"
    tree.nodes[4]["characters"] = "103000000"
    tree.nodes[5]["characters"] = "000056700"
    tree.nodes[6]["characters"] = "000406089"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(tree.get_edge_length(0, 1), 0.203, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(0, 2), 0.082, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(1, 3), 0.175, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(1, 4), 0.175, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(2, 5), 0.295, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(2, 6), 0.295, decimal=3)
    np.testing.assert_almost_equal(log_likelihood, -22.689, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_symmetric_tree():
    r"""
    Symmetric tree should have equal length edges.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "000"
    tree.nodes[1]["characters"] = "100"
    tree.nodes[2]["characters"] = "100"
    tree.nodes[3]["characters"] = "110"
    tree.nodes[4]["characters"] = "110"
    tree.nodes[5]["characters"] = "110"
    tree.nodes[6]["characters"] = "110"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 1), tree.get_edge_length(0, 2)
    )
    np.testing.assert_almost_equal(
        tree.get_edge_length(1, 3), tree.get_edge_length(1, 4)
    )
    np.testing.assert_almost_equal(
        tree.get_edge_length(1, 4), tree.get_edge_length(2, 5)
    )
    np.testing.assert_almost_equal(
        tree.get_edge_length(2, 5), tree.get_edge_length(2, 6)
    )
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_small_tree_with_infinite_legs():
    r"""
    Perfect binary tree with saturated leaves. The first level of the tree
    should be normal (can be solved by hand, solution is log(2)),
    the branches for the leaves should be infinity (>15 for all practical
    purposes)
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "00"
    tree.nodes[1]["characters"] = "10"
    tree.nodes[2]["characters"] = "10"
    tree.nodes[3]["characters"] = "11"
    tree.nodes[4]["characters"] = "11"
    tree.nodes[5]["characters"] = "11"
    tree.nodes[6]["characters"] = "11"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(tree.get_edge_length(0, 1), 0.693, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(0, 2), 0.693, decimal=3)
    assert tree.get_edge_length(1, 3) > 15
    assert tree.get_edge_length(1, 4) > 15
    assert tree.get_edge_length(2, 5) > 15
    assert tree.get_edge_length(2, 6) > 15
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_on_simulated_data():
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["age"] = 1
    tree.nodes[1]["age"] = 0.9
    tree.nodes[2]["age"] = 0.1
    tree.nodes[3]["age"] = 0
    tree.nodes[4]["age"] = 0
    tree.nodes[5]["age"] = 0
    tree.nodes[6]["age"] = 0
    np.random.seed(1)
    tree = Tree(tree)
    IIDExponentialLineageTracer(
        mutation_rate=1.0, num_characters=100
    ).overlay_lineage_tracing_data(tree)
    for node in tree.nodes():
        tree.set_age(node, -1)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    assert 0.9 < tree.get_age(0) < 1.1
    assert 0.8 < tree.get_age(1) < 1.0
    assert 0.05 < tree.get_age(2) < 0.15
    np.testing.assert_almost_equal(tree.get_age(3), 0)
    np.testing.assert_almost_equal(tree.get_age(4), 0)
    np.testing.assert_almost_equal(tree.get_age(5), 0)
    np.testing.assert_almost_equal(tree.get_age(6), 0)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_subtree_collapses_when_no_mutations():
    r"""
    A subtree with no mutations should collapse to 0. It reduces the problem to
    the same as in 'test_hand_solvable_problem_1'
    """
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4]),
    tree.add_edges_from([(0, 1), (1, 2), (1, 3), (0, 4)])
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "1"
    tree.nodes[2]["characters"] = "1"
    tree.nodes[3]["characters"] = "1"
    tree.nodes[4]["characters"] = "0"
    tree = Tree(tree)
    model = IIDExponentialBLE()
    model.estimate_branch_lengths(tree)
    log_likelihood = model.log_likelihood
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 1), np.log(2), decimal=3
    )
    np.testing.assert_almost_equal(tree.get_edge_length(1, 2), 0.0, decimal=3)
    np.testing.assert_almost_equal(tree.get_edge_length(1, 3), 0.0, decimal=3)
    np.testing.assert_almost_equal(
        tree.get_edge_length(0, 4), np.log(2), decimal=3
    )
    np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
    log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
    np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)


def test_IIDExponentialBLEGridSearchCV():
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1]),
    tree.add_edges_from([(0, 1)])
    tree.nodes[0]["characters"] = "000"
    tree.nodes[1]["characters"] = "001"
    tree = Tree(tree)
    model = IIDExponentialBLEGridSearchCV(
        minimum_branch_lengths=(0, 1.0, 3.0),
        l2_regularizations=(0,),
        verbose=True,
    )
    model.estimate_branch_lengths(tree)
    minimum_branch_length = model.minimum_branch_length
    np.testing.assert_almost_equal(minimum_branch_length, 1.0)


def test_IIDExponentialPosteriorMeanBLE():
    r"""
    TODO
    """
    from scipy.special import binom
    from scipy.special import logsumexp

    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3])
    tree.add_edges_from([(0, 1), (1, 2), (1, 3)])
    tree.nodes[0]["characters"] = "000000000"
    tree.nodes[1]["characters"] = "010000110"
    tree.nodes[2]["characters"] = "010110111"
    tree.nodes[3]["characters"] = "011100111"
    tree = Tree(tree)

    mutation_rate = 0.3
    birth_rate = 0.8
    discretization_level = 200
    model = IIDExponentialPosteriorMeanBLE(
        mutation_rate=mutation_rate,
        birth_rate=birth_rate,
        discretization_level=discretization_level,
    )

    def cuts(parent, child):
        zeros_parent = tree.get_state(parent).count("0")
        zeros_child = tree.get_state(child).count("0")
        new_cuts_child = zeros_parent - zeros_child
        return new_cuts_child

    def uncuts(parent, child):
        zeros_child = tree.get_state(child).count("0")
        return zeros_child

    def analytical_log_joint(t):
        r"""
        when node 1 has age t, i.e. hangs at distance 1.0 - t from the root.
        """
        t = 1.0 - t
        if t == 0 or t == 1:
            return -np.inf
        e = np.exp
        lg = np.log
        lam = birth_rate
        r = mutation_rate
        res = 0.0
        res += (
            lg(lam) + -t * lam + -2 * (1.0 - t) * lam
        )  # Tree topology likelihood
        res += -t * r * uncuts(0, 1) + lg(1.0 - e(-t * r)) * cuts(
            0, 1
        )  # 0->1 edge likelihood
        res += -(1.0 - t) * r * uncuts(1, 2) + lg(
            1.0 - e(-(1.0 - t) * r)
        ) * cuts(
            1, 2
        )  # 1->2 edge likelihood
        res += -(1.0 - t) * r * uncuts(1, 3) + lg(
            1.0 - e(-(1.0 - t) * r)
        ) * cuts(
            1, 3
        )  # 1->3 edge likelihood
        # Adjust by the grid size so we don't overestimate the bucket's
        # probability.
        res -= np.log(discretization_level)
        # Finally, we need to account for repetitions
        res += (
            np.log(binom(cuts(0, 1) + uncuts(0, 1), cuts(0, 1)))
            + np.log(binom(cuts(1, 2) + uncuts(1, 2), cuts(1, 2)))
            + np.log(binom(cuts(1, 3) + uncuts(1, 3), cuts(1, 3)))
        )
        return res

    model.estimate_branch_lengths(tree)
    print(f"{model.log_likelihood} = model.log_likelihood")

    # Test the model log likelihood vs its computation from the joint of the
    # age of vertex 1.
    model_log_joints = model.log_joints[
        1
    ]  # P(t_1 = t, X, T) where t_1 is the age of the first node.
    model_log_likelihood_2 = logsumexp(model_log_joints)
    print(f"{model_log_likelihood_2} = {model_log_likelihood_2}")
    np.testing.assert_approx_equal(
        model.log_likelihood, model_log_likelihood_2, significant=3
    )

    # Test the model log likelihood vs its computation from a leaf node.
    leaf = 2
    model_log_likelihood_up = model.up(leaf, 0, tree.num_cuts(leaf))
    print(f"{model_log_likelihood_up} = model_log_likelihood_up")
    np.testing.assert_approx_equal(
        model.log_likelihood, model_log_likelihood_up, significant=3
    )

    # Test the model log likelihood against its analytic computation
    analytical_log_joints = np.array(
        [
            analytical_log_joint(t)
            for t in np.array(range(discretization_level + 1))
            / discretization_level
        ]
    )
    analytical_log_likelihood = logsumexp(analytical_log_joints)
    print(f"{analytical_log_likelihood} = analytical_log_likelihood")
    np.testing.assert_approx_equal(
        model.log_likelihood, analytical_log_likelihood, significant=3
    )

    np.testing.assert_array_almost_equal(
        analytical_log_joints[50:150], model.log_joints[1][50:150], decimal=1
    )

    # import matplotlib.pyplot as plt

    # plt.plot(model.posteriors[1])
    # plt.show()
    # print(model.posterior_means[1])

    # Analytical posterior
    analytical_posterior = np.exp(
        analytical_log_joints - analytical_log_joints.max()
    )
    analytical_posterior /= analytical_posterior.sum()
    # plt.plot(analytical_posterior)
    # plt.show()
    total_variation = np.sum(np.abs(analytical_posterior - model.posteriors[1]))
    assert total_variation < 0.03


def test_IIDExponentialPosteriorMeanBLE_2():
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6]),
    tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    tree.nodes[0]["characters"] = "00"
    tree.nodes[1]["characters"] = "00"
    tree.nodes[2]["characters"] = "10"
    tree.nodes[3]["characters"] = "00"
    tree.nodes[4]["characters"] = "01"
    tree.nodes[5]["characters"] = "10"
    tree.nodes[6]["characters"] = "11"
    tree = Tree(tree)

    mutation_rate = 0.625
    birth_rate = 0.75
    discretization_level = 100
    model = IIDExponentialPosteriorMeanBLE(
        mutation_rate=mutation_rate,
        birth_rate=birth_rate,
        discretization_level=discretization_level,
    )

    model.estimate_branch_lengths(tree)
    print(model.log_likelihood)

    for leaf in tree.leaves():
        model_log_likelihood_up = model.up(leaf, 0, tree.num_cuts(leaf))
        print(model_log_likelihood_up)
        np.testing.assert_approx_equal(
            model.log_likelihood, model_log_likelihood_up, significant=3
        )

        model_log_likelihood_up_wrong = model.up(
            leaf, 0, (tree.num_cuts(leaf) + 1) % 2
        )
        with pytest.raises(AssertionError):
            np.testing.assert_approx_equal(
                model.log_likelihood,
                model_log_likelihood_up_wrong,
                significant=3,
            )

    # import matplotlib.pyplot as plt

    # plt.plot(model.posteriors[1])
    # plt.show()
    # print(model.posterior_means[1])


def test_IIDExponentialPosteriorMeanBLEGridSeachCV():
    # This is same tree as test_subtree_collapses_when_no_mutations. Should no
    # longer collapse!
    tree = nx.DiGraph()
    tree.add_nodes_from([0, 1, 2, 3, 4]),
    tree.add_edges_from([(0, 1), (1, 2), (1, 3), (0, 4)])
    tree.nodes[0]["characters"] = "0"
    tree.nodes[1]["characters"] = "1"
    tree.nodes[2]["characters"] = "1"
    tree.nodes[3]["characters"] = "1"
    tree.nodes[4]["characters"] = "0"
    tree = Tree(tree)

    discretization_level = 100
    mutation_rates = (0.625, 0.750, 0.875)
    birth_rates = (0.25, 0.50, 0.75)
    model = IIDExponentialPosteriorMeanBLEGridSearchCV(
        mutation_rates=mutation_rates,
        birth_rates=birth_rates,
        discretization_level=discretization_level,
        verbose=True,
    )

    model.estimate_branch_lengths(tree)

    # import seaborn as sns

    # import matplotlib.pyplot as plt

    # sns.heatmap(
    #     model.grid,
    #     yticklabels=mutation_rates,
    #     xticklabels=birth_rates
    # )
    # plt.ylabel('Mutation Rate')
    # plt.xlabel('Birth Rate')
    # plt.show()

    # import matplotlib.pyplot as plt
    # plt.plot(model.posteriors[1])
    # plt.show()
    # print(model.posterior_means[1])

    np.testing.assert_almost_equal(model.posterior_means[1], 0.3184, decimal=3)
    np.testing.assert_almost_equal(model.mutation_rate, 0.75)
    np.testing.assert_almost_equal(model.birth_rate, 0.5)
