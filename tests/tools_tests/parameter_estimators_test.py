"""
Tests for cassiopeia/tools/parameter_estimators.py
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cassiopeia as cas
from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning
from cassiopeia.tools import parameter_estimators


@pytest.fixture
def cassiopeia_trees():
    # Small test network
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

    # Discrete tree
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
    discrete_tree = cas.data.CassiopeiaTree(tree=small_net, character_matrix=cm1, priors=priors1)
    root_time = discrete_tree.get_time(discrete_tree.root)
    for node in discrete_tree.nodes:
        discrete_tree._CassiopeiaTree__network.nodes[node]["depth"] = (
            discrete_tree.get_time(node) - root_time
        )
    # Continuous tree
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
    continuous_tree = cas.data.CassiopeiaTree(tree=small_net, character_matrix=cm2, priors=priors2)
    continuous_tree.set_branch_length("node5", "node0", 1.5)
    continuous_tree.set_branch_length("node6", "node3", 2)
    root_time = continuous_tree.get_time(continuous_tree.root)
    for node in continuous_tree.nodes:
        continuous_tree._CassiopeiaTree__network.nodes[node]["depth"] = (
            continuous_tree.get_time(node) - root_time
        )

    return discrete_tree, continuous_tree


def test_proportions(cassiopeia_trees):
    discrete_tree, continuous_tree = cassiopeia_trees

    prop_mut = parameter_estimators.get_proportion_of_mutation(discrete_tree)
    prop_missing = parameter_estimators.get_proportion_of_missing_data(discrete_tree)
    assert prop_mut == pytest.approx(5 / 6)
    assert prop_missing == pytest.approx(0.6)

    prop_mut = parameter_estimators.get_proportion_of_mutation(continuous_tree)
    prop_missing = parameter_estimators.get_proportion_of_missing_data(continuous_tree)
    assert prop_mut == pytest.approx(7 / 8)
    assert prop_missing == pytest.approx(0.2)


def test_estimate_mutation_rate(cassiopeia_trees):
    discrete_tree, continuous_tree = cassiopeia_trees

    mut_rate = parameter_estimators.estimate_mutation_rate(
        discrete_tree, continuous=False, depth_key="depth"
    )
    assert np.isclose(mut_rate, 0.44967879185089554)

    # Test with time (continuous)
    mut_rate = parameter_estimators.estimate_mutation_rate(
        continuous_tree, continuous=True, depth_key="time"
    )
    assert np.isclose(mut_rate, 0.5917110077950752)

    # Test that using depth with continuous=True gives a warning
    with pytest.warns(UserWarning, match="continuous=True with discrete branches"):
        mut_rate = parameter_estimators.estimate_mutation_rate(
            discrete_tree, continuous=True, depth_key="depth"
        )


def test_estimate_missing_data_bad_cases(cassiopeia_trees):
    discrete_tree, continuous_tree = cassiopeia_trees

    with pytest.raises(ParameterEstimateError):
        parameter_estimators.estimate_missing_data_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateError):
        parameter_estimators.estimate_missing_data_rates(
            discrete_tree,
            continuous=False,
            heritable_missing_rate=0.25,
            stochastic_missing_probability=0.2,
        )

    with pytest.raises(ParameterEstimateError):
        discrete_tree.parameters["heritable_missing_rate"] = 0.25
        discrete_tree.parameters["stochastic_missing_probability"] = 0.2
        parameter_estimators.estimate_missing_data_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateWarning):
        discrete_tree.reset_parameters()
        discrete_tree.parameters["heritable_missing_rate"] = 0.5
        parameter_estimators.estimate_missing_data_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateWarning):
        continuous_tree.parameters["stochastic_missing_probability"] = 0.9
        parameter_estimators.estimate_missing_data_rates(continuous_tree, continuous=True)


def test_estimate_stochastic_missing_data_probability(cassiopeia_trees):
    discrete_tree, continuous_tree = cassiopeia_trees

    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False, heritable_missing_rate=0.25
    )[0]
    assert np.isclose(s_missing_prob, 0.0518518518518518)

    discrete_tree.parameters["heritable_missing_rate"] = 0.25
    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False
    )[0]
    assert np.isclose(s_missing_prob, 0.0518518518518518)

    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False, assume_root_implicit_branch=False
    )[0]
    assert np.isclose(s_missing_prob, 13 / 45)

    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True, heritable_missing_rate=0.05
    )[0]
    assert np.isclose(s_missing_prob, 0.046322071416968195)

    continuous_tree.parameters["heritable_missing_rate"] = 0.05
    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True
    )[0]
    assert np.isclose(s_missing_prob, 0.046322071416968195)

    s_missing_prob = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True, assume_root_implicit_branch=False
    )[0]
    assert np.isclose(s_missing_prob, 0.10250124994244929)


def test_estimate_heritable_missing_data_rate(cassiopeia_trees):
    discrete_tree, continuous_tree = cassiopeia_trees

    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False, stochastic_missing_probability=0.12
    )[1]
    assert np.isclose(h_missing_rate, 0.23111904017137075)

    discrete_tree.parameters["stochastic_missing_probability"] = 0.2
    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False
    )[1]
    assert np.isclose(h_missing_rate, 0.2062994740159002)

    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        discrete_tree, continuous=False, assume_root_implicit_branch=False
    )[1]
    assert np.isclose(h_missing_rate, 0.2928932188134524)

    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True, stochastic_missing_probability=0.04
    )[1]
    assert np.isclose(h_missing_rate, 0.05188011778689765)

    continuous_tree.parameters["stochastic_missing_probability"] = 0.1
    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True
    )[1]
    assert np.isclose(h_missing_rate, 0.0335154979510034)

    h_missing_rate = parameter_estimators.estimate_missing_data_rates(
        continuous_tree, continuous=True, assume_root_implicit_branch=False
    )[1]
    assert np.isclose(h_missing_rate, 0.05121001550277538)
