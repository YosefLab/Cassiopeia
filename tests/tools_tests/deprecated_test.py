"""Tests that renamed tools functions emit DeprecationWarning and delegate."""

import numpy as np
import pandas as pd
import pytest

import cassiopeia as cas
from cassiopeia.tools import parameter_estimators, tree_metrics

from .conftest import SMALL_NET_EDGES, build_tree


def test_get_proportion_of_missing_data_delegates(discrete_tree):
    with pytest.warns(DeprecationWarning, match="get_proportion_of_missing_data"):
        result = cas.tl.get_proportion_of_missing_data(discrete_tree)
    assert result == pytest.approx(
        parameter_estimators.fraction_missing(discrete_tree, key_added=None)
    )


def test_get_proportion_of_mutation_delegates(discrete_tree):
    with pytest.warns(DeprecationWarning, match="get_proportion_of_mutation"):
        result = cas.tl.get_proportion_of_mutation(discrete_tree)
    assert result == pytest.approx(
        parameter_estimators.fraction_mutated(discrete_tree, key_added=None)
    )


def test_estimate_missing_data_rates_delegates(discrete_tree):
    with pytest.warns(DeprecationWarning, match="estimate_missing_data_rates"):
        result = cas.tl.estimate_missing_data_rates(
            discrete_tree, continuous=False, stochastic_missing_probability=0.1
        )
    expected = parameter_estimators.estimate_missing_rates(
        discrete_tree, continuous=False, stochastic_missing_probability=0.1
    )
    assert np.allclose(result, expected)


def test_get_lineage_tracing_parameters_delegates(discrete_tree):
    discrete_tree.uns["stochastic_missing_probability"] = 0.3
    with pytest.warns(DeprecationWarning, match="get_lineage_tracing_parameters"):
        result = cas.tl.get_lineage_tracing_parameters(
            discrete_tree, continuous=False, assume_root_implicit_branch=True
        )
    expected = tree_metrics.get_tracing_parameters(
        discrete_tree, continuous=False, assume_root_implicit_branch=True
    )
    assert np.allclose(result, expected)


def _likelihood_tree():
    cm = pd.DataFrame.from_dict(
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
    # Non-integer branch lengths so the continuous model does not warn about
    # discrete branches during parameter estimation.
    branch_lengths = dict.fromkeys(SMALL_NET_EDGES, 0.5)
    tdata = build_tree(SMALL_NET_EDGES, cm, priors=priors, branch_lengths=branch_lengths)
    tdata.uns["stochastic_missing_probability"] = 0.3
    return tdata


def test_calculate_likelihood_discrete_delegates():
    with pytest.warns(DeprecationWarning, match="calculate_likelihood_discrete"):
        result = cas.tl.calculate_likelihood_discrete(_likelihood_tree())
    expected = tree_metrics.calculate_likelihood(_likelihood_tree(), model="discrete")
    assert np.isclose(result, expected)


def test_calculate_likelihood_continuous_delegates():
    with pytest.warns(DeprecationWarning, match="calculate_likelihood_continuous"):
        result = cas.tl.calculate_likelihood_continuous(_likelihood_tree())
    expected = tree_metrics.calculate_likelihood(_likelihood_tree(), model="continuous")
    assert np.isclose(result, expected)
