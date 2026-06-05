"""
Tests for cassiopeia/tools/parameter_estimators.py
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning
from cassiopeia.tools import parameter_estimators

from .conftest import build_tree


def test_proportions(discrete_tree, continuous_tree):
    prop_mut = parameter_estimators.fraction_mutated(discrete_tree)
    prop_missing = parameter_estimators.fraction_missing(discrete_tree)
    assert prop_mut == pytest.approx(5 / 6)
    assert prop_missing == pytest.approx(0.6)

    # per-cell values are written to obs under the default key_added
    assert discrete_tree.obs["fraction_missing"]["node0"] == pytest.approx(2 / 3)
    assert discrete_tree.obs["fraction_missing"]["node1"] == pytest.approx(1 / 3)
    assert discrete_tree.obs["fraction_mutated"]["node0"] == pytest.approx(0.0)
    assert discrete_tree.obs["fraction_mutated"]["node1"] == pytest.approx(1.0)

    prop_mut = parameter_estimators.fraction_mutated(continuous_tree)
    prop_missing = parameter_estimators.fraction_missing(continuous_tree)
    assert prop_mut == pytest.approx(7 / 8)
    assert prop_missing == pytest.approx(0.2)


def test_fraction_key_added_none_skips_obs(discrete_tree):
    parameter_estimators.fraction_missing(discrete_tree, key_added=None)
    assert "fraction_missing" not in discrete_tree.obs.columns


def test_estimate_mutation_rate(discrete_tree, continuous_tree):
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
        parameter_estimators.estimate_mutation_rate(
            discrete_tree, continuous=True, depth_key="depth"
        )


def test_estimate_missing_data_bad_cases(discrete_tree, continuous_tree):
    with pytest.raises(ParameterEstimateError):
        parameter_estimators.estimate_missing_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateError):
        parameter_estimators.estimate_missing_rates(
            discrete_tree,
            continuous=False,
            heritable_missing_rate=0.25,
            stochastic_missing_probability=0.2,
        )

    with pytest.raises(ParameterEstimateError):
        discrete_tree.uns["heritable_missing_rate"] = 0.25
        discrete_tree.uns["stochastic_missing_probability"] = 0.2
        parameter_estimators.estimate_missing_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateWarning):
        discrete_tree.uns.pop("heritable_missing_rate", None)
        discrete_tree.uns.pop("stochastic_missing_probability", None)
        discrete_tree.uns["heritable_missing_rate"] = 0.5
        parameter_estimators.estimate_missing_rates(discrete_tree, continuous=False)

    with pytest.raises(ParameterEstimateWarning):
        continuous_tree.uns["stochastic_missing_probability"] = 0.9
        parameter_estimators.estimate_missing_rates(continuous_tree, continuous=True)


def test_estimate_stochastic_missing_data_probability(discrete_tree, continuous_tree):
    s_missing_prob = parameter_estimators.estimate_missing_rates(
        discrete_tree, continuous=False, heritable_missing_rate=0.25
    )[0]
    assert np.isclose(s_missing_prob, 0.0518518518518518)

    discrete_tree.uns["heritable_missing_rate"] = 0.25
    s_missing_prob = parameter_estimators.estimate_missing_rates(discrete_tree, continuous=False)[0]
    assert np.isclose(s_missing_prob, 0.0518518518518518)

    s_missing_prob = parameter_estimators.estimate_missing_rates(
        discrete_tree, continuous=False, assume_root_implicit_branch=False
    )[0]
    assert np.isclose(s_missing_prob, 13 / 45)

    s_missing_prob = parameter_estimators.estimate_missing_rates(
        continuous_tree, continuous=True, heritable_missing_rate=0.05
    )[0]
    assert np.isclose(s_missing_prob, 0.046322071416968195)

    continuous_tree.uns["heritable_missing_rate"] = 0.05
    s_missing_prob = parameter_estimators.estimate_missing_rates(continuous_tree, continuous=True)[
        0
    ]
    assert np.isclose(s_missing_prob, 0.046322071416968195)

    s_missing_prob = parameter_estimators.estimate_missing_rates(
        continuous_tree, continuous=True, assume_root_implicit_branch=False
    )[0]
    assert np.isclose(s_missing_prob, 0.10250124994244929)


def test_estimate_heritable_missing_data_rate(discrete_tree, continuous_tree):
    h_missing_rate = parameter_estimators.estimate_missing_rates(
        discrete_tree, continuous=False, stochastic_missing_probability=0.12
    )[1]
    assert np.isclose(h_missing_rate, 0.23111904017137075)

    discrete_tree.uns["stochastic_missing_probability"] = 0.2
    h_missing_rate = parameter_estimators.estimate_missing_rates(discrete_tree, continuous=False)[1]
    assert np.isclose(h_missing_rate, 0.2062994740159002)

    h_missing_rate = parameter_estimators.estimate_missing_rates(
        discrete_tree, continuous=False, assume_root_implicit_branch=False
    )[1]
    assert np.isclose(h_missing_rate, 0.2928932188134524)

    h_missing_rate = parameter_estimators.estimate_missing_rates(
        continuous_tree, continuous=True, stochastic_missing_probability=0.04
    )[1]
    assert np.isclose(h_missing_rate, 0.05188011778689765)

    continuous_tree.uns["stochastic_missing_probability"] = 0.1
    h_missing_rate = parameter_estimators.estimate_missing_rates(continuous_tree, continuous=True)[
        1
    ]
    assert np.isclose(h_missing_rate, 0.0335154979510034)

    h_missing_rate = parameter_estimators.estimate_missing_rates(
        continuous_tree, continuous=True, assume_root_implicit_branch=False
    )[1]
    assert np.isclose(h_missing_rate, 0.05121001550277538)


def test_mutation_proportion_out_of_bounds(discrete_tree):
    """Test that invalid mutation proportions raise ParameterEstimateError."""
    discrete_tree.uns["mutation_proportion"] = 1.5
    with pytest.raises(ParameterEstimateError, match="Mutation proportion must be between 0 and 1"):
        parameter_estimators.estimate_mutation_rate(discrete_tree)

    discrete_tree.uns["mutation_proportion"] = -0.5
    with pytest.raises(ParameterEstimateError, match="Mutation proportion must be between 0 and 1"):
        parameter_estimators.estimate_mutation_rate(discrete_tree)


def test_layer_deprecation_in_all_functions(discrete_tree):
    """Test that the 'layer' deprecation warning appears in all relevant functions."""
    with pytest.warns(DeprecationWarning, match="'layer' is deprecated"):
        parameter_estimators.fraction_mutated(discrete_tree, layer="characters")

    with pytest.warns(DeprecationWarning, match="'layer' is deprecated"):
        parameter_estimators.fraction_missing(discrete_tree, layer="characters")

    with pytest.warns(DeprecationWarning, match="'layer' is deprecated"):
        parameter_estimators.estimate_mutation_rate(
            discrete_tree, continuous=False, layer="characters"
        )

    with pytest.warns(DeprecationWarning, match="'layer' is deprecated"):
        parameter_estimators.estimate_missing_rates(
            discrete_tree, continuous=False, stochastic_missing_probability=0.1, layer="characters"
        )


def test_check_continuous_not_int_empty_edges():
    """Test that empty edges list returns early without error."""
    tree = nx.DiGraph()
    parameter_estimators._check_continuous_not_int(tree, [], continuous=True)


def test_fraction_missing_various_missing_states():
    """Test fraction_missing with str and list[str] input types."""
    cm_int = pd.DataFrame({"A": [0, 1, -1], "B": [1, -2, -1]}).T
    tdata = build_tree([("root", "A"), ("root", "B")], cm_int)
    result = parameter_estimators.fraction_missing(tdata, missing_state=-1, key_added=None)
    assert result == pytest.approx(2 / 6)
    result = parameter_estimators.fraction_missing(tdata, missing_state=[-1, -2], key_added=None)
    assert result == pytest.approx(3 / 6)

    cm_str = pd.DataFrame({"C": ["0", "1", "NA"], "D": ["1", "-", "NA"]}).T
    tdata = build_tree([("root", "C"), ("root", "D")], cm_str)
    result = parameter_estimators.fraction_missing(tdata, missing_state="NA", key_added=None)
    assert result == pytest.approx(2 / 6)
    result = parameter_estimators.fraction_missing(tdata, missing_state=["NA", "-"], key_added=None)
    assert result == pytest.approx(3 / 6)


def test_fraction_mutated_various_unmodified_states():
    """Test fraction_mutated with str and list[str] input types."""
    cm_int = pd.DataFrame({"A": [0, 1, -1], "B": [99, 0, -1]}).T
    tdata = build_tree([("root", "A"), ("root", "B")], cm_int)
    result = parameter_estimators.fraction_mutated(
        tdata, missing_state=-1, unmodified_state=0, key_added=None
    )
    assert result == pytest.approx(2 / 4)
    result = parameter_estimators.fraction_mutated(
        tdata, missing_state=-1, unmodified_state=[0, 99], key_added=None
    )
    assert result == pytest.approx(1 / 4)

    cm_str = pd.DataFrame({"C": ["0", "1", "NA"], "D": ["*", "0", "NA"]}).T
    tdata = build_tree([("root", "C"), ("root", "D")], cm_str)
    result = parameter_estimators.fraction_mutated(
        tdata, missing_state="NA", unmodified_state="0", key_added=None
    )
    assert result == pytest.approx(2 / 4)
    result = parameter_estimators.fraction_mutated(
        tdata, missing_state="NA", unmodified_state=["0", "*"], key_added=None
    )
    assert result == pytest.approx(1 / 4)


if __name__ == "__main__":
    pytest.main([__file__])
