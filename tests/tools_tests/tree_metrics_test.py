"""
Tests for cassiopeia/tools/tree_metrics.py (TreeData functional API).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from cassiopeia.mixins import CassiopeiaError, TreeMetricError
from cassiopeia.tools import tree_metrics
from cassiopeia.tools.ancestral_characters import _seed_leaf_states
from cassiopeia.tools.topology import get_root
from cassiopeia.utils import _get_characters, _get_digraph

from .conftest import SMALL_NET_EDGES, build_tree

PARSIMONY_CM = pd.DataFrame.from_dict(
    {
        "node0": [1, -1, -1],
        "node1": [2, 1, -1],
        "node2": [2, -1, -1],
        "node3": [1, 2, 2],
        "node4": [1, 1, 2],
    },
    orient="index",
)


def seeded_graph(tdata, characters_key="characters"):
    """Return a copy of the tree graph with leaf states seeded, and the root."""
    g, _ = _get_digraph(tdata, "tree", copy=True)
    _seed_leaf_states(g, _get_characters(tdata, characters_key), characters_key)
    return g, get_root(g)


def set_states(tdata, states, key="characters"):
    """Set character-state node attributes on the tree graph in place."""
    g = tdata.obst["tree"].copy()
    for node, st in states.items():
        g.nodes[node][key] = st
    tdata.obst["tree"] = g


def reset_params(tdata):
    for k in (
        "mutation_rate",
        "heritable_missing_rate",
        "stochastic_missing_probability",
        "mutation_proportion",
        "missing_proportion",
    ):
        tdata.uns.pop(k, None)


# ---------------------------------------------------------------------------
# Parsimony
# ---------------------------------------------------------------------------
def test_parsimony_infer_internal_states():
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    p = tree_metrics.calculate_parsimony(tdata, tree_key="tree", infer_ancestral_characters=True)
    assert p == 8

    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    p = tree_metrics.calculate_parsimony(
        tdata,
        tree_key="tree",
        infer_ancestral_characters=True,
        treat_missing_as_mutation=True,
    )
    assert p == 12


def test_parsimony_requires_states():
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    with pytest.raises(CassiopeiaError):
        tree_metrics.calculate_parsimony(tdata, tree_key="tree", infer_ancestral_characters=False)


def test_parsimony_specify_internal_states():
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    states = {leaf: list(PARSIMONY_CM.loc[leaf]) for leaf in PARSIMONY_CM.index}
    states.update({"node7": [0, 0, 0], "node5": [0, 0, 0], "node6": [0, 0, 2]})
    set_states(tdata, states)

    p = tree_metrics.calculate_parsimony(tdata, tree_key="tree", infer_ancestral_characters=False)
    assert p == 9

    set_states(tdata, states)
    p = tree_metrics.calculate_parsimony(
        tdata,
        tree_key="tree",
        infer_ancestral_characters=False,
        treat_missing_as_mutation=True,
    )
    assert p == 14


# ---------------------------------------------------------------------------
# log_transition_probability
# ---------------------------------------------------------------------------
def test_log_transition_probability():
    priors = {0: {1: 0.2, 2: 0.7, 3: 0.1}, 1: {1: 0.2, 2: 0.6, 3: 0.2}}
    missing_state = -1
    mut = lambda t: t * 0.2
    miss = lambda t: t * 0.1

    def p(character, s, s_, t):
        return tree_metrics.log_transition_probability(
            character, priors, missing_state, s, s_, t, mut, miss
        )

    assert p(0, -1, -1, 1) == np.log(1)
    assert np.isclose(p(0, 1, -1, 1), np.log(0.1))
    assert np.isclose(p(0, 0, -1, 2), np.log(0.2))
    assert np.isclose(p(0, 2, -1, 3), np.log(0.3))
    assert p(0, -1, "&", 1) == -1e16
    assert p(0, 0, "&", 1) == np.log(0.9)
    assert p(0, 1, "&", 1) == np.log(0.9)
    assert np.isclose(p(0, 0, 0, 1), np.log(0.72))
    assert np.isclose(p(0, 0, 0, 2), np.log(0.48))
    assert p(0, -1, 0, 1) == -1e16
    assert p(0, 1, 0, 1) == -1e16
    assert p(0, -1, 2, 1) == -1e16
    assert np.isclose(p(0, 1, 1, 1), np.log(0.9))
    assert np.isclose(p(0, 2, 2, 3), np.log(0.7))
    assert np.isclose(p(0, 0, 2, 1), np.log(0.2 * 0.9 * 0.7))
    assert np.isclose(p(1, 0, 2, 1), np.log(0.2 * 0.9 * 0.6))


# ---------------------------------------------------------------------------
# log_likelihood_of_character
# ---------------------------------------------------------------------------
def test_log_likelihood_of_character():
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
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors)
    g, root = seeded_graph(tdata)

    stochastic_missing_probability = 0.3
    mut = lambda t: 0.44967879185089554
    miss = lambda t: 0.17017346663375654

    def ll(character):
        return tree_metrics.log_likelihood_of_character(
            g,
            root,
            character,
            priors,
            -1,
            "characters",
            False,
            mut,
            miss,
            stochastic_missing_probability,
            1,
        )

    assert np.isclose(ll(0), np.log(0.0014153576307335343))
    assert np.isclose(ll(1), np.log(0.03230988091167525))
    assert np.isclose(ll(2), np.log(0.23080700775778995))


# ---------------------------------------------------------------------------
# get_tracing_parameters / parameter validation
# ---------------------------------------------------------------------------
def test_missing_priors_raises():
    # No priors in uns -> calculate_likelihood raises before parameter estimation.
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    tdata.uns["stochastic_missing_probability"] = 0.2
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tdata, model="discrete")


def test_bad_tracing_parameters():
    # All three rate parameters are supplied (so no estimation occurs); each
    # case makes exactly one invalid and expects a TreeMetricError.
    def tree(mutation_rate, heritable_missing_rate, stochastic_missing_probability):
        tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM, priors={0: {1: 1}})
        tdata.uns["mutation_rate"] = mutation_rate
        tdata.uns["heritable_missing_rate"] = heritable_missing_rate
        tdata.uns["stochastic_missing_probability"] = stochastic_missing_probability
        return tdata

    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(-1, 0.2, 0.2), model="continuous")
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(-1, 0.2, 0.2), model="discrete")
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(0.5, -1, 0.2), model="continuous")
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(0.5, 1.5, 0.2), model="discrete")
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(0.5, 0.2, -1), model="continuous")
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tree(0.5, 0.2, 1.5), model="continuous")


def test_invalid_model():
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM, priors={0: {1: 1}})
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tdata, model="not_a_model")


def test_get_tracing_parameters_discrete():
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
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors)

    tdata.uns["stochastic_missing_probability"] = 0.3
    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=False, assume_root_implicit_branch=True
    )
    assert params == pytest.approx((0.44967879185089554, 0.17017346663375654, 0.3), abs=1e-6)

    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=False, assume_root_implicit_branch=False
    )
    assert params == pytest.approx((0.5917517095361371, 0.2440710539815455, 0.3), abs=1e-6)

    reset_params(tdata)
    tdata.uns["heritable_missing_rate"] = 0.25
    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=False, assume_root_implicit_branch=True
    )
    assert params == pytest.approx((0.44967879185089554, 0.25, 0.0518518518518518), abs=1e-6)

    reset_params(tdata)
    tdata.uns["stochastic_missing_probability"] = 0.3
    tdata.uns["heritable_missing_rate"] = 0.25
    tdata.uns["mutation_rate"] = 0.25
    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=False, assume_root_implicit_branch=True
    )
    assert params == (0.25, 0.25, 0.3)


def test_get_tracing_parameters_continuous():
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
    priors = {0: {1: 0.2, 2: 0.7, 3: 0.1}, 1: {1: 0.2, 2: 0.7, 3: 0.1}, 2: {1: 0.2, 2: 0.7, 3: 0.1}}
    branch_lengths = {("node5", "node0"): 1.5, ("node6", "node3"): 2.0}
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors, branch_lengths=branch_lengths)

    tdata.uns["stochastic_missing_probability"] = 0.1
    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=True, assume_root_implicit_branch=True
    )
    assert params == pytest.approx((0.5917110077950752, 0.033515497951003406, 0.1), abs=1e-6)

    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=True, assume_root_implicit_branch=False
    )
    assert params == pytest.approx((0.90410501812166781, 0.05121001550277539, 0.1), abs=1e-6)

    reset_params(tdata)
    tdata.uns["heritable_missing_rate"] = 0.05
    params = tree_metrics.get_tracing_parameters(
        tdata, continuous=True, assume_root_implicit_branch=True
    )
    assert params == pytest.approx((0.5917110077950752, 0.05, 0.046322071416968195), abs=1e-6)


# ---------------------------------------------------------------------------
# calculate_likelihood
# ---------------------------------------------------------------------------
def test_likelihood_bad_cases():
    # priors not specified
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM)
    tdata.uns["stochastic_missing_probability"] = 0.2
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(tdata, model="discrete")

    # use_internal_character_states with no internal states annotated
    priors = {0: {1: 0.3, 2: 0.7}, 1: {1: 0.3, 2: 0.7}, 2: {1: 0.3, 2: 0.7}}
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM, priors=priors)
    tdata.uns["stochastic_missing_probability"] = 0.2
    with pytest.raises(TreeMetricError):
        tree_metrics.calculate_likelihood(
            tdata, model="discrete", use_internal_character_states=True
        )

    # fully annotated internal states yielding a -inf likelihood
    states = {leaf: list(PARSIMONY_CM.loc[leaf]) for leaf in PARSIMONY_CM.index}
    states.update({"node7": [0, 0, 0], "node5": [0, 0, 0], "node6": [0, 0, 1]})
    set_states(tdata, states)
    tdata.uns["mutation_rate"] = 0.5
    tdata.uns["heritable_missing_rate"] = 0.25
    tdata.uns["stochastic_missing_probability"] = 0
    # node6's state violates irreversibility w.r.t. its leaves, so the
    # configuration is effectively impossible (a sum of -1e16 sentinels).
    L = tree_metrics.calculate_likelihood(
        tdata, model="discrete", use_internal_character_states=True
    )
    assert L < -1e15


def test_likelihood_simple_mostly_missing():
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
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors)

    tdata.uns["stochastic_missing_probability"] = 0.3
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="discrete"), -11.458928604116634
    )

    tdata.uns["mutation_rate"] = 0.5
    tdata.uns["stochastic_missing_probability"] = 0.2
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="discrete"), -11.09716890609409
    )

    tdata.uns.pop("stochastic_missing_probability")
    tdata.uns["heritable_missing_rate"] = 0.25
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="discrete"), -10.685658651089808
    )

    tdata.uns["stochastic_missing_probability"] = 0
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="discrete"), -10.549534744691526
    )


def test_likelihood_more_complex_case():
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
    priors = {i: {1: 0.3, 2: 0.7} for i in range(4)}
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors)
    tdata.uns["mutation_rate"] = 0.5
    tdata.uns["heritable_missing_rate"] = 0.25
    tdata.uns["stochastic_missing_probability"] = 0
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="discrete"), -33.11623901010781
    )


def test_likelihood_set_internal_states():
    priors = {i: {1: 0.3, 2: 0.7} for i in range(3)}
    tdata = build_tree(SMALL_NET_EDGES, PARSIMONY_CM, priors=priors)
    tdata.uns["mutation_rate"] = 0.5
    tdata.uns["heritable_missing_rate"] = 0.25
    tdata.uns["stochastic_missing_probability"] = 0

    # infer ancestral states, then use them explicitly
    import cassiopeia as cas

    cas.tl.ancestral_characters(tdata, tree_key="tree")
    L = tree_metrics.calculate_likelihood(
        tdata, model="discrete", use_internal_character_states=True
    )
    assert np.isclose(L, -24.57491637086155)

    states = {leaf: list(PARSIMONY_CM.loc[leaf]) for leaf in PARSIMONY_CM.index}
    states.update({"node7": [0, 0, 0], "node5": [0, 0, 0], "node6": [0, 0, 2]})
    set_states(tdata, states)
    L = tree_metrics.calculate_likelihood(
        tdata, model="discrete", use_internal_character_states=True
    )
    assert np.isclose(L, -28.68500929005179)


def test_likelihood_time():
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
    priors = {0: {1: 0.2, 2: 0.7, 3: 0.1}, 1: {1: 0.2, 2: 0.7, 3: 0.1}, 2: {1: 0.2, 2: 0.7, 3: 0.1}}
    branch_lengths = {("node5", "node0"): 1.5, ("node6", "node3"): 2.0}
    tdata = build_tree(SMALL_NET_EDGES, small_cm, priors=priors, branch_lengths=branch_lengths)

    tdata.uns["stochastic_missing_probability"] = 0.1
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="continuous"), -20.5238276768878
    )

    tdata.uns["mutation_rate"] = 0.5
    tdata.uns["stochastic_missing_probability"] = 0.1
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="continuous"), -20.67410206503938
    )

    tdata.uns.pop("stochastic_missing_probability")
    tdata.uns["heritable_missing_rate"] = 0.05
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="continuous"), -20.959879404598198
    )

    tdata.uns["heritable_missing_rate"] = 0.25
    tdata.uns["stochastic_missing_probability"] = 0
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="continuous"), -21.943439525312456
    )

    tdata.uns["stochastic_missing_probability"] = 0.2
    assert np.isclose(
        tree_metrics.calculate_likelihood(tdata, model="continuous"), -22.926786566275887
    )


def test_likelihood_sum_to_one():
    priors = {0: {1: 0.2, 2: 0.8}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.2, 2: 0.8}}
    edges = [("node2", "node0"), ("node2", "node1"), ("node3", "node2")]
    ls_branch = []
    ls_no_branch = []
    for a, b in itertools.product([0, 1, -1, 2], repeat=2):
        for a_, b_ in itertools.product([0, 1, -1, 2], repeat=2):
            cm = pd.DataFrame.from_dict({"node0": [a, a_], "node1": [b, b_]}, orient="index")
            tdata = build_tree(edges, cm, priors=priors)
            tdata.uns["mutation_rate"] = 0.5
            tdata.uns["heritable_missing_rate"] = 0.25
            tdata.uns["stochastic_missing_probability"] = 0.25
            L_no_branch = tree_metrics.calculate_likelihood(tdata, model="discrete")
            L_branch = tree_metrics.calculate_likelihood(tdata, model="continuous")
            ls_no_branch.append(np.exp(L_no_branch))
            ls_branch.append(np.exp(L_branch))
    assert np.isclose(sum(ls_branch), 1.0)
    assert np.isclose(sum(ls_no_branch), 1.0)


# cPHS character matrix: node0 (under node5) and node2 (under node6) share a
# mutated state at every character, and their LCA is the root, so each shared
# state is a homoplasy under the mutation model.
CPHS_CM = pd.DataFrame.from_dict(
    {
        "node0": [1, 2, 3],
        "node1": [0, 0, 0],
        "node2": [1, 2, 3],
        "node3": [0, 0, 0],
        "node4": [0, 0, 0],
    },
    orient="index",
)

CPHS_PRIORS = {
    0: {1: 0.2, 2: 0.5, 3: 0.3},
    1: {1: 0.2, 2: 0.5, 3: 0.3},
    2: {1: 0.2, 2: 0.5, 3: 0.3},
}


def test_cphs_requires_ancestral_states():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM, priors=CPHS_PRIORS)
    with pytest.raises(TreeMetricError, match="ancestral_characters"):
        tree_metrics.calculate_cPHS(tdata)


def test_cphs_returns_scalar():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM, priors=CPHS_PRIORS)
    tree_metrics.ancestral_characters(tdata)
    score = tree_metrics.calculate_cPHS(tdata)
    assert isinstance(score, float)
    assert 0 < score <= 1


def test_cphs_explicit_params_reproducible():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM, priors=CPHS_PRIORS)
    tree_metrics.ancestral_characters(tdata)
    a = tree_metrics.calculate_cPHS(tdata, mutation_rate=0.7, collision_probability=0.3)
    b = tree_metrics.calculate_cPHS(tdata, mutation_rate=0.7, collision_probability=0.3)
    assert a == b
    # More homoplasy-prone parameters (higher collision) should not increase
    # confidence in the tree (i.e. the cPHS p-value should not decrease).
    c = tree_metrics.calculate_cPHS(tdata, mutation_rate=0.7, collision_probability=0.6)
    assert c >= a


def test_cphs_multiple_trees_returns_dict():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM, priors=CPHS_PRIORS)
    tdata.obst["tree2"] = tdata.obst["tree"].copy()
    tree_metrics.ancestral_characters(tdata, tree_key="tree")
    tree_metrics.ancestral_characters(tdata, tree_key="tree2")

    result = tree_metrics.calculate_cPHS(tdata)
    assert isinstance(result, dict)
    assert set(result) == {"tree", "tree2"}
    assert result["tree"] == result["tree2"]

    # An explicit tree_key always yields a scalar, even with multiple trees.
    scalar = tree_metrics.calculate_cPHS(tdata, tree_key="tree")
    assert isinstance(scalar, float)
    assert scalar == result["tree"]


def test_cphs_requires_ultrametric():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM, priors=CPHS_PRIORS)
    tree_metrics.ancestral_characters(tdata)
    g = tdata.obst["tree"].copy()
    g.nodes["node0"]["time"] = 5.0
    tdata.obst["tree"] = g
    with pytest.raises(TreeMetricError, match="same depth"):
        tree_metrics.calculate_cPHS(tdata)


def test_cphs_collision_probability_from_priors():
    # dict-of-dicts priors: q is the mean over characters of sum_s p_s^2.
    q = tree_metrics._collision_probability(CPHS_PRIORS, CPHS_CM, -1, 0)
    assert np.isclose(q, 0.2**2 + 0.5**2 + 0.3**2)
    # flat state->prob priors use that single distribution directly.
    q_flat = tree_metrics._collision_probability({1: 0.5, 2: 0.5}, CPHS_CM, -1, 0)
    assert np.isclose(q_flat, 0.5)

def test_cphs_collision_probability_normalizes_priors():
    # Priors are not required to sum to 1: KPTracer-style files store
    # unnormalized allele-frequency weights. q must be identical whether the
    # same distribution is given normalized or as raw weights.
    normalized = {0: {1: 0.2, 2: 0.5, 3: 0.3}}
    unnormalized = {0: {1: 0.6, 2: 1.5, 3: 0.9}}  # 3x the weights above
    q_norm = tree_metrics._collision_probability(normalized, CPHS_CM, -1, 0)
    q_raw = tree_metrics._collision_probability(unnormalized, CPHS_CM, -1, 0)
    assert np.isclose(q_norm, q_raw)
    assert q_raw <= 1.0

    # per-character priors may have different distributions; q is their mean
    mixed = {0: {1: 1.0, 2: 1.0}, 1: {1: 3.0, 2: 1.0}}
    q_mixed = tree_metrics._collision_probability(mixed, CPHS_CM, -1, 0)
    assert np.isclose(q_mixed, np.mean([0.5, 0.75**2 + 0.25**2]))

    # zero-weight priors are rejected rather than silently dividing by zero
    with pytest.raises(tree_metrics.TreeMetricError):
        tree_metrics._collision_probability({0: {1: 0.0, 2: 0.0}}, CPHS_CM, -1, 0)


def test_cphs_collision_probability_default_warns():
    tdata = build_tree(SMALL_NET_EDGES, CPHS_CM)
    tree_metrics.ancestral_characters(tdata)
    with pytest.warns(UserWarning, match="uniform distribution"):
        tree_metrics.calculate_cPHS(tdata)


if __name__ == "__main__":
    pytest.main([__file__])
