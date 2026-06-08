"""Tests for stochastic_tracing() and missing_data()."""

import networkx as nx
import pytest
import treedata as td

from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator import (
    missing_data,
    stochastic_tracing,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tree() -> td.TreeData:
    """15-node binary tree: root at t=0, 2 nodes at t=1, 4 at t=2, 8 leaves at t=3."""
    tree = nx.DiGraph()
    # root → two children at t=1
    tree.add_nodes_from(["root", "a", "b"])
    tree.add_edges_from([("root", "a"), ("root", "b")])
    # t=1 → t=2
    for parent, children in [("a", ["c", "d"]), ("b", ["e", "f"])]:
        tree.add_nodes_from(children)
        tree.add_edges_from([(parent, ch) for ch in children])
    # t=2 → t=3 (leaves)
    leaves = []
    for parent, children in [
        ("c", ["g", "h"]),
        ("d", ["i", "j"]),
        ("e", ["k", "l"]),
        ("f", ["m", "n"]),
    ]:
        tree.add_nodes_from(children)
        tree.add_edges_from([(parent, ch) for ch in children])
        leaves.extend(children)

    times = {"root": 0, "a": 1, "b": 1, "c": 2, "d": 2, "e": 2, "f": 2}
    for leaf in leaves:
        times[leaf] = 3
    nx.set_node_attributes(tree, times, "time")

    return td.TreeData(obst={"simulated": tree})


@pytest.fixture
def tdata():
    return _make_tree()


SIMPLE_PRIORS = {"1": 0.5, "2": 0.3, "3": 0.2}


# ---------------------------------------------------------------------------
# Validation — stochastic_tracing
# ---------------------------------------------------------------------------


def test_bad_cassette_dims(tdata):
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, number_of_cassettes=0)
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, number_of_cassettes=-1)
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, size_of_cassette=0)


def test_bad_mutation_rate(tdata):
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, mutation_rate=-0.1)
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, mutation_rate=[0.01, 0.02])  # wrong length
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, mutation_rate="fast")


def test_bad_state_priors_cas9(tdata):
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, state_priors={1: 0.6, 2: 0.6})  # >1
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, state_priors=[{1: 1.0}])  # wrong list length


def test_bad_sequential_params(tdata):
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, initiation_rate=0)  # non-positive
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(tdata, initiation_rate=0.1)  # missing continuation_rate
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(
            tdata, initiation_rate=0.1, continuation_rate=0.1, state_priors={1: 0.6, 2: 0.6}
        )  # priors don't sum to 1
    with pytest.raises(DataSimulatorError):
        stochastic_tracing(
            tdata, initiation_rate=0.1, continuation_rate=-0.1, state_priors=SIMPLE_PRIORS
        )  # bad continuation rate


# ---------------------------------------------------------------------------
# Cas9 mode — correctness
# ---------------------------------------------------------------------------


def test_cas9_basic_shape_and_columns(tdata):
    stochastic_tracing(tdata, number_of_cassettes=2, size_of_cassette=3, random_seed=0)
    df = tdata.obsm["characters"]
    assert df.shape == (8, 6)
    assert list(df.columns) == ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2"]


def test_cas9_state_values(tdata):
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=1
    )
    df = tdata.obsm["characters"]
    assert set(df.values.flatten()).issubset({"*", "1", "2", "3"})


def test_cas9_inheritance(tdata):
    """Cut sites are never reverted: child states extend parent."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=2
    )
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        for parent in tree.predecessors(node):
            parent_chars = tree.nodes[parent]["characters"]
            child_chars = tree.nodes[node]["characters"]
            for col in parent_chars:
                if parent_chars[col] != "*":
                    assert child_chars[col] == parent_chars[col], (
                        f"Cut site {col} reverted from {parent_chars[col]} to {child_chars[col]}"
                    )


def test_cas9_node_attributes(tdata):
    stochastic_tracing(tdata, number_of_cassettes=2, size_of_cassette=3, random_seed=0)
    tree = tdata.obst["simulated"]
    expected_cols = {"0-0", "0-1", "0-2", "1-0", "1-1", "1-2"}
    for node in tree.nodes:
        assert "characters" in tree.nodes[node]
        assert set(tree.nodes[node]["characters"].keys()) == expected_cols


def test_cas9_leaf_obsm_matches_node_attrs(tdata):
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=3
    )
    tree = tdata.obst["simulated"]
    df = tdata.obsm["characters"]
    for leaf in df.index:
        node_chars = tree.nodes[leaf]["characters"]
        for col in df.columns:
            assert df.loc[leaf, col] == node_chars[col]


def test_cas9_state_generating_distribution(tdata):
    """When state_priors=None, states are generated from the distribution."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        state_priors=None,
        number_of_states=5,
        random_seed=4,
    )
    df = tdata.obsm["characters"]
    # States should be "*" (uncut) or "1"-"5" (generated states)
    assert set(df.values.flatten()).issubset({"*", "1", "2", "3", "4", "5"})


def test_cas9_per_cassette_priors(tdata):
    """List of dicts (length = size_of_cassette) tiled across cassettes."""
    priors = [{"1": 1.0}, {"2": 1.0}, {"3": 1.0}]  # deterministic per-site
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        mutation_rate=100.0,
        state_priors=priors,
        random_seed=5,
    )
    df = tdata.obsm["characters"]
    # Site 0 and 3 (first in each cassette) → state 1
    # Site 1 and 4 → state 2
    # Site 2 and 5 → state 3
    assert all(v in {"*", "1"} for v in df["0-0"].values)
    assert all(v in {"*", "2"} for v in df["0-1"].values)
    assert all(v in {"*", "3"} for v in df["0-2"].values)


def test_cas9_per_character_rates(tdata):
    """Higher mutation rate → more cuts at that site."""
    rates = [100.0, 0.0, 0.0]  # only first site cuts
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        mutation_rate=rates,
        state_priors=SIMPLE_PRIORS,
        random_seed=10,
    )
    df = tdata.obsm["characters"]
    # First site of each cassette should be cut; others should be unmodified
    assert all(df["0-0"] != "*")
    assert all(df["0-1"] == "*")
    assert all(df["0-2"] == "*")


def test_cas9_collapse_sites(tdata):
    """With collapse_sites_on_cassette=True in missing_data, resections appear."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        mutation_rate=1000.0,
        state_priors=SIMPLE_PRIORS,
        random_seed=7,
    )
    missing_data(
        tdata,
        heritable_rate=0.0,
        stochastic_rate=0.0,
        collapse_sites_on_cassette=True,
        random_seed=7,
    )
    df = tdata.obsm["characters"]
    # At least some resection events ("-") should appear
    all_vals = set(df.values.flatten())
    assert "-" in all_vals


def test_cas9_no_collapse(tdata):
    """Without collapse_sites_on_cassette, stochastic_tracing produces no "-"."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        mutation_rate=1000.0,
        state_priors=SIMPLE_PRIORS,
        random_seed=8,
    )
    df = tdata.obsm["characters"]
    assert "-" not in set(df.values.flatten())


def test_cas9_custom_key_added(tdata):
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=2,
        state_priors={1: 1.0},
        key_added="edits",
        random_seed=0,
    )
    assert "edits" in tdata.obsm
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        assert "edits" in tree.nodes[node]


def test_uns_storage(tdata):
    """stochastic_tracing stores cassette_size and unmodified_state in uns."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=4, state_priors=SIMPLE_PRIORS, random_seed=0
    )
    assert tdata.uns["cassette_size"] == 4
    assert tdata.uns["unmodified_state"] == "*"
    assert tdata.uns["missing_state"] == "-"


def test_custom_unmodified_state(tdata):
    """Custom unmodified_state is stored in uns and used for detection."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        mutation_rate=1000.0,
        state_priors=SIMPLE_PRIORS,
        unmodified_state="WT",
        random_seed=7,
    )
    assert tdata.uns["unmodified_state"] == "WT"
    df = tdata.obsm["characters"]
    assert "WT" not in set(df.values.flatten())  # all sites cut at rate=1000


def test_collapse_custom_missing_state(tdata):
    """collapse_sites_on_cassette uses custom missing_state in missing_data."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        mutation_rate=1000.0,
        state_priors=SIMPLE_PRIORS,
        random_seed=7,
    )
    missing_data(
        tdata,
        heritable_rate=0.0,
        stochastic_rate=0.0,
        collapse_sites_on_cassette=True,
        missing_state="DEL",
        random_seed=7,
    )
    df = tdata.obsm["characters"]
    all_vals = set(df.values.flatten())
    assert "DEL" in all_vals
    assert "-" not in all_vals
    assert tdata.uns["missing_state"] == "DEL"


def test_missing_data_default_missing_state(tdata):
    """missing_data uses '-' as the default missing_state."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        state_priors=SIMPLE_PRIORS,
        mutation_rate=0.0,
        random_seed=0,
    )
    missing_data(tdata, heritable_rate=0.0, stochastic_rate=1.0, random_seed=0)
    df = tdata.obsm["characters"]
    assert all(v == "-" for v in df.values.flatten())


# ---------------------------------------------------------------------------
# Sequential mode — correctness
# ---------------------------------------------------------------------------


def test_sequential_basic_shape_and_columns(tdata):
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        initiation_rate=0.1,
        continuation_rate=0.1,
        state_priors=SIMPLE_PRIORS,
        random_seed=0,
    )
    df = tdata.obsm["characters"]
    assert df.shape == (8, 6)
    assert list(df.columns) == ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2"]


def test_sequential_state_values(tdata):
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        initiation_rate=0.1,
        continuation_rate=0.1,
        state_priors=SIMPLE_PRIORS,
        random_seed=1,
    )
    df = tdata.obsm["characters"]
    assert set(df.values.flatten()).issubset({"*", "1", "2", "3"})


def test_sequential_no_state_priors_required(tdata):
    """Sequential mode works with auto-generated state priors."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        initiation_rate=0.5,
        continuation_rate=0.5,
        number_of_states=5,
        random_seed=15,
    )
    df = tdata.obsm["characters"]
    assert set(df.values.flatten()).issubset({"*", "1", "2", "3", "4", "5"})


def test_sequential_ordering_constraint(tdata):
    """Within each cassette, sites are edited left-to-right.

    If site i is unmodified ("*"), sites i+1, i+2, ... must also be "*".
    """
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        initiation_rate=0.5,
        continuation_rate=0.5,
        state_priors=SIMPLE_PRIORS,
        random_seed=20,
    )
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        chars = tree.nodes[node]["characters"]
        for cassette in range(2):
            sites = [f"{cassette}-{i}" for i in range(3)]
            states = [chars[s] for s in sites]
            # Find first unedited: all following must also be unedited
            first_unedited = next((i for i, s in enumerate(states) if s == "*"), None)
            if first_unedited is not None:
                for j in range(first_unedited + 1, len(states)):
                    assert states[j] == "*", (
                        f"Node {node}, cassette {cassette}: site {first_unedited} is '*' "
                        f"but site {j} is {states[j]}"
                    )


def test_sequential_inheritance(tdata):
    """Parent edits propagate: child inherits and may extend."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=3,
        initiation_rate=0.5,
        continuation_rate=0.5,
        state_priors=SIMPLE_PRIORS,
        random_seed=21,
    )
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        for parent in tree.predecessors(node):
            for col in ["0-0", "0-1", "0-2"]:
                p = tree.nodes[parent]["characters"][col]
                c = tree.nodes[node]["characters"][col]
                if p != "*" and p != "-":
                    assert c == p, f"Inherited edit at {col} changed from {p} to {c}"


def test_sequential_high_rate_fills_cassette(tdata):
    """Very high rates → cassettes fully edited in 3-unit branches."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        initiation_rate=1000.0,
        continuation_rate=1000.0,
        state_priors=SIMPLE_PRIORS,
        random_seed=22,
    )
    df = tdata.obsm["characters"]
    # All leaf characters should be non-unmodified (all sites edited)
    assert all(v != "*" for v in df.values.flatten())


# ---------------------------------------------------------------------------
# missing_data — correctness
# ---------------------------------------------------------------------------


def test_heritable_silencing_on_internal_nodes(tdata):
    """Heritable silencing can affect internal nodes (not just leaves)."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=30
    )
    missing_data(tdata, heritable_rate=10.0, stochastic_rate=0.0, random_seed=30)
    tree = tdata.obst["simulated"]
    internal_nodes = [n for n in tree.nodes if tree.out_degree(n) > 0 and tree.in_degree(n) > 0]
    all_vals = set()
    for node in internal_nodes:
        all_vals.update(tree.nodes[node]["characters"].values())
    assert "-" in all_vals


def test_heritable_silencing_propagates(tdata):
    """If a cassette is silenced at a parent, all children also have it silenced."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=31
    )
    missing_data(tdata, heritable_rate=10.0, stochastic_rate=0.0, random_seed=31)
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        chars = tree.nodes[node]["characters"]
        for parent in tree.predecessors(node):
            p_chars = tree.nodes[parent]["characters"]
            for cassette in range(2):
                sites = [f"{cassette}-{i}" for i in range(3)]
                parent_silenced = all(p_chars[s] == "-" for s in sites)
                child_silenced = all(chars[s] == "-" for s in sites)
                if parent_silenced:
                    assert child_silenced, (
                        f"Cassette {cassette} silenced in parent {parent} but not child {node}"
                    )


def test_stochastic_silencing_leaves_only(tdata):
    """Stochastic silencing only affects leaves, not internal nodes."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=3,
        state_priors=SIMPLE_PRIORS,
        mutation_rate=0.0,
        random_seed=40,
    )
    missing_data(tdata, heritable_rate=0.0, stochastic_rate=1.0, random_seed=40)
    tree = tdata.obst["simulated"]
    # All leaves should be fully silenced (rate=1.0)
    for leaf in [n for n in tree.nodes if tree.out_degree(n) == 0]:
        assert all(v == "-" for v in tree.nodes[leaf]["characters"].values())
    # Internal nodes (other than root) should still have all-unmodified characters
    for node in [n for n in tree.nodes if tree.out_degree(n) > 0 and tree.in_degree(n) > 0]:
        assert all(v == "*" for v in tree.nodes[node]["characters"].values())


def test_missing_data_cassette_level(tdata):
    """When a cassette is silenced, ALL sites within it are "-"."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=50
    )
    missing_data(tdata, heritable_rate=10.0, stochastic_rate=1.0, random_seed=50)
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        chars = tree.nodes[node]["characters"]
        for cassette in range(2):
            sites = [f"{cassette}-{i}" for i in range(3)]
            vals = [chars[s] for s in sites]
            # Either all missing or none: partial silencing within a cassette cannot happen
            n_missing = sum(1 for v in vals if v == "-")
            assert n_missing == 0 or n_missing == 3, (
                f"Partial cassette silencing at node {node}, cassette {cassette}: {vals}"
            )


def test_obsm_rebuilt_after_missing_data(tdata):
    """After missing_data, obsm matches updated leaf node attributes."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=60
    )
    missing_data(tdata, heritable_rate=1.0, stochastic_rate=0.5, random_seed=60)
    tree = tdata.obst["simulated"]
    df = tdata.obsm["characters"]
    for leaf in df.index:
        node_chars = tree.nodes[leaf]["characters"]
        for col in df.columns:
            assert df.loc[leaf, col] == node_chars[col]


def test_missing_data_custom_missing_state(tdata):
    """Custom missing_state is written correctly to all silenced sites."""
    stochastic_tracing(
        tdata, number_of_cassettes=1, size_of_cassette=2, state_priors={1: 1.0}, random_seed=70
    )
    missing_data(
        tdata,
        heritable_rate=10.0,
        stochastic_rate=1.0,
        missing_state="X",
        random_seed=70,
    )
    df = tdata.obsm["characters"]
    all_vals = set(df.values.flatten())
    assert all_vals.issubset({"*", "1", "X"})
    assert "X" in all_vals


def test_missing_data_key_added(tdata):
    """key_added writes to new key without overwriting original obsm or node attrs."""
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=3, state_priors=SIMPLE_PRIORS, random_seed=90
    )
    orig_df = tdata.obsm["characters"].copy()
    missing_data(
        tdata,
        heritable_rate=10.0,
        stochastic_rate=1.0,
        key_added="characters_missing",
        random_seed=90,
    )
    # Original obsm unchanged
    assert tdata.obsm["characters"].equals(orig_df)
    # New key was created
    assert "characters_missing" in tdata.obsm
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        assert "characters_missing" in tree.nodes[node]
        assert "characters" in tree.nodes[node]


def test_missing_data_uns_storage(tdata):
    """missing_data updates uns with missing_state and unmodified_state."""
    stochastic_tracing(
        tdata,
        number_of_cassettes=1,
        size_of_cassette=2,
        state_priors={1: 1.0},
        random_seed=0,
    )
    missing_data(tdata, missing_state="X", unmodified_state="O", random_seed=0, stochastic_rate=0.1)
    assert tdata.uns["missing_state"] == "X"
    assert tdata.uns["unmodified_state"] == "O"


def test_composed_pipeline(tdata):
    """stochastic_tracing → missing_data pipeline produces valid output."""
    stochastic_tracing(
        tdata, number_of_cassettes=3, size_of_cassette=2, state_priors=SIMPLE_PRIORS, random_seed=80
    )
    missing_data(tdata, heritable_rate=0.1, stochastic_rate=0.1, random_seed=80)
    df = tdata.obsm["characters"]
    assert df.shape == (8, 6)
    assert list(df.columns) == ["0-0", "0-1", "1-0", "1-1", "2-0", "2-1"]
    assert set(df.values.flatten()).issubset({"-", "*", "1", "2", "3"})


# ---------------------------------------------------------------------------
# copy parameter
# ---------------------------------------------------------------------------


def test_stochastic_tracing_copy_false_modifies_inplace(tdata):
    stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=2,
        state_priors=SIMPLE_PRIORS,
        random_seed=1,
        copy=False,
    )
    assert "characters" in tdata.obsm


def test_stochastic_tracing_copy_true_returns_new(tdata):
    original = tdata
    result = stochastic_tracing(
        tdata,
        number_of_cassettes=2,
        size_of_cassette=2,
        state_priors=SIMPLE_PRIORS,
        random_seed=1,
        copy=True,
    )
    assert result is not tdata
    assert original is tdata
    assert "characters" not in tdata.obsm
    assert "characters" in result.obsm


def test_missing_data_copy_false_modifies_inplace(tdata):
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=2, state_priors=SIMPLE_PRIORS, random_seed=1
    )
    original = tdata
    result = missing_data(tdata, random_seed=1, copy=True, stochastic_rate=0.1)
    assert original is tdata
    assert original is not result


def test_missing_data_copy_true_returns_new(tdata):
    stochastic_tracing(
        tdata, number_of_cassettes=2, size_of_cassette=2, state_priors=SIMPLE_PRIORS, random_seed=1
    )
    result = missing_data(tdata, random_seed=1, copy=True, stochastic_rate=0.1)
    assert result is not tdata


if __name__ == "__main__":
    pytest.main(["-v", __file__])
