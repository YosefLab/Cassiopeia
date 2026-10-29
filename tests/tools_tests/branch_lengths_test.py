"""Tests for tl.estimate_branch_lengths (ConvexML).

The hand-solvable problems and regressions are ported from ConvexML's tests
(https://github.com/songlab-cal/ConvexML).
"""

import itertools
import math
from io import StringIO
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.mixins.errors import CassiopeiaError

pytest.importorskip("cvxpy")

DATA = Path(__file__).parent / "data" / "convexml"
SOLVERS = ["CLARABEL", "SCS"]
# Turn off regularization so the problems can be solved by hand.
EXACT = {"min_branch_length": 1e-4, "pseudo_mutations": 0, "pseudo_non_mutations": 0}
BINARY_EDGES = [("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"), ("2", "5"), ("2", "6")]


def make_tdata(edges, states: dict, leaf_times: dict | None = None) -> td.TreeData:
    """TreeData with character states on every node (as ancestral_characters writes)."""
    g = nx.DiGraph(edges)
    for node, state in states.items():
        g.nodes[node]["characters"] = list(state)
    leaves = [n for n in g if g.out_degree(n) == 0]
    obs = pd.DataFrame(index=leaves)
    if leaf_times is not None:
        obs["time"] = pd.Series(leaf_times)
    return td.TreeData(
        obs=obs,
        obst={"tree": g},
        obsm={"characters": pd.DataFrame([states[leaf] for leaf in leaves], index=leaves)},
        uns={"missing_state": -1, "unmodified_state": 0},
    )


def fit(tdata: td.TreeData, **kwargs) -> tuple[nx.DiGraph, dict]:
    cas.tl.estimate_branch_lengths(tdata, **kwargs)
    return tdata.obst["tree"], tdata.uns["branch_lengths"]


def nwk_to_digraph(nwk: str) -> nx.DiGraph:
    from Bio import Phylo

    tree = Phylo.read(StringIO(nwk.strip()), "newick")
    counter = itertools.count()
    for clade in tree.find_clades(order="preorder"):
        if clade.name is None:
            clade.name = f"internal-{next(counter)}"
    g = nx.DiGraph()
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            g.add_edge(clade.name, child.name, length=child.branch_length or 0.0)
    return g


def clade_times(g: nx.DiGraph, times: dict) -> dict:
    """Node times keyed by the leaves below the node (and position in unifurcations)."""
    out, below, level = {}, {}, {}
    root = next(n for n in g if g.in_degree(n) == 0)
    for node in nx.dfs_postorder_nodes(g, root):
        kids = list(g.successors(node))
        below[node] = (
            frozenset([node]) if not kids else frozenset().union(*(below[c] for c in kids))
        )
        level[node] = level[kids[0]] + 1 if len(kids) == 1 else 0
        out[below[node], level[node]] = times[node]
    return out


def newick_times(nwk: str) -> dict:
    g = nwk_to_digraph(nwk)
    root = next(n for n in g if g.in_degree(n) == 0)
    times = {root: 0.0}
    for u, v in nx.dfs_edges(g, root):
        times[v] = times[u] + g.edges[u, v]["length"]
    return clade_times(g, times)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "child, pseudo, rate, log_likelihood",
    [
        # max_t log(exp(-t)) + log(1 - exp(-t)): t = ln 2
        ([0, 1], {}, np.log(2), -1.386),
        # max_t log(exp(-t)) + 2 log(1 - exp(-t)): t = ln 3
        ([0, 1, 1], {}, np.log(3), -1.910),
        # max_t 2 log(exp(-t)) + log(1 - exp(-t)): t = ln 1.5
        ([0, 0, 1], {}, np.log(1.5), -1.910),
        # pseudocounts double the counts of [0, 1]
        ([0, 1], {"pseudo_mutations": 1, "pseudo_non_mutations": 1}, np.log(2), -1.386),
    ],
)
def test_hand_solvable_single_edge(solver, child, pseudo, rate, log_likelihood):
    tdata = make_tdata([("0", "1")], {"0": [0] * len(child), "1": child})
    g, result = fit(tdata, solver=solver, **{**EXACT, **pseudo})
    assert result["mutation_rate"] == pytest.approx(rate, abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(log_likelihood, abs=1e-3)
    assert g.nodes["0"]["time"] == 0.0
    assert g.nodes["1"]["time"] == pytest.approx(1.0, abs=1e-3)
    assert g.edges["0", "1"]["length"] == pytest.approx(1.0, abs=1e-3)
    if not pseudo:
        assert result["penalized_log_likelihood"] == pytest.approx(log_likelihood, abs=1e-3)
    else:
        assert result["penalized_log_likelihood"] == pytest.approx(2 * log_likelihood, abs=1e-2)
    assert result["method"] == "convexml"
    assert result["solver"] == solver


@pytest.mark.parametrize("solver", SOLVERS)
def test_pseudo_mutations_regularize(solver):
    # One pseudo-mutation on [0, 1] gives the likelihood of [0, 1, 1] (ConvexML).
    tdata = make_tdata([("0", "1")], {"0": [0, 0], "1": [0, 1]})
    _, result = fit(tdata, solver=solver, min_branch_length=1e-4, pseudo_mutations=1,
                    pseudo_non_mutations=0)  # fmt: skip
    assert result["mutation_rate"] == pytest.approx(np.log(3), abs=1e-3)
    assert result["penalized_log_likelihood"] == pytest.approx(-1.910, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
def test_small_tree_with_one_mutation(solver):
    states = {n: [0] for n in "0123456"}
    states["6"] = [1]
    g, result = fit(make_tdata(BINARY_EDGES, states), solver=solver, **EXACT)
    expected = {
        ("0", "1"): 1,
        ("0", "2"): 0,
        ("1", "3"): 0,
        ("1", "4"): 0,
        ("2", "5"): 1,
        ("2", "6"): 1,
    }
    for edge, length in expected.items():
        assert g.edges[edge]["length"] == pytest.approx(length, abs=1e-3)
    assert result["mutation_rate"] == pytest.approx(np.log(1.5), abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-1.910, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
def test_small_tree_regression(solver):
    states = {
        "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        "2": [0, 0, 0, 0, 0, 6, 0, 0, 0, -1],
        "3": [1, 2, 0, 0, 0, 0, 0, 0, 0, -1],
        "4": [1, 0, 3, 0, 0, 0, 0, 0, 0, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
    }
    g, result = fit(make_tdata(BINARY_EDGES, states), solver=solver, **EXACT)
    expected = {("0", "1"): 0.537, ("0", "2"): 0.219, ("1", "3"): 0.463, ("1", "4"): 0.463,
                ("2", "5"): 0.781, ("2", "6"): 0.781}  # fmt: skip
    for edge, length in expected.items():
        assert g.edges[edge]["length"] == pytest.approx(length, abs=1e-3)
    assert result["mutation_rate"] == pytest.approx(0.378, abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-22.689, abs=1e-3)
    assert result["penalized_log_likelihood"] == pytest.approx(-22.689, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
def test_subtree_without_mutations_collapses(solver):
    states = {"0": [0], "1": [1], "2": [1], "3": [1], "4": [0]}
    edges = [("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")]
    g, result = fit(make_tdata(edges, states), solver=solver, **EXACT)
    expected = {("0", "1"): 1, ("1", "2"): 0, ("1", "3"): 0, ("0", "4"): 1}
    for edge, length in expected.items():
        assert g.edges[edge]["length"] == pytest.approx(length, abs=1e-3)
    assert result["mutation_rate"] == pytest.approx(np.log(2), abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-1.386, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
def test_min_branch_length(solver):
    states = {n: [0] for n in "0123456"}
    states["6"] = [1]
    g, result = fit(
        make_tdata(BINARY_EDGES, states),
        solver=solver,
        min_branch_length=0.01,
        min_leaf_branch_length=0.01,
        pseudo_mutations=0,
        pseudo_non_mutations=0,
    )
    # Node 2 is two minimum branch lengths below the root: the implicit branch from
    # the founder to its first division (node 0's states), then the edge to 2.
    expected = {("0", "1"): 0.99, ("0", "2"): 0.02, ("1", "3"): 0.01, ("1", "4"): 0.01,
                ("2", "5"): 0.98, ("2", "6"): 0.98}  # fmt: skip
    for edge, length in expected.items():
        assert g.edges[edge]["length"] == pytest.approx(length, abs=1e-3)
    # max_r -2 r + log(1 - exp(-0.98 r))
    assert result["mutation_rate"] == pytest.approx(0.407, abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-1.926, abs=1e-3)


def test_min_leaf_branch_length_defaults_to_half():
    states = {n: [0] for n in "0123456"}
    states["6"] = [1]
    g, _ = fit(make_tdata(BINARY_EDGES, states), min_branch_length=0.02, pseudo_mutations=0,
               pseudo_non_mutations=0)  # fmt: skip
    assert g.edges["1", "3"]["length"] == pytest.approx(0.01, abs=1e-3)
    # The implicit founder branch plus the edge to node 2.
    assert g.edges["0", "2"]["length"] == pytest.approx(0.04, abs=1e-3)


def test_on_simulated_data():
    # Data simulated under the model on a tree with known times (ConvexML).
    rng = np.random.default_rng(1)
    times = {"0": 0, "1": 0.1, "2": 0.9, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0}
    g = nx.DiGraph(BINARY_EDGES)
    n_sites, rate = 400, 1.5
    states = {"0": np.zeros(n_sites, dtype=int)}
    for parent, child in nx.dfs_edges(g, "0"):
        p_mut = 1 - np.exp(-rate * (times[child] - times[parent]))
        new = (states[parent] == 0) & (rng.random(n_sites) < p_mut)
        states[child] = np.where(new, rng.integers(1, 100, n_sites), states[parent])
    g, result = fit(make_tdata(BINARY_EDGES, {n: s.tolist() for n, s in states.items()}), **EXACT)
    for node, time in times.items():
        assert g.nodes[node]["time"] == pytest.approx(time, abs=0.05)
    assert result["mutation_rate"] == pytest.approx(rate, abs=0.15)


@pytest.mark.parametrize(
    "solver_rates, math_rates, matches",
    [
        ([1.5, 2, 2.5], [1.5, 2, 2.5], True),
        ([0.05, 0.08, 0.09], [0.05, 0.08, 0.09], True),
        ([150, 200, 250], [150, 200, 250], True),
        ([1.5, 2, 2.5], [1.47, 2, 2.5], False),
        ([1.5, 2, 2.5], [1.5, 1.97, 2.5], False),
        ([1.5, 2, 2.5], [1.5, 2, 2.52], False),
    ],
)
def test_relative_rates_hand_solvable(solver_rates, math_rates, matches):
    # 0 -> 1 -> 2 with states 000 -> 100 -> 110 and per-site rates r1, r2, r3.
    tdata = make_tdata([("0", "1"), ("1", "2")], {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]})
    g, result = fit(tdata, relative_rates=solver_rates, min_leaf_branch_length=1e-4, **EXACT)
    r1, r2, r3 = math_rates
    branch1 = math.log((r1 + r2 + r3) / (r2 + r3)) / r1
    branch2 = math.log((r2 + r3) / r3) / r2
    total = branch1 + branch2
    places = 3 if matches else 4
    close = g.nodes["1"]["time"] == pytest.approx(branch1 / total, abs=1.5 * 10**-places)
    assert close == matches
    assert g.nodes["2"]["time"] == pytest.approx(1.0, abs=1e-3)
    rates = np.array(result["mutation_rate"])
    expected = np.array(solver_rates) * total
    assert np.allclose(rates, expected, atol=1.5 * 10**-places) == matches
    assert result["log_likelihood"] == pytest.approx(result["penalized_log_likelihood"], abs=1e-3)


@pytest.mark.parametrize("rates", [[1.5, -1, 2.5], [1, 3, 0], [1, 1, 1, 1], [2, 2], []])
def test_invalid_relative_rates(rates):
    tdata = make_tdata([("0", "1"), ("1", "2")], {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]})
    with pytest.raises(ValueError, match="relative_rates"):
        cas.tl.estimate_branch_lengths(tdata, relative_rates=rates, **EXACT)


def test_relative_rates_series_aligned_to_columns():
    states = {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]}
    a = make_tdata([("0", "1"), ("1", "2")], states)
    b = make_tdata([("0", "1"), ("1", "2")], states)
    cas.tl.estimate_branch_lengths(a, relative_rates=[1.5, 2, 2.5], **EXACT)
    cas.tl.estimate_branch_lengths(b, relative_rates=pd.Series({2: 2.5, 0: 1.5, 1: 2}), **EXACT)
    assert a.obst["tree"].nodes["1"]["time"] == pytest.approx(
        b.obst["tree"].nodes["1"]["time"], abs=1e-6
    )


@pytest.mark.parametrize("matches", [True, False])
def test_long_edge_mutations_with_leaf_times(matches):
    # Internal nodes with missing data that carry no information: the edits below
    # them are placed on the path from the nearest ancestor with data (ConvexML).
    solver_rates = [1.5, 2, 2.5]
    r1, r2, r3 = solver_rates if matches else [1.5, 1.97, 2.5]
    edges = [("0", "f01"), ("f01", "1"), ("f01", "l01"), ("1", "f12"), ("f12", "2"), ("f12", "l12")]
    states = {
        "0": [0, 0, 0],
        "f01": [-1, 0, 0],
        "l01": [-1, -1, -1],
        "1": [1, 0, 0],
        "f12": [1, -1, 0],
        "l12": [1, -1, -1],
        "2": [1, 1, 0],
    }
    tdata = make_tdata(edges, states, leaf_times={"2": 10.0, "l12": 9.0, "l01": 8.0})
    g, result = fit(tdata, relative_rates=solver_rates, leaf_time_key="time",
                    min_leaf_branch_length=1e-4, **EXACT)  # fmt: skip
    branch1 = math.log((r1 + r2 + r3) / (r2 + r3)) / r1
    branch2 = math.log((r2 + r3) / r3) / r2
    total = branch1 + branch2
    # Times are reported in the units of obs["time"].
    assert g.nodes["2"]["time"] == pytest.approx(10.0, abs=1e-3)
    assert g.nodes["l12"]["time"] == pytest.approx(9.0, abs=1e-3)
    assert g.nodes["l01"]["time"] == pytest.approx(8.0, abs=1e-3)
    close = g.nodes["1"]["time"] == pytest.approx(
        10 * branch1 / total, abs=1.5e-3 if matches else 1.5e-4
    )
    assert close == matches
    rates = np.array(result["mutation_rate"]) * 10
    assert np.allclose(rates, np.array(solver_rates) * total, atol=1.5e-3) == matches


def test_leaf_time_key_scales_times_and_rate():
    states = {n: [0] for n in "0123456"}
    states["6"] = [1]
    a = make_tdata(BINARY_EDGES, states)
    b = make_tdata(BINARY_EDGES, states, leaf_times=dict.fromkeys("3456", 20.0))
    cas.tl.estimate_branch_lengths(a, **EXACT)
    cas.tl.estimate_branch_lengths(b, leaf_time_key="time", **EXACT)
    for node in "0123456":
        assert b.obst["tree"].nodes[node]["time"] == pytest.approx(
            20 * a.obst["tree"].nodes[node]["time"]
        )
    assert b.uns["branch_lengths"]["mutation_rate"] == pytest.approx(
        a.uns["branch_lengths"]["mutation_rate"] / 20
    )


def test_leaf_time_key_errors():
    states = {"0": [0, 0], "1": [0, 1]}
    with pytest.raises(KeyError, match="not a column"):
        cas.tl.estimate_branch_lengths(make_tdata([("0", "1")], states), leaf_time_key="nope")
    with pytest.raises(ValueError, match="positive"):
        cas.tl.estimate_branch_lengths(
            make_tdata([("0", "1")], states, {"1": 0.0}), leaf_time_key="time"
        )
    tdata = make_tdata([("0", "1"), ("0", "2")], {**states, "2": [1, 0]}, {"1": 1.0})
    with pytest.raises(ValueError, match="no numeric time for leaf '2'"):
        cas.tl.estimate_branch_lengths(tdata, leaf_time_key="time")


def multifurcation_tdata() -> td.TreeData:
    edges = [("0", "1"), ("1", "2"), ("1", "3"), ("1", "4")]
    states = {"0": [0, 0], "1": [1, 0], "2": [1, 0], "3": [1, 0], "4": [1, 0]}
    return make_tdata(edges, states)


def test_multifurcations_keep_topology():
    tdata = multifurcation_tdata()
    g, _ = fit(tdata)
    assert set(g.edges) == {("0", "1"), ("1", "2"), ("1", "3"), ("1", "4")}
    for parent, child in g.edges:
        assert g.edges[parent, child]["length"] == pytest.approx(
            g.nodes[child]["time"] - g.nodes[parent]["time"]
        )


def test_multifurcations_are_resolved():
    # Resolving the multifurcation needs two divisions below node 1, which
    # shortens its edge from the root (ConvexML: 0.8347, 0.8885 if not resolved).
    g, _ = fit(multifurcation_tdata())
    assert g.edges["0", "1"]["length"] == pytest.approx(0.8347, abs=1e-3)


def test_root_with_several_children_gets_founder_branch():
    # The root's states belong to the founder's first division, which sits on an
    # implicit branch below the founder: the same as an explicit founder node.
    implicit = make_tdata([("0", "1"), ("0", "2")], {"0": [1, 0], "1": [1, 0], "2": [1, 1]})
    explicit = make_tdata([("f", "0"), ("0", "1"), ("0", "2")],
                          {"f": [0, 0], "0": [1, 0], "1": [1, 0], "2": [1, 1]})  # fmt: skip
    a, fit_a = fit(implicit)
    b, fit_b = fit(explicit)
    assert a.nodes["0"]["time"] == 0.0
    assert b.nodes["0"]["time"] > 0.0
    for node in "12":
        assert a.nodes[node]["time"] == pytest.approx(b.nodes[node]["time"], abs=1e-6)
    assert fit_a["mutation_rate"] == pytest.approx(fit_b["mutation_rate"], abs=1e-6)


def test_states_of_root_with_one_child_are_ignored():
    # A root with one child is the unedited founder, whatever states it stores.
    edges = [("0", "1"), ("1", "2"), ("1", "3")]
    founder = make_tdata(edges, {"0": [0, 0], "1": [1, 0], "2": [1, 0], "3": [1, 1]})
    edited = make_tdata(edges, {"0": [1, 0], "1": [1, 0], "2": [1, 0], "3": [1, 1]})
    a, _ = fit(founder)
    b, _ = fit(edited)
    for node in "123":
        assert a.nodes[node]["time"] == pytest.approx(b.nodes[node]["time"], abs=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_edited_root_hand_solvable(solver):
    # Site 0 is edited in the root: no mutation, and no branch above the root's
    # division. max_r -r + log(1 - exp(-r)) over the two unit edges: r = ln 2.
    tdata = make_tdata([("0", "1"), ("0", "2")], {"0": [5, 0], "1": [5, 1], "2": [5, 0]})
    g, result = fit(tdata, solver=solver, unedited_root=False, **EXACT)
    assert g.nodes["0"]["time"] == 0.0
    for edge in [("0", "1"), ("0", "2")]:
        assert g.edges[edge]["length"] == pytest.approx(1.0, abs=1e-3)
    assert result["mutation_rate"] == pytest.approx(np.log(2), abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-1.386, abs=1e-3)
    # By default the edit at site 0 is a mutation before the root's division, at
    # time t: max 2 log(1 - exp(-r / 2)) - r at t = 1/2, so r = 2 ln 2.
    tdata = make_tdata([("0", "1"), ("0", "2")], {"0": [5, 0], "1": [5, 1], "2": [5, 0]})
    _, result = fit(tdata, solver=solver, **EXACT)
    assert result["mutation_rate"] == pytest.approx(2 * np.log(2), abs=1e-3)
    assert result["log_likelihood"] == pytest.approx(-4 * np.log(2), abs=1e-3)


def test_edited_root_matches_dropping_its_edits():
    # Sites edited in the root carry no information about branch lengths.
    states = {
        "0": [7, 0, 0, 0, 0],
        "1": [7, 1, 0, 0, -1],
        "2": [7, 0, 0, 6, -1],
        "3": [7, 1, 2, 0, -1],
        "4": [-1, 1, 0, 3, 0],
        "5": [7, 0, 4, 6, -1],
        "6": [7, 0, 0, 6, 8],
    }
    a, fit_a = fit(make_tdata(BINARY_EDGES, states), unedited_root=False)
    b, fit_b = fit(make_tdata(BINARY_EDGES, {n: s[1:] for n, s in states.items()}),
                   unedited_root=False)  # fmt: skip
    for node in "0123456":
        assert a.nodes[node]["time"] == pytest.approx(b.nodes[node]["time"], abs=1e-5)
    assert fit_a["mutation_rate"] == pytest.approx(fit_b["mutation_rate"], rel=1e-4)


def test_edited_root_missing_sites_are_unedited():
    edges = [("0", "1"), ("0", "2")]
    missing = make_tdata(edges, {"0": [7, -1], "1": [7, 1], "2": [7, 0]})
    unedited = make_tdata(edges, {"0": [7, 0], "1": [7, 1], "2": [7, 0]})
    a, _ = fit(missing, unedited_root=False)
    b, _ = fit(unedited, unedited_root=False)
    for node in "12":
        assert a.nodes[node]["time"] == pytest.approx(b.nodes[node]["time"], abs=1e-6)


def test_string_states_match_integer_states():
    states = {
        "0": [0, 0, 0, 0],
        "1": [1, 0, 0, -1],
        "2": [0, 0, 6, -1],
        "3": [1, 2, 0, -1],
        "4": [1, 0, 3, 0],
        "5": [0, 4, 6, -1],
        "6": [0, 0, 6, 8],
    }
    code = {0: "*", -1: "-"}
    as_str = {n: [code.get(x, str(x)) for x in s] for n, s in states.items()}
    a = make_tdata(BINARY_EDGES, states)
    b = make_tdata(BINARY_EDGES, as_str)
    b.uns.update(missing_state="-", unmodified_state="*")
    cas.tl.estimate_branch_lengths(a)
    cas.tl.estimate_branch_lengths(b)
    for node in "0123456":
        assert a.obst["tree"].nodes[node]["time"] == pytest.approx(
            b.obst["tree"].nodes[node]["time"], abs=1e-6
        )


def test_intmemoir_regression():
    leaf_sequences = {
        "10": [1, 0, 0, 2, 2, 0, 2, 2, 2, 0],
        "5": [1, 1, 0, 2, 2, 0, 2, 2, 2, 0],
        "6": [1, 1, 0, 2, 1, 0, 2, 2, 2, 0],
        "8": [2, 0, 2, 1, 1, 0, 1, 1, 2, 0],
        "9": [2, 0, 2, 1, 1, 0, 1, 1, 2, 0],
        "1": [2, 1, 0, 2, 2, 0, 2, 1, 2, 0],
        "2": [2, 0, 0, 2, 2, 0, 2, 1, 2, 0],
        "3": [2, 0, 0, 2, 2, 0, 2, 1, 2, 0],
        "4": [2, 0, 0, 2, 2, 0, 2, 1, 2, 0],
    }
    g = nwk_to_digraph("((((1,2),(3,4)),(5,6)),((8,9),10));")
    leaves = [n for n in g if g.out_degree(n) == 0]
    tdata = td.TreeData(
        obs=pd.DataFrame(index=leaves),
        obst={"tree": g},
        obsm={"characters": pd.DataFrame(leaf_sequences).T.loc[leaves]},
        uns={"missing_state": -1, "unmodified_state": 0},
    )
    cas.tl.ancestral_characters(tdata, method="conservative")
    cas.tl.estimate_branch_lengths(tdata, min_branch_length=0.15)
    got = clade_times(tdata.obst["tree"], dict(tdata.obst["tree"].nodes(data="time")))
    expected = newick_times((DATA / "intmemoir_expected_newick.txt").read_text())
    assert got.keys() == expected.keys()
    for clade, time in expected.items():
        assert got[clade] == pytest.approx(time, abs=5e-5)


def test_benchmark_regression():
    g = nwk_to_digraph((DATA / "tree_0_newick.txt").read_text())
    characters = pd.read_csv(DATA / "tree_0_character_matrix.csv", index_col=0)
    characters.index = characters.index.astype(str)
    leaves = [n for n in g if g.out_degree(n) == 0]
    tdata = td.TreeData(
        obs=pd.DataFrame(index=leaves),
        obst={"tree": g},
        obsm={"characters": characters.loc[leaves]},
        uns={"missing_state": -1, "unmodified_state": 0},
    )
    cas.tl.ancestral_characters(tdata, method="conservative")
    cas.tl.estimate_branch_lengths(tdata)
    got = clade_times(tdata.obst["tree"], dict(tdata.obst["tree"].nodes(data="time")))
    expected = newick_times((DATA / "expected_newick.txt").read_text())
    assert got.keys() == expected.keys()
    for clade, time in expected.items():
        assert got[clade] == pytest.approx(time, abs=1e-5)


def test_copy_and_custom_keys():
    tdata = make_tdata([("0", "1")], {"0": [0, 0], "1": [0, 1]})
    out = cas.tl.estimate_branch_lengths(
        tdata, copy=True, time_key="t", length_key="bl", key_added="fit", **EXACT
    )
    assert isinstance(out, td.TreeData)
    assert "t" not in tdata.obst["tree"].nodes["1"]
    assert out.obst["tree"].nodes["1"]["t"] == pytest.approx(1.0, abs=1e-3)
    assert out.obst["tree"].edges["0", "1"]["bl"] == pytest.approx(1.0, abs=1e-3)
    assert out.uns["fit"]["mutation_rate"] == pytest.approx(np.log(2), abs=1e-3)


@pytest.mark.parametrize(
    "child, kwargs",
    [
        ([0], {"pseudo_mutations": 0}),  # no edits
        ([1], {"pseudo_non_mutations": 0}),  # fully edited
    ],
)
def test_degenerate_data(child, kwargs):
    tdata = make_tdata([("0", "1")], {"0": [0], "1": child})
    with pytest.raises(ValueError, match="pseudo"):
        cas.tl.estimate_branch_lengths(tdata, **kwargs)


def test_min_branch_length_too_large():
    edges = [("0", "1"), ("1", "2"), ("2", "3")]
    tdata = make_tdata(edges, {"0": [0], "1": [1], "2": [1], "3": [1]})
    with pytest.raises(ValueError, match="min_branch_length is too large"):
        cas.tl.estimate_branch_lengths(tdata, min_branch_length=0.5)


def test_requires_ancestral_characters():
    tdata = make_tdata([("0", "1")], {"1": [0, 1]})
    with pytest.raises(CassiopeiaError, match="ancestral_characters"):
        cas.tl.estimate_branch_lengths(tdata)


def test_ambiguous_states_raise():
    tdata = make_tdata([("0", "1")], {"0": [0, 0], "1": [(1, 2), 0]})
    with pytest.raises(CassiopeiaError, match="Ambiguous"):
        cas.tl.estimate_branch_lengths(tdata)


def test_rejects_non_treedata_and_bad_arguments():
    with pytest.raises(TypeError, match="TreeData"):
        cas.tl.estimate_branch_lengths(nx.DiGraph([("0", "1")]))
    tdata = make_tdata([("0", "1")], {"0": [0, 0], "1": [0, 1]})
    with pytest.raises(ValueError, match="Unknown method"):
        cas.tl.estimate_branch_lengths(tdata, method="laml")
    with pytest.raises(ValueError, match="Unknown solver"):
        cas.tl.estimate_branch_lengths(tdata, solver="GUROBI")
    with pytest.raises(ValueError, match="non-negative"):
        cas.tl.estimate_branch_lengths(tdata, pseudo_mutations=-1)


def test_sankoff_pipeline_on_simulated_data():
    tdata = cas.sim.birth_death_process(num_extant=60, random_seed=0, key_added="tree")
    cas.sim.stochastic_tracing(tdata, number_of_cassettes=6, size_of_cassette=3, mutation_rate=0.3,
                               random_seed=0)  # fmt: skip
    cas.sim.missing_data(tdata, stochastic_rate=0.1, random_seed=0)
    truth = dict(tdata.obst["tree"].nodes(data="time"))
    end = max(truth.values())
    cas.tl.ancestral_characters(tdata, method="sankoff")
    cas.tl.estimate_branch_lengths(tdata)
    g = tdata.obst["tree"]
    inner = [n for n in g if g.out_degree(n) > 0 and g.in_degree(n) > 0]
    error = np.mean([abs(g.nodes[n]["time"] - truth[n] / end) for n in inner])
    assert error < 0.15
