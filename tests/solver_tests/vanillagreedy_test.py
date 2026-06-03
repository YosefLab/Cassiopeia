"""Tests for the functional greedy() API on TreeData."""

import warnings

import networkx as nx
import pandas as pd
import treedata as td

import cassiopeia as cas
from cassiopeia.solver.greedy import _compute_mutation_frequencies


def find_triplet_structure(triplet, T):
    a, b, c = triplet
    a_anc = set(nx.ancestors(T, a))
    b_anc = set(nx.ancestors(T, b))
    c_anc = set(nx.ancestors(T, c))
    ab = len(a_anc & b_anc)
    ac = len(a_anc & c_anc)
    bc = len(b_anc & c_anc)
    if ab > bc and ab > ac:
        return "ab"
    if ac > bc and ac > ab:
        return "ac"
    if bc > ab and bc > ac:
        return "bc"
    return "-"


def leaves(g):
    return [n for n in g.nodes if g.out_degree(n) == 0]


def chars_tdata(cm, priors=None):
    uns = {"missing_state_indicator": -1}
    if priors is not None:
        uns["priors"] = priors
    return td.TreeData(obs=pd.DataFrame(index=list(cm.index)), obsm={"characters": cm}, uns=uns)


# ── mutation-frequency helper ─────────────────────────────────────────────────


def test_basic_freq_dict():
    cm = pd.DataFrame.from_dict(
        {
            "c1": [5, 0, 1, 2, -1],
            "c2": [0, 0, 3, 2, -1],
            "c3": [-1, 4, 0, 2, 2],
            "c4": [4, 4, 1, 2, 0],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )
    freq = _compute_mutation_frequencies(["c1", "c2", "c3", "c4"], cm.drop_duplicates(), -1)
    assert len(freq) == 5
    assert len(freq[0]) == 4
    assert len(freq[1]) == 3
    assert len(freq[2]) == 4
    assert len(freq[3]) == 2
    assert len(freq[4]) == 3
    assert freq[0][5] == 1
    assert freq[1][0] == 2
    assert freq[2][-1] == 0
    assert 3 not in freq[1]


def test_ambiguous_freq_dict():
    cm = pd.DataFrame.from_dict(
        {
            "c1": [5, (0, 1), 1, 2, -1],
            "c2": [0, 0, 3, 2, -1],
            "c3": [-1, 4, 0, (2, 3), 2],
            "c4": [4, 4, 1, 2, 0],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )
    freq = _compute_mutation_frequencies(["c1", "c2", "c3", "c4"], cm.drop_duplicates(), -1)
    assert freq[1][0] == 2
    assert freq[1][1] == 1


# ── greedy() reconstruction ───────────────────────────────────────────────────


def test_greedy_basic_topology():
    cm = pd.DataFrame.from_dict(
        {
            "a": [1, 1, 0],
            "b": [1, 2, 0],
            "c": [1, 2, 1],
            "d": [2, 0, 0],
            "e": [2, 0, 2],
        },
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = chars_tdata(cm)
    cas.solver.greedy(tdata, tree_key="greedy")
    tree = tdata.obst["greedy"]

    assert set(leaves(tree)) == set(cm.index)
    # a/b/c share the first-character mutation and split from d/e.
    assert find_triplet_structure(("a", "d", "e"), tree) == "bc"
    assert find_triplet_structure(("b", "c", "d"), tree) == "ab"


def test_greedy_with_priors_runs():
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
    tdata = chars_tdata(cm, priors=priors)
    cas.solver.greedy(tdata, tree_key="greedy")
    assert set(leaves(tdata.obst["greedy"])) == set(cm.index)


def test_greedy_duplicates_preserved():
    cm = pd.DataFrame.from_dict(
        {
            "a": [1, 1, 0],
            "b": [1, 2, 0],
            "c": [1, 2, 1],
            "d": [2, 0, 0],
            "e": [2, 0, 2],
            "f": [2, 0, 2],
        },
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = chars_tdata(cm)
    cas.solver.greedy(tdata, tree_key="greedy")
    assert set(leaves(tdata.obst["greedy"])) == set(cm.index)


# ── deprecation ───────────────────────────────────────────────────────────────


def test_vanilla_greedy_solver_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cas.solver.VanillaGreedySolver()
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
