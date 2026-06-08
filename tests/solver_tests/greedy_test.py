"""Tests for the functional greedy() API on TreeData."""

import warnings

import networkx as nx
import pandas as pd
import pytest
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
    uns = {"missing_state": -1, "unmodified_state": 0}
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
    cas.solver.greedy(tdata, key_added="greedy", missing_state=-1, unmodified_state=0, priors=False)
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
    # priors=True (default) reads tdata.uns["priors"].
    cas.solver.greedy(tdata, key_added="greedy", missing_state=-1, unmodified_state=0)
    assert set(leaves(tdata.obst["greedy"])) == set(cm.index)


def test_greedy_priors_true_without_priors_raises():
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = chars_tdata(cm)  # no priors in uns
    with pytest.raises(ValueError, match="priors=True but no priors"):
        cas.solver.greedy(tdata, key_added="greedy", priors=True)


def test_greedy_priors_list_normalized_to_dict():
    # The simulator stores priors as a list of per-character dicts; this is
    # normalized to a position-keyed dict and must match the dict form.
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    priors_dict = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
    priors_list = [{1: 0.5, 2: 0.5}, {1: 0.2, 2: 0.8}, {1: 0.3, 2: 0.7}]

    td_d = chars_tdata(cm)
    td_l = chars_tdata(cm)
    cas.solver.greedy(td_d, key_added="greedy", priors=priors_dict)
    cas.solver.greedy(td_l, key_added="greedy", priors=priors_list)
    assert set(leaves(td_d.obst["greedy"])) == set(leaves(td_l.obst["greedy"]))


def test_greedy_priors_dict_passed_directly():
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
    tdata = chars_tdata(cm)  # no priors in uns; passed explicitly
    cas.solver.greedy(tdata, key_added="greedy", priors=priors)
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
    cas.solver.greedy(tdata, key_added="greedy", priors=False)
    assert set(leaves(tdata.obst["greedy"])) == set(cm.index)


def test_greedy_copy_returns_new_and_leaves_original():
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = chars_tdata(cm)
    out = cas.solver.greedy(tdata, key_added="greedy", copy=True, priors=False)
    assert isinstance(out, td.TreeData)
    assert "greedy" in out.obst
    # original is untouched when copy=True
    assert "greedy" not in tdata.obst
    # in-place returns None
    assert cas.solver.greedy(tdata, key_added="greedy", priors=False) is None
    assert "greedy" in tdata.obst


# ── missing-data classifier ───────────────────────────────────────────────────


def test_greedy_average_classifier_string_and_callable_match():
    from cassiopeia.solver.greedy import _assign_missing_average

    cm = pd.DataFrame.from_dict(
        {
            "a": [1, 1, 0],
            "b": [1, 2, 0],
            "c": [-1, 2, 1],
            "d": [2, 0, 0],
            "e": [2, 0, 2],
        },
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    str_tdata = chars_tdata(cm.copy())
    cb_tdata = chars_tdata(cm.copy())
    # the "average" string default resolves to _assign_missing_average
    cas.solver.greedy(
        str_tdata, key_added="greedy", missing_data_classifier="average", priors=False
    )
    cas.solver.greedy(
        cb_tdata, key_added="greedy", missing_data_classifier=_assign_missing_average, priors=False
    )
    assert set(leaves(str_tdata.obst["greedy"])) == set(cm.index)
    for t in [("a", "d", "e"), ("b", "c", "d")]:
        assert find_triplet_structure(t, str_tdata.obst["greedy"]) == find_triplet_structure(
            t, cb_tdata.obst["greedy"]
        )


def test_greedy_unknown_classifier_raises():
    from cassiopeia.mixins import GreedySolverError

    cm = pd.DataFrame.from_dict(
        {"a": [1, 0], "b": [1, 1], "c": [2, 0]}, orient="index", columns=["x1", "x2"]
    )
    tdata = chars_tdata(cm)
    with pytest.raises(GreedySolverError):
        cas.solver.greedy(
            tdata, key_added="greedy", missing_data_classifier="not_a_method", priors=False
        )


# ── deprecation ───────────────────────────────────────────────────────────────


def test_vanilla_greedy_solver_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cas.solver.VanillaGreedySolver()
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
