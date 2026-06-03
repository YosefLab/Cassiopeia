"""Tests for the reroot() API and the rooting procedures."""

import networkx as nx
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas


def roots(g):
    return [n for n in g.nodes if g.in_degree(n) == 0]


def leaves(g):
    return sorted(n for n in g.nodes if g.out_degree(n) == 0)


CM = pd.DataFrame.from_dict(
    {"a": [1, 1, 0], "b": [1, 1, 0], "d": [2, 0, 0], "e": [2, 0, 2]},
    orient="index",
    columns=["x1", "x2", "x3"],
)


@pytest.fixture
def tdata():
    g = nx.DiGraph()
    for u, v, length in [
        ("r", "i1", 1.0),
        ("r", "e", 5.0),
        ("i1", "i2", 1.0),
        ("i1", "d", 1.0),
        ("i2", "a", 1.0),
        ("i2", "b", 1.0),
    ]:
        g.add_edge(u, v, length=length)
    return td.TreeData(
        obs=pd.DataFrame(index=list(CM.index)),
        obst={"t": g},
        obsm={"characters": CM},
        uns={"missing_state_indicator": -1},
    )


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("outgroup", {"outgroup": "e"}),
        ("midpoint", {}),
        ("centroid", {}),
        ("shared_mutation", {}),
    ],
)
def test_reroot_produces_valid_rooted_tree(tdata, method, kwargs):
    cas.solver.reroot(tdata, method=method, tree_key="t", **kwargs)
    rt = tdata.obst["t"]
    assert nx.is_tree(rt)
    assert len(roots(rt)) == 1
    assert leaves(rt) == sorted(CM.index)


def test_reroot_copy_returns_new(tdata):
    out = cas.solver.reroot(tdata, method="centroid", tree_key="t", copy=True)
    assert isinstance(out, td.TreeData)


def test_reroot_key_added(tdata):
    cas.solver.reroot(tdata, method="centroid", tree_key="t", key_added="rerooted")
    assert "rerooted" in tdata.obst
    assert "t" in tdata.obst


def test_reroot_unknown_method_raises(tdata):
    with pytest.raises(ValueError):
        cas.solver.reroot(tdata, method="not_a_method", tree_key="t")


def test_reroot_rejects_non_treedata():
    g = nx.DiGraph()
    g.add_edge("r", "a")
    with pytest.raises(TypeError):
        cas.solver.reroot(g, method="centroid")


def test_midpoint_uses_branch_lengths(tdata):
    # The longest path runs through the long 'r'-'e' branch (length 5); the
    # midpoint root should separate 'e' from the rest.
    cas.solver.reroot(tdata, method="midpoint", tree_key="t")
    rt = tdata.obst["t"]
    root = roots(rt)[0]
    children = list(rt.successors(root))
    # one side of the root is the single leaf 'e'
    side_leaf_sets = [
        {n for n in ({c} | nx.descendants(rt, c)) if rt.out_degree(n) == 0} for c in children
    ]
    assert {"e"} in side_leaf_sets


@pytest.mark.parametrize("method", ["midpoint", "centroid", "shared_mutation"])
def test_nj_supports_rooting_methods(method):
    cm = pd.DataFrame.from_dict(
        {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = td.TreeData(
        obs=pd.DataFrame(index=list(cm.index)),
        obsm={"characters": cm},
        uns={"missing_state_indicator": -1},
    )
    cas.solver.nj(tdata, root=method, tree_key="nj")
    rt = tdata.obst["nj"]
    assert nx.is_tree(rt)
    assert len(roots(rt)) == 1
    assert leaves(rt) == sorted(cm.index)
