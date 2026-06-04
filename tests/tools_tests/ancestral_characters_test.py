"""Tests for tools.ancestral_characters and tools.collapse_edges (TreeData)."""

import networkx as nx
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.mixins.errors import CassiopeiaError


def make_tdata():
    g = nx.DiGraph()
    for u, v in [("4", "0"), ("4", "1"), ("5", "2"), ("5", "3"), ("6", "4"), ("6", "5")]:
        g.add_edge(u, v)
    cm = pd.DataFrame.from_dict(
        {
            "0": [1, 0, 3, 4, 5],
            "1": [1, 0, 3, 3, -1],
            "2": [1, 2, 3, 0, -1],
            "3": [1, 0, 3, 0, -1],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )
    return td.TreeData(
        obs=pd.DataFrame(index=["0", "1", "2", "3"]),
        obst={"tree": g},
        obsm={"characters": cm},
        uns={"missing_state_indicator": -1},
    )


def leaves(g):
    return [n for n in g.nodes if g.out_degree(n) == 0]


def test_ancestral_characters_infers_internal_states():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    # internal node 6 is the LCA of all four leaves under Camin-Sokal
    assert g.nodes["6"]["characters"] == [1, 0, 3, 0, 5]
    # leaves are seeded under the same key
    assert g.nodes["0"]["characters"] == [1, 0, 3, 4, 5]


def test_ancestral_characters_copy_returns_new():
    tdata = make_tdata()
    out = cas.tl.ancestral_characters(tdata, tree_key="tree", copy=True)
    assert isinstance(out, td.TreeData)
    # original untouched
    assert "characters" not in tdata.obst["tree"].nodes["6"]


def test_collapse_edges_mutationless():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.collapse_edges(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    # node 4 has identical states to its parent 6 and is collapsed out
    assert "4" not in g.nodes
    assert set(leaves(g)) == {"0", "1", "2", "3"}
    # 0 and 1 reattach directly to 6
    assert set(g.successors("6")) == {"0", "1", "5"}


def test_collapse_edges_requires_ancestral_states():
    tdata = make_tdata()
    with pytest.raises(CassiopeiaError):
        cas.tl.collapse_edges(tdata, tree_key="tree")


def test_collapse_edges_rejects_non_treedata():
    g = nx.DiGraph()
    g.add_edge("r", "a")
    with pytest.raises(TypeError):
        cas.tl.collapse_edges(g)


def test_collapse_edges_unknown_criteria():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    with pytest.raises(ValueError):
        cas.tl.collapse_edges(tdata, tree_key="tree", criteria="not_a_criterion")


def test_collapse_edges_copy():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    out = cas.tl.collapse_edges(tdata, tree_key="tree", copy=True)
    assert isinstance(out, td.TreeData)
    # original tree still has node 4
    assert "4" in tdata.obst["tree"].nodes
    assert "4" not in out.obst["tree"].nodes


def test_collapse_edges_mutationless_sums_branch_lengths():
    tdata = make_tdata()
    g = tdata.obst["tree"].copy()
    # node 4 is mutationless w.r.t. parent 6; its branch lengths should sum
    g["6"]["4"]["length"] = 1.5
    g["4"]["0"]["length"] = 2.0
    g["4"]["1"]["length"] = 0.5
    tdata.obst["tree"] = g
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.collapse_edges(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    assert "4" not in g.nodes
    assert g["6"]["0"]["length"] == pytest.approx(3.5)
    assert g["6"]["1"]["length"] == pytest.approx(2.0)


def make_unifurcation_tdata():
    g = nx.DiGraph()
    for u, v in [("0", "1"), ("0", "2"), ("1", "3"), ("2", "4"), ("2", "5")]:
        g.add_edge(u, v)
    g["0"]["1"]["length"] = 1.0
    g["0"]["2"]["length"] = 2.0
    g["1"]["3"]["length"] = 1.0
    g["2"]["4"]["length"] = 2.0
    g["2"]["5"]["length"] = 1.0
    return td.TreeData(obs=pd.DataFrame(index=["3", "4", "5"]), obst={"tree": g})


def test_collapse_edges_unifurcation():
    tdata = make_unifurcation_tdata()
    cas.tl.collapse_edges(tdata, tree_key="tree", criteria="unifurcation")
    g = tdata.obst["tree"]
    # node 1 (single child) is spliced out, branch lengths summed
    assert "1" not in g.nodes
    assert set(g.successors("0")) == {"2", "3"}
    assert g["0"]["3"]["length"] == pytest.approx(2.0)


def test_collapse_edges_unifurcation_needs_no_characters():
    # the structural criterion must not require character states on nodes
    tdata = make_unifurcation_tdata()
    out = cas.tl.collapse_edges(tdata, tree_key="tree", criteria="unifurcation", copy=True)
    assert "1" not in out.obst["tree"].nodes
    assert "1" in tdata.obst["tree"].nodes


def test_mean_depth():
    g = nx.DiGraph()
    for u, v in [("0", "1"), ("0", "2"), ("2", "3"), ("2", "4")]:
        g.add_edge(u, v)
    depths = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
    for n, d in depths.items():
        g.nodes[n]["depth"] = d
    tdata = td.TreeData(obs=pd.DataFrame(index=["1", "3", "4"]), obst={"tree": g})
    # leaves are 1 (depth 1), 3 and 4 (depth 2) -> mean 5/3
    assert cas.tl.mean_depth(tdata, "depth", tree_key="tree") == pytest.approx(5 / 3)
