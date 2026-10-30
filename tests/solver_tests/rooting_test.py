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
        uns={"missing_state": -1, "unmodified_state": 0},
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
        uns={"missing_state": -1, "unmodified_state": 0},
    )
    cas.solver.nj(tdata, root=method, key_added="nj")
    rt = tdata.obst["nj"]
    assert nx.is_tree(rt)
    assert len(roots(rt)) == 1
    assert leaves(rt) == sorted(cm.index)


def _total_length(g, key="length"):
    return sum(d[key] for _, _, d in g.edges(data=True))


def _with_times_only(tdata):
    """Replace edge lengths with node times (cumulative distance from root)."""
    g = tdata.obst["t"]
    times = {"r": 0.0}
    for u, v in nx.bfs_edges(g, "r"):
        times[v] = times[u] + g[u][v]["length"]
    for u, v in g.edges():
        del g[u][v]["length"]
    nx.set_node_attributes(g, times, "time")
    return tdata


def test_reroot_default_method_is_centroid(tdata):
    expected = cas.solver.reroot(tdata, method="centroid", tree_key="t", copy=True)
    cas.solver.reroot(tdata)
    assert set(tdata.obst["t"].edges) == set(expected.obst["t"].edges)


@pytest.mark.parametrize("method", ["midpoint", "centroid", "outgroup"])
def test_reroot_preserves_total_length(tdata, method):
    kwargs = {"outgroup": "e"} if method == "outgroup" else {}
    cas.solver.reroot(tdata, method=method, **kwargs)
    assert _total_length(tdata.obst["t"]) == pytest.approx(10.0)


def test_midpoint_splits_root_edge(tdata):
    # Longest path e-r-i1-i2-a has length 8; the midpoint lies 4 along the e-r edge.
    cas.solver.reroot(tdata, method="midpoint")
    rt = tdata.obst["t"]
    root = roots(rt)[0]
    lengths = {c: rt[root][c]["length"] for c in rt.successors(root)}
    assert lengths == {"e": pytest.approx(4.0), "i1": pytest.approx(2.0)}


def test_reroot_from_node_times(tdata):
    _with_times_only(tdata)
    cas.solver.reroot(tdata, method="midpoint")
    rt = tdata.obst["t"]
    root = roots(rt)[0]
    assert rt.nodes[root]["time"] == 0.0
    assert rt.nodes["e"]["time"] == pytest.approx(4.0)
    assert rt.nodes["a"]["time"] == pytest.approx(4.0)
    assert rt.nodes[root]["depth"] == 0
    assert _total_length(rt) == pytest.approx(10.0)


def test_reroot_updates_times_and_depth(tdata):
    # Stale times are recomputed from the lengths, keeping the old root's time offset.
    g = tdata.obst["t"]
    nx.set_node_attributes(g, 99.0, "time")
    cas.solver.reroot(tdata, method="outgroup", outgroup="e")
    rt = tdata.obst["t"]
    root = roots(rt)[0]
    assert rt.nodes[root]["time"] == 99.0
    for u, v in rt.edges():
        assert rt.nodes[v]["time"] == pytest.approx(rt.nodes[u]["time"] + rt[u][v]["length"])
        assert rt.nodes[v]["depth"] == rt.nodes[u]["depth"] + 1


def test_mutation_outgroup_accepts_state_sequences():
    from cassiopeia.solver.rooting import _get_mutation_outgroup

    assert sorted(_get_mutation_outgroup(CM)) == ["a", "b"]
    # Treating 1 as missing makes 2 (shared by d and e) the most common mutation.
    assert sorted(_get_mutation_outgroup(CM, missing_state=(1, "1"), unedited_state=(0,))) == [
        "d",
        "e",
    ]


def test_shared_mutation_reads_states_from_uns(tdata, monkeypatch):
    from cassiopeia.solver import rooting

    seen = {}
    original = rooting._PROCEDURES["shared_mutation"]

    def spy(graph, **kwargs):
        seen.update(kwargs)
        return original(graph, **kwargs)

    monkeypatch.setitem(rooting._PROCEDURES, "shared_mutation", spy)
    tdata.uns["missing_state"] = "?"
    tdata.uns["unmodified_state"] = "*"
    cas.solver.reroot(tdata, method="shared_mutation")
    assert seen["missing_state"] == "?"
    assert seen["unedited_state"] == "*"


def test_reroot_fills_missing_lengths_from_times(tdata):
    g = tdata.obst["t"]
    times = {"r": 2.0}
    for u, v in nx.bfs_edges(g, "r"):
        times[v] = times[u] + g[u][v]["length"]
    nx.set_node_attributes(g, times, "time")
    del g["r"]["e"]["length"]  # recovered from times (5.0)
    cas.solver.reroot(tdata, method="midpoint")
    rt = tdata.obst["t"]
    assert _total_length(rt) == pytest.approx(10.0)
    assert rt.nodes[roots(rt)[0]]["time"] == 2.0


def splits(g):
    """Leaf bipartitions of a tree (its unrooted topology)."""
    all_leaves = frozenset(leaves(g))
    below = {n: frozenset(nx.descendants(g, n) & all_leaves) or frozenset({n}) for n in g}
    return {min(below[v], all_leaves - below[v], key=sorted) for _, v in g.edges}


@pytest.mark.parametrize("method", ["midpoint", "centroid", "shared_mutation"])
def test_reroot_nj_and_simulated_trees(method):
    # Simulated trees have a unifurcating root; nj names its root "root".
    tdata = cas.sim.birth_death_process(num_extant=20, random_seed=0, key_added="sim")
    cas.sim.stochastic_tracing(tdata, 4, 3, mutation_rate=0.3, random_seed=0, tree_key="sim")
    cas.solver.nj(tdata)
    g = tdata.obst["nj"].copy()
    nx.set_edge_attributes(g, 1.0, "length")
    tdata.obst["nj"] = g
    for key in ["sim", "nj"]:
        before = splits(tdata.obst[key])
        cas.solver.reroot(tdata, method=method, tree_key=key)
        assert leaves(tdata.obst[key]) == sorted(tdata.obs_names)
        assert splits(tdata.obst[key]) <= before


def test_reroot_outgroup_requires_name(tdata):
    with pytest.raises(ValueError, match="requires the name of an outgroup"):
        cas.solver.reroot(tdata, method="outgroup")
