"""Tests for the functional nj()/upgma() API, pairwise(), and the Cython backend.

These tests exercise the TreeData / functional path. Backward-compatible class
shims are covered only by a single deprecation-warning test each.
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas


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


CM = pd.DataFrame.from_dict(
    {
        "a": [0, 1, 2, 1, 0, 0, 2, 0, 0, 0],
        "b": [1, 1, 2, 1, 0, 0, 2, 0, 0, 0],
        "c": [2, 2, 2, 1, 0, 0, 2, 0, 0, 0],
        "d": [1, 1, 1, 1, 0, 0, 2, 0, 0, 0],
        "e": [0, 0, 0, 0, 1, 2, 1, 0, 2, 0],
        "f": [0, 0, 0, 0, 2, 2, 1, 0, 2, 0],
        "g": [0, 2, 0, 0, 1, 1, 1, 0, 2, 0],
        "h": [0, 2, 0, 0, 1, 0, 0, 1, 2, 1],
        "i": [1, 2, 0, 0, 1, 0, 0, 2, 2, 1],
        "j": [1, 2, 0, 0, 1, 0, 0, 1, 1, 1],
    },
    orient="index",
    columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
)

SMALL_CM = pd.DataFrame.from_dict(
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

DUPLICATE_CM = pd.DataFrame.from_dict(
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


def make_chars_tdata(cm):
    return td.TreeData(
        obs=pd.DataFrame(index=list(cm.index)),
        obsm={"characters": cm},
        uns={"missing_state": -1, "unmodified_state": 0, "missing_state_indicator": -1},
    )


@pytest.fixture
def dist_tdata():
    samples = list("abcde")
    dist = pd.DataFrame(
        [
            [0.0, 0.1, 0.8, 0.8, 0.8],
            [0.1, 0.0, 0.8, 0.8, 0.8],
            [0.8, 0.8, 0.0, 0.1, 0.7],
            [0.8, 0.8, 0.1, 0.0, 0.7],
            [0.8, 0.8, 0.7, 0.7, 0.0],
        ],
        index=samples,
        columns=samples,
    )
    tdata = td.TreeData(obs=pd.DataFrame(index=samples))
    tdata.obsp["distances"] = dist.to_numpy()
    tdata.obsm["characters"] = SMALL_CM
    return tdata


# ── TreeData distance-key path ────────────────────────────────────────────────


def test_nj_treedata_dist_key(dist_tdata):
    cas.solver.nj(dist_tdata, dissim_key="distances", key_added="nj")
    assert "nj" in dist_tdata.obst
    tree = dist_tdata.obst["nj"]
    assert isinstance(tree, nx.DiGraph)
    assert len(tree.nodes) > 0


def test_upgma_treedata_dist_key(dist_tdata):
    cas.solver.upgma(dist_tdata, dissim_key="distances", key_added="upgma")
    assert isinstance(dist_tdata.obst["upgma"], nx.DiGraph)


def test_nj_treedata_named_outgroup_clusters(dist_tdata):
    cas.solver.nj(dist_tdata, dissim_key="distances", root="outgroup", outgroup="e", key_added="nj")
    tree = dist_tdata.obst["nj"]
    assert find_triplet_structure(("a", "b", "c"), tree) == "ab"


def test_upgma_treedata_groups_cluster_correctly(dist_tdata):
    cas.solver.upgma(dist_tdata, dissim_key="distances", key_added="upgma")
    tree = dist_tdata.obst["upgma"]
    ab = set(nx.ancestors(tree, "a")) & set(nx.ancestors(tree, "b"))
    ac = set(nx.ancestors(tree, "a")) & set(nx.ancestors(tree, "c"))
    assert len(ab) > len(ac)


def test_nj_treedata_default_tree_key(dist_tdata):
    cas.solver.nj(dist_tdata, dissim_key="distances")
    assert "nj" in dist_tdata.obst


def test_upgma_treedata_default_tree_key(dist_tdata):
    cas.solver.upgma(dist_tdata, dissim_key="distances")
    assert "upgma" in dist_tdata.obst


# ── TreeData character path ───────────────────────────────────────────────────


def test_nj_treedata_from_characters():
    tdata = make_chars_tdata(CM)
    cas.solver.nj(tdata, root="outgroup", key_added="nj")
    tree = tdata.obst["nj"]
    assert set(leaves(tree)) == set(CM.index)
    structures = [
        find_triplet_structure(t, tree)
        for t in [("a", "b", "c"), ("a", "b", "d"), ("a", "c", "d"), ("b", "c", "d")]
    ]
    assert sum(s != "-" for s in structures) > 0


def test_upgma_treedata_from_characters():
    tdata = make_chars_tdata(CM)
    cas.solver.upgma(tdata, key_added="upgma")
    tree = tdata.obst["upgma"]
    assert set(leaves(tree)) == set(CM.index)


def test_nj_and_upgma_agree_on_known_groups():
    nj_tdata = make_chars_tdata(CM)
    upgma_tdata = make_chars_tdata(CM)
    cas.solver.nj(nj_tdata, root="outgroup", key_added="nj", dissim_fn="weighted_hamming")
    cas.solver.upgma(upgma_tdata, key_added="upgma", dissim_fn="weighted_hamming")
    nj_topo = nj_tdata.obst["nj"]
    upgma_topo = upgma_tdata.obst["upgma"]
    import itertools

    for triplet in itertools.combinations(["a", "b", "c", "d"], 3):
        nj_s = find_triplet_structure(triplet, nj_topo)
        upgma_s = find_triplet_structure(triplet, upgma_topo)
        if nj_s != "-" and upgma_s != "-":
            assert nj_s == upgma_s


def test_duplicate_sample_treedata():
    tdata = make_chars_tdata(DUPLICATE_CM)
    cas.solver.nj(tdata, root="outgroup", key_added="nj")
    assert set(leaves(tdata.obst["nj"])) == set(DUPLICATE_CM.index)


def test_nj_unknown_root_raises():
    tdata = make_chars_tdata(SMALL_CM)
    with pytest.raises(ValueError):
        cas.solver.nj(tdata, root="nonexistent_procedure")


# ── pairwise() ────────────────────────────────────────────────────────────────


def test_pairwise_treedata():
    tdata = make_chars_tdata(SMALL_CM)
    cas.dissimilarity.pairwise(tdata, key_added="distances")
    dm = tdata.obsp["distances"]
    n = SMALL_CM.shape[0]
    assert dm.shape == (n, n)
    assert np.allclose(np.diag(dm), 0)
    assert np.allclose(dm, dm.T)


def test_pairwise_rejects_non_treedata():
    with pytest.raises(TypeError):
        cas.dissimilarity.pairwise(object())


# ── save_dissim ───────────────────────────────────────────────────────────────


def test_nj_save_dissim_default_off():
    tdata = make_chars_tdata(SMALL_CM)
    cas.solver.nj(tdata, root="outgroup", key_added="nj")
    assert len(tdata.obsp) == 0


def test_nj_save_dissim_excludes_synthetic_outgroup():
    tdata = make_chars_tdata(SMALL_CM)
    cas.solver.nj(tdata, root="outgroup", save_dissim=True, dissim_key="d", key_added="nj")
    dm = tdata.obsp["d"]
    n = SMALL_CM.shape[0]
    # the synthetic 'root' outgroup is excluded from the saved matrix
    assert dm.shape == (n, n)


def test_upgma_save_dissim():
    tdata = make_chars_tdata(SMALL_CM)
    cas.solver.upgma(tdata, save_dissim=True, key_added="upgma")
    assert "distances" in tdata.obsp
    assert tdata.obsp["distances"].shape == (SMALL_CM.shape[0], SMALL_CM.shape[0])


def test_nj_copy_returns_new_and_leaves_original():
    tdata = make_chars_tdata(CM)
    out = cas.solver.nj(tdata, root="outgroup", key_added="nj", copy=True)
    assert isinstance(out, td.TreeData)
    assert "nj" in out.obst
    assert "nj" not in tdata.obst
    assert cas.solver.nj(tdata, root="outgroup", key_added="nj") is None
    assert "nj" in tdata.obst


# ── Rooting registry ──────────────────────────────────────────────────────────


def test_register_custom_procedure():
    @cas.solver.rooting.register("test_first_obs")
    def _first_obs(graph, **kwargs):
        root = list(graph.nodes)[0]
        rooted = nx.DiGraph()
        for e in nx.dfs_edges(graph, source=root):
            rooted.add_edge(e[0], e[1])
        return rooted

    assert "test_first_obs" in cas.solver.rooting._PROCEDURES


def test_outgroup_available():
    assert "outgroup" in cas.solver.rooting._PROCEDURES


# ── Backward-compat shim deprecation ──────────────────────────────────────────


def test_deprecated_implementation_kwarg_nj():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cas.solver.NeighborJoiningSolver(add_root=True, implementation="ccphylo_dnj")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_deprecated_implementation_kwarg_upgma():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cas.solver.UPGMASolver(implementation="ccphylo_upgma")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_nj_solver_fast_false_raises():
    with pytest.raises(NotImplementedError):
        cas.solver.NeighborJoiningSolver(fast=False)


def test_upgma_solver_fast_false_raises():
    with pytest.raises(NotImplementedError):
        cas.solver.UPGMASolver(fast=False)


# ── depth annotation ──────────────────────────────────────────────────────────


def test_solvers_annotate_depth():
    tdata = make_chars_tdata(SMALL_CM)
    cas.solver.nj(tdata, root="outgroup", key_added="nj")
    g = tdata.obst["nj"]
    assert all("depth" in g.nodes[n] for n in g.nodes)
    root = [n for n in g if g.in_degree(n) == 0][0]
    assert g.nodes[root]["depth"] == 0
    # a child of the root has depth 1
    assert all(g.nodes[c]["depth"] == 1 for c in g.successors(root))

    cas.solver.greedy(tdata, key_added="greedy", priors=False)
    gg = tdata.obst["greedy"]
    assert all("depth" in gg.nodes[n] for n in gg.nodes)
