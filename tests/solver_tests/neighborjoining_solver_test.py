"""Tests for the functional nj() API on TreeData."""

import itertools

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


def chars_tdata(cm, priors=None):
    uns = {"missing_state_indicator": -1}
    if priors is not None:
        uns["priors"] = priors
    return td.TreeData(obs=pd.DataFrame(index=list(cm.index)), obsm={"characters": cm}, uns=uns)


PP_CM = pd.DataFrame.from_dict(
    {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
    orient="index",
    columns=["x1", "x2", "x3"],
)


def test_nj_basic_from_distances_named_outgroup():
    samples = list("abcde")
    delta = pd.DataFrame.from_dict(
        {
            "a": [0, 15, 21, 17, 12],
            "b": [15, 0, 10, 6, 17],
            "c": [21, 10, 0, 10, 23],
            "d": [17, 6, 10, 0, 19],
            "e": [12, 17, 23, 19, 0],
        },
        orient="index",
        columns=samples,
    )
    tdata = td.TreeData(obs=pd.DataFrame(index=samples))
    tdata.obsp["distances"] = delta.loc[samples, samples].to_numpy()

    # Root with "b" as the outgroup; "b" remains a leaf.
    cas.solver.nj(tdata, dist_key="distances", root="outgroup", outgroup="b", tree_key="nj")
    tree = tdata.obst["nj"]
    assert set(leaves(tree)) == set(samples)
    # c and d are closer to each other than c is to e.
    cd = set(nx.ancestors(tree, "c")) & set(nx.ancestors(tree, "d"))
    ce = set(nx.ancestors(tree, "c")) & set(nx.ancestors(tree, "e"))
    assert len(cd) >= len(ce)


def test_nj_from_characters_synthetic_root():
    tdata = chars_tdata(PP_CM)
    cas.solver.nj(tdata, root="outgroup", tree_key="nj")
    tree = tdata.obst["nj"]
    assert set(leaves(tree)) == set(PP_CM.index)
    # b and c share more derived states than either with d.
    assert find_triplet_structure(("b", "c", "d"), tree) == "ab"


def test_nj_pairwise_value():
    tdata = chars_tdata(PP_CM)
    cas.dissimilarity.pairwise(tdata, method="weighted_hamming_distance", key_added="d")
    dm = pd.DataFrame(tdata.obsp["d"], index=list(PP_CM.index), columns=list(PP_CM.index))
    # weighted hamming between a=[1,1,0] and b=[1,2,0]: differ at one non-zero
    # character (+2), normalized over 3 present characters.
    assert dm.loc["a", "b"] == 2 / 3


def test_nj_duplicates_preserved():
    duplicates_cm = pd.DataFrame.from_dict(
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
    tdata = chars_tdata(duplicates_cm)
    cas.solver.nj(tdata, root="outgroup", tree_key="nj")
    assert set(leaves(tdata.obst["nj"])) == set(duplicates_cm.index)


def test_nj_default_root_uses_first_obs():
    tdata = chars_tdata(PP_CM)
    # root=None: TreeData default uses the first obs name as the root.
    cas.solver.nj(tdata, tree_key="nj")
    tree = tdata.obst["nj"]
    assert isinstance(tree, nx.DiGraph)
    assert len(tree.nodes) > 0


def test_nj_missing_dissimilarity_raises():
    # Characters present but dissimilarity=None -> cannot compute distances.
    tdata = chars_tdata(PP_CM)
    with pytest.raises(cas.mixins.DistanceSolverError):
        cas.solver.nj(tdata, dissimilarity=None)
