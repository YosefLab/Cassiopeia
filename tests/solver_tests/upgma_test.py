"""Tests for the functional upgma() API on TreeData."""

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


def assert_triplets_match(observed, expected, samples):
    for triplet in itertools.combinations(samples, 3):
        assert find_triplet_structure(triplet, observed) == find_triplet_structure(
            triplet, expected
        )


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


def test_upgma_basic_from_distances():
    samples = list("abcde")
    delta = pd.DataFrame.from_dict(
        {
            "a": [0, 17, 21, 31, 23],
            "b": [17, 0, 30, 34, 21],
            "c": [21, 30, 0, 28, 39],
            "d": [31, 34, 28, 0, 43],
            "e": [23, 21, 39, 43, 0],
        },
        orient="index",
        columns=samples,
    )
    tdata = td.TreeData(obs=pd.DataFrame(index=samples))
    tdata.obsp["distances"] = delta.loc[samples, samples].to_numpy()

    cas.solver.upgma(tdata, dist_key="distances", tree_key="upgma")
    tree = tdata.obst["upgma"]

    assert set(leaves(tree)) == set(samples)

    expected = nx.DiGraph()
    expected.add_edges_from(
        [("5", "a"), ("5", "b"), ("6", "5"), ("6", "e"), ("7", "c"), ("7", "d"), ("root", "6"), ("root", "7")]
    )
    assert_triplets_match(tree, expected, samples)


def test_upgma_from_characters_no_priors():
    tdata = chars_tdata(PP_CM)
    cas.solver.upgma(tdata, tree_key="upgma")
    tree = tdata.obst["upgma"]

    expected = nx.DiGraph()
    expected.add_edges_from(
        [
            ("root", "8"),
            ("root", "7"),
            ("7", "6"),
            ("7", "a"),
            ("6", "b"),
            ("6", "c"),
            ("8", "e"),
            ("8", "d"),
        ]
    )
    assert_triplets_match(tree, expected, list("abcde"))


def test_upgma_pairwise_value_no_priors():
    tdata = chars_tdata(PP_CM)
    cas.dissimilarity.pairwise(tdata, key_added="distances")
    dm = pd.DataFrame(tdata.obsp["distances"], index=list(PP_CM.index), columns=list(PP_CM.index))
    assert dm.loc["d", "e"] == 1 / 3


def test_upgma_with_priors_pairwise_value():
    priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
    tdata = chars_tdata(PP_CM, priors=priors)
    cas.dissimilarity.pairwise(tdata, key_added="distances")
    dm = pd.DataFrame(tdata.obsp["distances"], index=list(PP_CM.index), columns=list(PP_CM.index))
    np.testing.assert_almost_equal(dm.loc["a", "b"], (-np.log(0.2) - np.log(0.8)) / 3)


def test_upgma_duplicate_pairwise_value():
    duplicates_cm = pd.DataFrame.from_dict(
        {
            "a": [1, -1, 0],
            "b": [1, 2, 1],
            "c": [1, -1, 1],
            "d": [2, 0, -1],
            "e": [2, 0, 2],
            "f": [2, 0, 2],
        },
        orient="index",
        columns=["x1", "x2", "x3"],
    )
    tdata = chars_tdata(duplicates_cm)
    cas.dissimilarity.pairwise(tdata, key_added="distances")
    dm = pd.DataFrame(
        tdata.obsp["distances"], index=list(duplicates_cm.index), columns=list(duplicates_cm.index)
    )
    assert dm.loc["b", "d"] == 1.5

    cas.solver.upgma(tdata, tree_key="upgma")
    assert set(leaves(tdata.obst["upgma"])) == set(duplicates_cm.index)


def test_upgma_then_collapse_edges():
    priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
    tdata = chars_tdata(PP_CM, priors=priors)
    cas.solver.upgma(tdata, tree_key="upgma")
    before = tdata.obst["upgma"].number_of_nodes()

    cas.tl.ancestral_characters(tdata, tree_key="upgma")
    cas.tl.collapse_edges(tdata, tree_key="upgma")
    tree = tdata.obst["upgma"]

    # Leaves are preserved; collapsing does not increase node count.
    assert set(leaves(tree)) == set(PP_CM.index)
    assert tree.number_of_nodes() <= before


def test_pairwise_with_callable_metric():
    tdata = chars_tdata(PP_CM)
    cas.dissimilarity.pairwise(
        tdata, method=cas.dissimilarity.weighted_hamming_distance, key_added="distances"
    )
    assert tdata.obsp["distances"].shape == (5, 5)
