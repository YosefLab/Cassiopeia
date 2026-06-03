"""Tests for the functional ilp() API, its helpers, and the ILPSolver shim.

Full ILP solves require Gurobi and are gated behind ``GUROBI_INSTALLED``. The
potential-graph and post-processing helpers (Gurobi-free) are tested directly.
"""

import importlib.util
import itertools
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.mixins import ILPSolverError
from cassiopeia.solver import ilp_solver_utilities
from cassiopeia.solver.ilp import (
    _append_sample_names_and_remove_spurious_leaves,
    _infer_potential_graph,
    _post_process_steiner_solution,
)

GUROBI_INSTALLED = importlib.util.find_spec("gurobipy") is not None


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


def chars_tdata(cm):
    return td.TreeData(
        obs=pd.DataFrame(index=list(cm.index)),
        obsm={"characters": cm},
        uns={"missing_state_indicator": -1},
    )


PP_CM = pd.DataFrame.from_dict(
    {"a": [1, 1, 0], "b": [1, 2, 0], "c": [1, 2, 1], "d": [2, 0, 0], "e": [2, 0, 2]},
    orient="index",
    columns=["x1", "x2", "x3"],
)

DUPLICATES_CM = pd.DataFrame.from_dict(
    {
        "a": [1, 1, 0],
        "b": [1, 2, 0],
        "c": [1, 2, 1],
        "d": [2, 0, 0],
        "e": [2, 0, 2],
        "f": [1, 1, 0],
    },
    orient="index",
    columns=["x1", "x2", "x3"],
)

MISSING_CM = pd.DataFrame.from_dict(
    {
        "a": [1, 3, 1, 1],
        "b": [1, 3, 1, -1],
        "c": [1, 0, 1, 0],
        "d": [1, 1, 3, 0],
        "e": [1, 1, 0, 0],
        "f": [2, 0, 0, 0],
        "g": [2, 4, -1, -1],
        "h": [2, 4, 2, 0],
    },
    orient="index",
)


# ── Cython utilities ──────────────────────────────────────────────────────────


def test_get_lca_cython():
    cm = MISSING_CM.astype(str)
    lca = ilp_solver_utilities.get_lca_characters_cython(
        cm.loc["a"].values, cm.loc["b"].values, 4, "-1"
    )
    assert lca == "1|3|1|1"
    lca = ilp_solver_utilities.get_lca_characters_cython(
        cm.loc["h"].values, cm.loc["b"].values, 4, "-1"
    )
    assert lca == "0|0|0|0"


def test_cython_hamming_dist():
    s1 = np.array(["1", "2", "3", "0", "0"])
    s2 = np.array(["1", "4", "0", "0", "1"])
    assert ilp_solver_utilities.simple_hamming_distance_cython(s1, s2, "-") == 3

    s1 = np.array(["1", "2", "3", "0", "-"])
    s2 = np.array(["1", "-", "0", "0", "1"])
    assert ilp_solver_utilities.simple_hamming_distance_cython(s1, s2, "-") == 1


def test_get_layer_for_potential_graph():
    source_nodes = PP_CM.drop_duplicates().values
    dim = source_nodes.shape[1]
    source_node_strings = np.array(["|".join(arr) for arr in source_nodes.astype(str)])
    layer_nodes, layer_edges = ilp_solver_utilities.infer_layer_of_potential_graph(
        source_node_strings, 10
    )

    layer_nodes = np.unique(np.array([node.split("|") for node in layer_nodes], dtype=int), axis=0)
    for sample in np.array([[1, 0, 0], [1, 2, 0], [0, 0, 0], [2, 0, 0]]):
        assert sample in layer_nodes

    layer_edges = np.array([edge.split("|") for edge in layer_edges], dtype=int)
    layer_edges = [(list(e[:dim]), list(e[dim:])) for e in layer_edges]
    expected_edges = [
        ([1, 0, 0], [1, 1, 0]),
        ([1, 0, 0], [1, 2, 0]),
        ([1, 0, 0], [1, 2, 1]),
        ([1, 2, 0], [1, 2, 0]),
        ([1, 2, 0], [1, 2, 1]),
        ([0, 0, 0], [1, 1, 0]),
        ([0, 0, 0], [1, 2, 0]),
        ([0, 0, 0], [1, 2, 1]),
        ([0, 0, 0], [2, 0, 0]),
        ([0, 0, 0], [2, 0, 2]),
        ([2, 0, 0], [2, 0, 0]),
        ([2, 0, 0], [2, 0, 2]),
    ]
    for edge in expected_edges:
        assert edge in layer_edges
    uniq = []
    for edge in layer_edges:
        if edge not in uniq:
            uniq.append(edge)
    assert len(uniq) == len(expected_edges)


# ── Potential graph inference ─────────────────────────────────────────────────


@pytest.mark.parametrize("cm", [PP_CM, DUPLICATES_CM], ids=["simple", "duplicates"])
def test_potential_graph_inference(cm):
    unique = cm.drop_duplicates()
    potential_graph = _infer_potential_graph(unique, 0, 10, 10000, None, -1)

    expected_nodes = [
        (1, 1, 0),
        (1, 2, 0),
        (1, 2, 1),
        (2, 0, 0),
        (2, 0, 2),
        (1, 0, 0),
        (0, 0, 0),
    ]
    for node in expected_nodes:
        assert node in potential_graph.nodes()

    expected_edges = [
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 2, 0)),
        ((1, 0, 0), (1, 2, 1)),
        ((1, 2, 0), (1, 2, 1)),
        ((0, 0, 0), (1, 1, 0)),
        ((0, 0, 0), (1, 2, 0)),
        ((0, 0, 0), (1, 2, 1)),
        ((0, 0, 0), (2, 0, 0)),
        ((0, 0, 0), (2, 0, 2)),
        ((2, 0, 0), (2, 0, 2)),
        ((0, 0, 0), (1, 0, 0)),
    ]
    for edge in expected_edges:
        assert edge in potential_graph.edges()
    assert len(potential_graph.edges()) == len(expected_edges)


def test_post_process_steiner_solution():
    tree = nx.DiGraph()
    tree.add_weighted_edges_from(
        [
            (6, "c1", 1),
            (6, "c2", 1),
            (8, "c3", 1),
            (8, "c4", 1),
            (7, "c5", 1),
            (7, "c6", 1),
            (7, "c7", 1),
            (8, 6, 1),
            (9, 7, 1),
            (9, 8, 1),
            (10, "c4", 1.5),
            (11, 10, 1.5),
            (9, 6, 1.5),
            (8, "c2", 0.5),
        ]
    )
    processed = _post_process_steiner_solution(tree, 9)
    expected = nx.DiGraph()
    expected.add_weighted_edges_from(
        [
            (6, "c1", 1),
            (6, "c2", 1),
            (8, "c3", 1),
            (8, "c4", 1),
            (7, "c5", 1),
            (7, "c6", 1),
            (7, "c7", 1),
            (8, 6, 1),
            (9, 7, 1),
            (9, 8, 1),
        ]
    )
    assert set(processed.edges) == set(expected.edges)


def test_append_sample_nodes_and_remove_spurious_leaves():
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (2, 0, 0)),
            ((2, 0, 0), (2, 0, 2)),
            ((2, 0, 0), (2, 0, 1)),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 2, 0)),
            ((1, 2, 0), (1, 2, 1)),
            ((1, 2, 0), (1, 2, 2)),
            ((2, 0, 1), (2, 1, 1)),
            ((2, 0, 1), (2, 2, 1)),
            ((2, 0, 1), (2, 3, 1)),
            ((1, 0, 0), (1, 1, 2)),
            ((1, 0, 0), (1, 1, 1)),
        ]
    )
    processed = _append_sample_names_and_remove_spurious_leaves(tree, DUPLICATES_CM)
    expected = nx.DiGraph()
    expected.add_edges_from(
        [
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (2, 0, 0)),
            ((2, 0, 0), "d"),
            ((2, 0, 0), (2, 0, 2)),
            ((2, 0, 2), "e"),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 1, 0), "a"),
            ((1, 1, 0), "f"),
            ((1, 0, 0), (1, 2, 0)),
            ((1, 2, 0), "b"),
            ((1, 2, 0), (1, 2, 1)),
            ((1, 2, 1), "c"),
        ]
    )
    assert set(processed.edges) == set(expected.edges)


# ── Functional ilp() (Gurobi-free paths) ──────────────────────────────────────


def test_ilp_raises_on_ambiguous():
    cm = pd.DataFrame.from_dict(
        {
            "c1": [5, (0, 1), 1, 2, -1],
            "c2": [0, 0, 3, 2, -1],
            "c3": [-1, 4, 0, 2, 2],
            "c4": [4, 4, 1, 2, 0],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )
    tdata = chars_tdata(cm)
    with pytest.raises(ILPSolverError):
        cas.solver.ilp(tdata)


def test_ilp_single_sample():
    # A single unique state needs no ILP optimization (no Gurobi required).
    cm = pd.DataFrame([[1], [1], [1]], index=["a", "b", "c"], columns=["x1"])
    tdata = chars_tdata(cm)
    cas.solver.ilp(tdata, tree_key="ilp")
    assert set(leaves(tdata.obst["ilp"])) == {"a", "b", "c"}


# ── Gurobi-gated full solves ──────────────────────────────────────────────────


@pytest.mark.skipif(not GUROBI_INSTALLED, reason="Gurobi installation not found.")
def test_ilp_perfect_phylogeny():
    tdata = chars_tdata(PP_CM)
    cas.solver.ilp(tdata, mip_gap=0.0, tree_key="ilp")
    tree = tdata.obst["ilp"]
    assert len(roots := [n for n in tree if tree.in_degree(n) == 0]) == 1
    assert set(leaves(tree)) == set(PP_CM.index)
    assert [n for n in tree if tree.in_degree(n) > 1] == []

    expected = nx.DiGraph()
    expected.add_edges_from(
        [
            ("root", "9"),
            ("9", "8"),
            ("9", "7"),
            ("7", "6"),
            ("7", "a"),
            ("6", "b"),
            ("6", "c"),
            ("8", "e"),
            ("8", "d"),
        ]
    )
    for triplet in itertools.combinations(["a", "b", "c", "d", "e"], 3):
        assert find_triplet_structure(triplet, tree) == find_triplet_structure(triplet, expected)


@pytest.mark.skipif(not GUROBI_INSTALLED, reason="Gurobi installation not found.")
def test_ilp_missing_data():
    tdata = chars_tdata(MISSING_CM)
    cas.solver.ilp(tdata, mip_gap=0.0, tree_key="ilp")
    tree = tdata.obst["ilp"]
    assert len([n for n in tree if tree.in_degree(n) == 0]) == 1
    assert set(leaves(tree)) == set(MISSING_CM.index)


@pytest.mark.skipif(not GUROBI_INSTALLED, reason="Gurobi installation not found.")
def test_ilp_potential_graph_not_found_raises():
    tdata = chars_tdata(MISSING_CM)
    with pytest.raises(ILPSolverError):
        cas.solver.ilp(tdata, maximum_potential_graph_layer_size=3)


# ── Backward-compat shim ──────────────────────────────────────────────────────


def test_ilp_solver_deprecated_and_stores_params():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = cas.solver.ILPSolver(mip_gap=0.0)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert solver.convergence_time_limit == 12600
    assert solver.maximum_potential_graph_layer_size == 10000
    assert solver.weighted is False
