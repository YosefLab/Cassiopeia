"""Solver tests with string/categorical characters (unmodified state ``"*"``).

Mirrors the data produced by the simulator (``cas.sim.stochastic_tracing``), where
the unmodified/uncut state is the string ``"*"`` and mutated states are string
integers.  The solvers encode such matrices to integers internally
(``"*"`` -> 0) before reconstruction.
"""

import functools
import importlib.util

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas

GUROBI_INSTALLED = importlib.util.find_spec("gurobipy") is not None

# String character matrix with "*" as the unmodified state.
CM = pd.DataFrame.from_dict(
    {
        "a": ["*", "*", "1"],
        "b": ["*", "2", "1"],
        "c": ["3", "2", "1"],
        "d": ["3", "*", "*"],
        "e": ["3", "*", "2"],
    },
    orient="index",
    columns=["x1", "x2", "x3"],
).astype("category")

# The integer twin (unmodified "*" -> 0, states "1"/"2"/"3" -> 1/2/3), matching
# how the solvers encode CM internally.
INT_CM = pd.DataFrame.from_dict(
    {
        "a": [0, 0, 1],
        "b": [0, 2, 1],
        "c": [3, 2, 1],
        "d": [3, 0, 0],
        "e": [3, 0, 2],
    },
    orient="index",
    columns=["x1", "x2", "x3"],
)


def make_tdata(cm=CM, unmodified="*"):
    uns = {"unmodified_state": unmodified} if unmodified is not None else {}
    return td.TreeData(
        obs=pd.DataFrame(index=list(cm.index)), obsm={"characters": cm.copy()}, uns=uns
    )


def leaves(g):
    return sorted(n for n in g.nodes if g.out_degree(n) == 0)


def roots(g):
    return [n for n in g.nodes if g.in_degree(n) == 0]


SOLVERS = [
    pytest.param(lambda t: cas.solver.nj(t, tree_key="t"), id="nj"),
    pytest.param(lambda t: cas.solver.upgma(t, tree_key="t"), id="upgma"),
    pytest.param(lambda t: cas.solver.greedy(t, tree_key="t"), id="greedy"),
    pytest.param(
        lambda t: cas.solver.hybrid(
            t,
            bottom_solver=functools.partial(cas.solver.greedy),
            cell_cutoff=3,
            progress_bar=False,
            tree_key="t",
        ),
        id="hybrid",
    ),
]


@pytest.mark.parametrize("solve", SOLVERS)
def test_solvers_string_characters(solve):
    tdata = make_tdata()
    solve(tdata)
    g = tdata.obst["t"]
    assert nx.is_tree(g)
    assert len(roots(g)) == 1
    assert set(leaves(g)) == set(CM.index)
    assert all("depth" in g.nodes[n] for n in g.nodes)


@pytest.mark.skipif(not GUROBI_INSTALLED, reason="Gurobi installation not found.")
def test_ilp_string_characters():
    tdata = make_tdata()
    cas.solver.ilp(tdata, tree_key="t")
    g = tdata.obst["t"]
    assert len(roots(g)) == 1
    assert set(leaves(g)) == set(CM.index)


def test_solvers_string_match_integer_encoding():
    # nj on the string matrix should match nj on its integer twin (the "*" ->
    # 0 encoding is what makes the two equivalent).
    s = make_tdata()
    i = make_tdata(cm=INT_CM, unmodified=None)
    cas.solver.upgma(s, tree_key="t")
    cas.solver.upgma(i, tree_key="t")

    def triplet(g, a, b, c):
        anc = {x: set(nx.ancestors(g, x)) for x in (a, b, c)}
        ab, ac, bc = (len(anc[a] & anc[b]), len(anc[a] & anc[c]), len(anc[b] & anc[c]))
        return max(("ab", ab), ("ac", ac), ("bc", bc), key=lambda kv: kv[1])[0]

    import itertools

    for t in itertools.combinations("abcde", 3):
        assert triplet(s.obst["t"], *t) == triplet(i.obst["t"], *t)


def test_pairwise_string_unmodified_maps_to_zero():
    # pairwise on the string matrix equals pairwise on its integer twin.
    s = make_tdata()
    i = make_tdata(cm=INT_CM, unmodified=None)
    cas.dissimilarity.pairwise(s, method="weighted_hamming", key_added="d")
    cas.dissimilarity.pairwise(i, method="weighted_hamming", key_added="d")
    assert np.allclose(s.obsp["d"], i.obsp["d"])
