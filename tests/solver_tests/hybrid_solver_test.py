"""Tests for the functional hybrid() API on TreeData.

The functional ``hybrid`` composes a top-down split callable with a bottom
solver callable. These tests use the (Gurobi-free) functional ``greedy`` as the
bottom solver; ILP-bottom behavior is covered behind a Gurobi skip.
"""

import functools
import importlib.util
import warnings

import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas

GUROBI_INSTALLED = importlib.util.find_spec("gurobipy") is not None


def leaves(g):
    return [n for n in g.nodes if g.out_degree(n) == 0]


def roots(g):
    return [n for n in g.nodes if g.in_degree(n) == 0]


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

LARGE_CM = pd.DataFrame.from_dict(
    {
        "a": [1, 0, 0, 0, 0, 0, 0, 0],
        "b": [1, 1, 0, 0, 0, 0, 0, 0],
        "c": [1, 1, 1, 0, 0, 0, 0, 0],
        "d": [1, 1, 1, 1, 0, 0, 0, 0],
        "e": [1, 1, 1, 1, 1, 0, 0, 0],
        "f": [1, 1, 1, 1, 1, 1, 0, 0],
        "g": [1, 1, 1, 1, 1, 1, 1, 0],
        "h": [1, 1, 1, 1, 1, 1, 1, 1],
        "i": [2, 0, 0, 0, 0, 0, 0, 0],
        "j": [2, 2, 0, 0, 0, 0, 0, 0],
    },
    orient="index",
)


# ── Functional hybrid with a greedy bottom solver ─────────────────────────────


def test_hybrid_cell_cutoff_greedy_bottom():
    tdata = chars_tdata(PP_CM)
    cas.solver.hybrid(
        tdata,
        bottom_solver=functools.partial(cas.solver.greedy),
        cell_cutoff=3,
        progress_bar=False,
        tree_key="hybrid",
    )
    tree = tdata.obst["hybrid"]
    assert set(leaves(tree)) == set(PP_CM.index)
    assert len(roots(tree)) == 1


def test_hybrid_lca_cutoff_greedy_bottom():
    tdata = chars_tdata(LARGE_CM)
    cas.solver.hybrid(
        tdata,
        bottom_solver=functools.partial(cas.solver.greedy),
        lca_cutoff=2,
        progress_bar=False,
        tree_key="hybrid",
    )
    tree = tdata.obst["hybrid"]
    assert set(leaves(tree)) == set(LARGE_CM.index)
    assert len(roots(tree)) == 1


def test_hybrid_multithreaded_pickles():
    tdata = chars_tdata(LARGE_CM)
    cas.solver.hybrid(
        tdata,
        bottom_solver=functools.partial(cas.solver.greedy),
        cell_cutoff=3,
        threads=2,
        progress_bar=False,
        tree_key="hybrid",
    )
    tree = tdata.obst["hybrid"]
    assert set(leaves(tree)) == set(LARGE_CM.index)
    assert len(roots(tree)) == 1


def test_hybrid_custom_top_solver():
    # Explicitly pass the greedy split as the top solver.
    from cassiopeia.solver.greedy import _greedy_split

    tdata = chars_tdata(PP_CM)
    cas.solver.hybrid(
        tdata,
        top_solver=_greedy_split,
        bottom_solver=functools.partial(cas.solver.greedy),
        cell_cutoff=2,
        progress_bar=False,
    )
    assert set(leaves(tdata.obst["hybrid"])) == set(PP_CM.index)


# ── Errors ────────────────────────────────────────────────────────────────────


def test_hybrid_requires_cutoff():
    tdata = chars_tdata(PP_CM)
    with pytest.raises(cas.mixins.HybridSolverError):
        cas.solver.hybrid(tdata, bottom_solver=functools.partial(cas.solver.greedy))


def test_hybrid_requires_bottom_solver():
    tdata = chars_tdata(PP_CM)
    with pytest.raises(cas.mixins.HybridSolverError):
        cas.solver.hybrid(tdata, cell_cutoff=3)


# ── Gurobi-gated ILP bottom solver ────────────────────────────────────────────


@pytest.mark.skipif(not GUROBI_INSTALLED, reason="Gurobi installation not found.")
def test_hybrid_ilp_bottom():
    tdata = chars_tdata(PP_CM)
    cas.solver.hybrid(
        tdata,
        bottom_solver=functools.partial(cas.solver.ilp, mip_gap=0.0, logfile=None),
        cell_cutoff=3,
        progress_bar=False,
        tree_key="hybrid",
    )
    assert set(leaves(tdata.obst["hybrid"])) == set(PP_CM.index)


# ── Backward-compat shim ──────────────────────────────────────────────────────


def test_hybrid_solver_deprecated_and_solves():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = cas.solver.HybridSolver(
            cas.solver.VanillaGreedySolver(),
            cas.solver.VanillaGreedySolver(),
            cell_cutoff=3,
        )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    # The shim adapts solver instances to callables and delegates to hybrid().
    tdata = chars_tdata(PP_CM)
    solver.solve(tdata)
    assert set(leaves(tdata.obst["hybrid"])) == set(PP_CM.index)
