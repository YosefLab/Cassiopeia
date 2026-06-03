"""Tests for the deprecated (removed) solver stubs.

These solvers were removed in the functional refactor. They remain importable
but warn on instantiation and raise on solve.
"""

import warnings

import pytest

import cassiopeia as cas

DEPRECATED_SOLVERS = [
    "MaxCutSolver",
    "MaxCutGreedySolver",
    "SpectralSolver",
    "SpectralGreedySolver",
    "SharedMutationJoiningSolver",
    "PercolationSolver",
    "SpectralNeighborJoiningSolver",
]


@pytest.mark.parametrize("name", DEPRECATED_SOLVERS)
def test_deprecated_solver_warns_on_init(name):
    cls = getattr(cas.solver, name)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls()
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.parametrize("name", DEPRECATED_SOLVERS)
def test_deprecated_solver_solve_raises(name):
    cls = getattr(cas.solver, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = cls()
    with pytest.raises(NotImplementedError):
        solver.solve(None)
