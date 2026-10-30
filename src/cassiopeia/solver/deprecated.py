"""Deprecated solver classes.

These solvers were removed in the 3.0 functional refactor.  They remain as
import-compatible stubs that warn on instantiation and raise on ``solve``,
directing users to the maintained functional solvers.

The removed solvers are:
``MaxCutSolver``, ``MaxCutGreedySolver``, ``SpectralSolver``,
``SpectralGreedySolver``, ``SharedMutationJoiningSolver``, ``PercolationSolver``,
and ``SpectralNeighborJoiningSolver``.
"""

from __future__ import annotations

import warnings

_RECOMMENDATION = (
    "Use cassiopeia.solver.nj() or cassiopeia.solver.greedy(), which more "
    "accurately reconstruct lineage trees."
)


class _DeprecatedSolver:
    """Base for removed solver classes: warns on init, raises on ``solve``."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            f"{type(self).__name__} is deprecated and its implementation has been "
            f"removed. {_RECOMMENDATION}",
            DeprecationWarning,
            stacklevel=2,
        )

    def solve(self, *args, **kwargs):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(f"{type(self).__name__} has been removed. {_RECOMMENDATION}")


class MaxCutSolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class MaxCutGreedySolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class SpectralSolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class SpectralGreedySolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class SharedMutationJoiningSolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class PercolationSolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""


class SpectralNeighborJoiningSolver(_DeprecatedSolver):
    """Deprecated and removed. See :mod:`cassiopeia.solver.deprecated`."""
