"""Backward-compatibility re-export.

``VanillaGreedySolver`` now lives in :mod:`cassiopeia.solver.greedy` alongside
the functional :func:`cassiopeia.solver.greedy` entry point.  This module
re-exports it so existing imports continue to work.
"""

from cassiopeia.solver.greedy import VanillaGreedySolver  # noqa: F401

__all__ = ["VanillaGreedySolver"]
