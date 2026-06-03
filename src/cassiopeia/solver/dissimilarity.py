"""Deprecated module: dissimilarity computation for tree solvers.

The dissimilarity computation utilities have moved to
:mod:`cassiopeia.dissimilarity` (``cas.dissimilarity``).  The public
:func:`dissimilarity` function is superseded by
:func:`cassiopeia.dissimilarity.pairwise`.  This module re-exports the internals
for backward compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from cassiopeia.dissimilarity import (  # noqa: F401
    _pairwise,
    _resolve_dissimilarity,
    pairwise,
)

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def dissimilarity(
    tdata: CassiopeiaTree | TreeData,
    method: str | Callable = "weighted_hamming_distance",
    characters_key: str | None = None,
    key_added: str = "distances",
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> None:
    """Deprecated. Use :func:`cassiopeia.dissimilarity.pairwise` instead."""
    warnings.warn(
        "cassiopeia.solver.dissimilarity() is deprecated and will be removed in a "
        "future release. Use cassiopeia.dissimilarity.pairwise() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pairwise(
        tdata,
        method=method,
        characters_key=characters_key,
        key_added=key_added,
        prior_transformation=prior_transformation,
        threads=threads,
    )
