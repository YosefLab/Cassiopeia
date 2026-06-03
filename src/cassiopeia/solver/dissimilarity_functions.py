"""Deprecated module: dissimilarity metric functions.

These functions have moved to :mod:`cassiopeia.dissimilarity` (``cas.dissimilarity``).
This module re-exports them for backward compatibility and will be removed in a
future release.
"""

import warnings

from cassiopeia.dissimilarity import (
    cluster_dissimilarity,
    cluster_weighted_hamming,
    hamming,
    nonmissing_hamming,
    weighted_hamming,
)

warnings.warn(
    "cassiopeia.solver.dissimilarity_functions is deprecated; "
    "use cassiopeia.dissimilarity instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "cluster_dissimilarity",
    "cluster_weighted_hamming",
    "hamming",
    "nonmissing_hamming",
    "weighted_hamming",
]
