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

# 2.x names of the metrics that were renamed.
weighted_hamming_distance = weighted_hamming
hamming_distance = hamming
cluster_dissimilarity_weighted_hamming_distance_min_linkage = cluster_weighted_hamming

_REMOVED = (
    "hamming_similarity_without_missing",
    "hamming_similarity_normalized_over_missing",
    "weighted_hamming_similarity",
    "exponential_negative_hamming_distance",
)


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(
            f"{name} was removed in v3.0.0 with the similarity-based solvers that used "
            "it. Use a dissimilarity from cassiopeia.dissimilarity (e.g. "
            "nonmissing_hamming or weighted_hamming)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "cluster_dissimilarity",
    "cluster_dissimilarity_weighted_hamming_distance_min_linkage",
    "cluster_weighted_hamming",
    "hamming",
    "hamming_distance",
    "nonmissing_hamming",
    "weighted_hamming",
    "weighted_hamming_distance",
]
