"""Deprecated module: dissimilarity metric functions.

These functions have moved to :mod:`cassiopeia.dissimilarity` (``cas.dissimilarity``).
This module re-exports them for backward compatibility and will be removed in a
future release.
"""

from cassiopeia.dissimilarity import (  # noqa: F401
    cluster_dissimilarity,
    cluster_dissimilarity_weighted_hamming_distance_min_linkage,
    exponential_negative_hamming_distance,
    hamming_distance,
    hamming_similarity_normalized_over_missing,
    hamming_similarity_without_missing,
    weighted_hamming_distance,
    weighted_hamming_similarity,
)

__all__ = [
    "cluster_dissimilarity",
    "cluster_dissimilarity_weighted_hamming_distance_min_linkage",
    "exponential_negative_hamming_distance",
    "hamming_distance",
    "hamming_similarity_normalized_over_missing",
    "hamming_similarity_without_missing",
    "weighted_hamming_distance",
    "weighted_hamming_similarity",
]
