"""Pairwise dissimilarity metrics and computation for phylogenetic samples."""

from ._compute import compute_dissimilarity_map
from ._metrics import (
    _resolve_dissimilarity,
    cluster_dissimilarity,
    cluster_weighted_hamming,
    hamming,
    nonmissing_hamming,
    weighted_hamming,
)
from ._pairwise import _encode_integer_matrix, _pairwise, _prepare_integer_matrix, pairwise

__all__ = [
    "pairwise",
    "compute_dissimilarity_map",
    "weighted_hamming",
    "hamming",
    "nonmissing_hamming",
    "cluster_dissimilarity",
    "cluster_weighted_hamming",
]
