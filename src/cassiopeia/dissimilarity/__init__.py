"""Pairwise dissimilarity metrics and computation for phylogenetic samples."""

from ._metrics import (
    _resolve_dissimilarity,
    _uses_weights,
    cluster_dissimilarity,
    cluster_weighted_hamming,
    hamming,
    nonmissing_hamming,
    weighted_hamming,
)
from ._pairwise import (
    _encode_integer_matrix,
    _encode_priors,
    _pairwise,
    _prepare_integer_matrix,
    pairwise,
)

__all__ = [
    "pairwise",
    "weighted_hamming",
    "hamming",
    "nonmissing_hamming",
    "cluster_dissimilarity",
    "cluster_weighted_hamming",
]
