"""Deprecated data utilities.

The CassiopeiaTree-era helpers in :mod:`cassiopeia.data` are deprecated in favor
of :class:`~treedata.TreeData`-based functions in Cassiopeia, `treedata
<https://treedata.readthedocs.io/>`_ and `Pycea <https://pycea.readthedocs.io/>`_.

Functions with a replacement in Cassiopeia emit a :class:`DeprecationWarning`
and delegate. Removed functions raise :class:`NotImplementedError` pointing to
the replacement in TreeData or Pycea, where one exists.
"""

from __future__ import annotations

import warnings

import networkx as nx


def _warn_deprecated(old: str, advice: str) -> None:
    warnings.warn(
        f"{old}() is deprecated and will be removed in a future release. {advice}",
        DeprecationWarning,
        stacklevel=3,
    )


# ── Warn + delegate ──────────────────────────────────────────────────────────


def get_lca_characters(
    vecs: list[list[int] | list[tuple[int, ...]]],
    missing_state_indicator: int,
) -> list[int]:
    """Deprecated. Use :func:`cassiopeia.tl.ancestral_characters` instead."""
    _warn_deprecated(
        "get_lca_characters", "Use cassiopeia.tl.ancestral_characters() with TreeData instead."
    )
    from cassiopeia.tools.ancestral_characters import _get_lca_characters

    return _get_lca_characters(vecs, {missing_state_indicator}, 0)


def newick_to_networkx(newick_string: str) -> nx.DiGraph:
    """Deprecated. Use :func:`treedata.io.read_nwk` instead."""
    _warn_deprecated("newick_to_networkx", "Use treedata.io.read_nwk() instead.")
    from cassiopeia.data.CassiopeiaTree import _newick_to_networkx

    return _newick_to_networkx(newick_string)


# ── Removed ──────────────────────────────────────────────────────────────────


def compute_phylogenetic_weight_matrix(*args, **kwargs):
    """Removed. Use :func:`pycea.tl.tree_distance` instead."""
    raise NotImplementedError(
        "compute_phylogenetic_weight_matrix() required CassiopeiaTree and was removed in "
        "v3.0.0. Use pycea.tl.tree_distance() for distances between leaves on the tree, or "
        "cassiopeia.dissimilarity.pairwise() for character-based dissimilarities."
    )


def net_relatedness_index(*args, **kwargs):
    """Removed. Use :func:`pycea.tl.ancestral_linkage` instead."""
    raise NotImplementedError(
        "net_relatedness_index() was removed in v3.0.0. Use pycea.tl.ancestral_linkage() "
        "(e.g. aggregate='mean') to compute distances between groups of cells."
    )


def compute_inter_cluster_distances(*args, **kwargs):
    """Removed. Use :func:`pycea.tl.ancestral_linkage` instead."""
    raise NotImplementedError(
        "compute_inter_cluster_distances() required CassiopeiaTree and was removed in "
        "v3.0.0. Use pycea.tl.ancestral_linkage() to compute distances between "
        "categories of cells."
    )


def to_newick(*args, **kwargs):
    """Removed. Use :func:`treedata.io.write_nwk` instead."""
    raise NotImplementedError(
        "to_newick() was removed in v3.0.0. Use treedata.io.write_nwk() instead."
    )


def sample_bootstrap_character_matrices(*args, **kwargs):
    """Removed without replacement."""
    raise NotImplementedError(
        "sample_bootstrap_character_matrices() was removed in v3.0.0. Install "
        "cassiopeia v2 for the legacy implementation."
    )


def sample_bootstrap_allele_tables(*args, **kwargs):
    """Removed without replacement."""
    raise NotImplementedError(
        "sample_bootstrap_allele_tables() was removed in v3.0.0. Install "
        "cassiopeia v2 for the legacy implementation."
    )
