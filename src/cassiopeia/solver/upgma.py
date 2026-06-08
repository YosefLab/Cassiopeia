"""UPGMA solver: functional API and UPGMASolver shim."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia import dissimilarity as dissimilarity_functions
from cassiopeia.dissimilarity import _pairwise, _resolve_dissimilarity, _uses_weights
from cassiopeia.utils import (
    _get_characters,
    _get_parameter,
    _node_name_generator,
    _resolve_priors,
    _save_dissimilarity,
    _set_tree,
)

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def _build_graph(
    dissimilarity_map: pd.DataFrame,
    node_name_generator: Generator[str, None, None],
) -> nx.DiGraph:
    """Run Cython UPGMA and return a complete rooted directed tree.

    Args:
        dissimilarity_map: Symmetric n×n distance DataFrame.
        node_name_generator: Generator supplying unique names for internal nodes.

    Returns:
        A complete rooted :class:`~networkx.DiGraph`.
    """
    from cassiopeia.solver import nj_solver_utilities

    sample_names = list(dissimilarity_map.index)
    n = len(sample_names)
    D = np.ascontiguousarray(dissimilarity_map.to_numpy(dtype=np.float64))
    parents, children, _, node_a, node_b = nj_solver_utilities.upgma(D)

    id_to_name: dict[int, str] = {i: sample_names[i] for i in range(n)}
    for k in range(n - 2):
        id_to_name[n + k] = next(node_name_generator)

    graph = nx.Graph()
    for name in sample_names:
        graph.add_node(name)
    for p, c in zip(parents, children, strict=False):
        graph.add_edge(id_to_name[int(p)], id_to_name[int(c)])

    # Add root node above the two remaining unmerged nodes
    root_node = next(node_name_generator)
    graph.add_node(root_node)
    graph.add_edge(root_node, id_to_name[node_a])
    graph.add_edge(root_node, id_to_name[node_b])

    rooted = nx.DiGraph()
    for e in nx.dfs_edges(graph, source=root_node):
        rooted.add_edge(e[0], e[1])
    return rooted


def upgma(
    tdata: TreeData,
    dissim_key: str | None = None,
    dissim_fn: str | Callable | None = "nonmissing_hamming",
    save_dissim: bool = False,
    characters_key: str | None = None,
    key_added: str = "upgma",
    prior_transformation: str = "negative_log",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict[int, dict[int, float]] | bool = False,
    threads: int = 1,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct a tree with UPGMA (Cython, O(n²) average case).

    Builds the tree from a precomputed dissimilarity map in ``tdata.obsp`` or,
    when none is available, from the character matrix in ``tdata.obsm``. The
    result is stored as an ``nx.DiGraph`` in ``tdata.obst[key_added]``.

    Args:
        tdata: TreeData to operate on.
        dissim_key: Key in ``tdata.obsp`` for a precomputed dissimilarity map.
        dissim_fn: Dissimilarity function used when distances are not
            precomputed. Accepts a callable or a string name of a built-in
            metric in :mod:`cassiopeia.dissimilarity`.
        save_dissim: Whether to store the computed dissimilarity map in
            ``tdata.obsp[dissim_key or 'distances']``.
        characters_key: Key in ``tdata.obsm`` for the character matrix
            (default ``'characters'``).
        key_added: Key in ``tdata.obst`` for the resulting tree.
        prior_transformation: Transformation applied to priors to form weights.
        missing_state: Missing-state value (read from ``tdata.uns`` if ``None``).
        unmodified_state: Unmodified/uncut state value (read from ``tdata.uns``
            if ``None``).
        priors: Priors for character states. ``False`` (default) reconstructs
            without priors; ``True`` reads priors from ``tdata.uns["priors"]``
            and raises if none are stored; a dict (character index -> {state:
            probability}) is used directly.
        threads: Threads for parallel dissimilarity computation.
        copy: If ``True``, return a copy of *tdata*; otherwise modify in-place
            and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.
    """
    tdata = tdata.copy() if copy else tdata
    dissimilarity_fn = _resolve_dissimilarity(dissim_fn)

    if dissim_key is None:
        unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
        missing_state = _get_parameter(tdata, "missing_state", value=missing_state)
        characters = _get_characters(tdata, characters_key)
        priors = _resolve_priors(tdata, priors)
        if priors and not _uses_weights(dissim_fn):
            warnings.warn(
                f"priors were provided but the dissimilarity function {dissim_fn!r} "
                "does not use weights, so the priors will be ignored. Use a "
                "weighted metric (e.g. 'weighted_hamming') to make use of priors.",
                UserWarning,
                stacklevel=2,
            )
        dist_df = _pairwise(
            characters,
            dissimilarity_fn,
            missing_state,
            priors,
            prior_transformation,
            threads,
            unmodified_state=unmodified_state,
        )
        if save_dissim:
            _save_dissimilarity(tdata, dist_df, dissim_key)
    else:
        dist_df = pd.DataFrame(
            tdata.obsp[dissim_key], index=tdata.obs_names, columns=tdata.obs_names
        )

    node_gen = _node_name_generator()
    rooted = _build_graph(dist_df, node_gen)

    _set_tree(tdata, rooted, key_added)

    return tdata if copy else None


# ── Backward-compat class wrapper ────────────────────────────────────────────


class UPGMASolver:
    """UPGMA solver for Cassiopeia.

    Thin wrapper around :func:`cassiopeia.solver.upgma` for backward
    compatibility.  Uses the Cython UPGMA implementation (O(n²) average case).

    For new code, prefer calling :func:`cassiopeia.solver.upgma` directly.

    Args:
        dissimilarity_function: Function to compute the dissimilarity map.
            Optional when a precomputed map is already present on the tree.
        prior_transformation: Transformation applied to priors.  Supports
            ``"negative_log"``, ``"inverse"``, ``"square_root_inverse"``.
        fast: Must be ``True``.  The generic (slow) UPGMA path has been removed.
        implementation: Deprecated.  Use ``fast=True`` instead.
        threads: Threads for dissimilarity map computation.
    """

    def __init__(
        self,
        dissimilarity_function: Callable[
            [np.array, np.array, int, dict[int, dict[int, float]]], float
        ]
        | None = dissimilarity_functions.nonmissing_hamming,
        prior_transformation: str = "negative_log",
        fast: bool = True,
        implementation: str | None = None,
        threads: int = 1,
    ):
        if implementation is not None:
            warnings.warn(
                "The 'implementation' parameter is deprecated and will be removed "
                "in a future release. Use fast=True instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fast = True

        if not fast:
            raise NotImplementedError(
                "The generic (slow) UPGMA path has been removed. "
                "Use UPGMASolver(fast=True) or cas.solver.upgma()."
            )

        self._implementation = "upgma_fast"
        self.dissimilarity_function = dissimilarity_function
        self.prior_transformation = prior_transformation
        self.threads = threads

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solve the tree topology using UPGMA.

        Args:
            cassiopeia_tree: CassiopeiaTree to solve in-place.
            layer: Character matrix layer to use.
            collapse_mutationless_edges: Whether to collapse edges with no
                inferred mutations after solving.
            logfile: Ignored (kept for API compatibility).
        """
        # Preserve legacy behavior: use priors if the tree carries them, else not.
        upgma(
            cassiopeia_tree,
            dissim_fn=self.dissimilarity_function,
            characters_key=layer,
            prior_transformation=self.prior_transformation,
            priors=_get_parameter(cassiopeia_tree, "priors") or False,
            save_dissim=True,
            threads=self.threads,
        )
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    def root_tree(self, tree, root_sample, remaining_samples):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "root_tree is removed in favor of the fast Cython implementation. "
            "Use cas.solver.upgma() directly."
        )

    def find_cherry(self, dissimilarity_matrix):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "find_cherry is removed in favor of the fast Cython implementation. "
            "Use cas.solver.upgma() directly."
        )

    def update_dissimilarity_map(self, dissimilarity_map, cherry, new_node):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "update_dissimilarity_map is removed in favor of the fast Cython "
            "implementation. Use cas.solver.upgma() directly."
        )

    def setup_root_finder(self, cassiopeia_tree):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "setup_root_finder is removed in favor of the fast Cython "
            "implementation. Use cas.solver.upgma() directly."
        )
