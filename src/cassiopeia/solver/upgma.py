"""UPGMA solver: functional API and UPGMASolver shim."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.solver import dissimilarity_functions, solver_utilities
from cassiopeia.solver.dissimilarity import _get_distances, _resolve_dissimilarity

if TYPE_CHECKING:
    from cassiopeia.data import CassiopeiaTree
    from treedata import TreeData


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
    for p, c in zip(parents, children):
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
    tdata: CassiopeiaTree | TreeData,
    dist_key: str | None = None,
    dissimilarity: str | Callable | None = "weighted_hamming_distance",
    characters_key: str | None = None,
    tree_key: str = "upgma",
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> None:
    """UPGMA with O(n²) Cython implementation. Modifies tdata in-place.

    Produces an ultrametric tree.  UPGMA is self-rooting; no ``root`` parameter
    is needed.

    For :class:`~cassiopeia.data.CassiopeiaTree`: populates tree topology via
    ``populate_tree()``.  For :class:`~treedata.TreeData`: stores the result
    ``nx.DiGraph`` in ``tdata.obst[tree_key]``.

    Args:
        tdata: CassiopeiaTree or TreeData to solve.
        dist_key: Key in ``tdata.obsp`` for precomputed distances (TreeData only).
        dissimilarity: Function used when distances are not precomputed.
            Accepts a callable or a string name of a function in
            :mod:`cassiopeia.solver.dissimilarity_functions`.
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm`` key
            (TreeData, default ``'characters'``).
        tree_key: Key in ``tdata.obst`` for the result (TreeData only).
        prior_transformation: Transformation applied to priors when computing
            dissimilarity weights.
        threads: Threads for parallel dissimilarity computation.
    """
    from treedata import TreeData

    dissimilarity_fn = _resolve_dissimilarity(dissimilarity)
    is_treedata = isinstance(tdata, TreeData)

    characters: pd.DataFrame | None = None
    if is_treedata and dist_key is None:
        characters = solver_utilities._get_characters(tdata, characters_key)
    elif not is_treedata and characters_key is not None:
        characters = solver_utilities._get_characters(tdata, characters_key)

    dist_df = _get_distances(
        tdata, dist_key, dissimilarity_fn, characters, prior_transformation, threads
    )

    node_gen = solver_utilities.node_name_generator()
    rooted = _build_graph(dist_df, node_gen)

    solver_utilities._set_tree(tdata, rooted, characters_key, tree_key)


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
        | None = dissimilarity_functions.weighted_hamming_distance,
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
        upgma(
            cassiopeia_tree,
            dissimilarity=self.dissimilarity_function,
            characters_key=layer,
            prior_transformation=self.prior_transformation,
            threads=self.threads,
        )
        if collapse_mutationless_edges:
            solver_utilities.collapse_mutationless_edges(cassiopeia_tree)

    def root_tree(self, tree, root_sample, remaining_samples):
        raise NotImplementedError(
            "root_tree is removed in favor of the fast Cython implementation. "
            "Use cas.solver.upgma() directly."
        )

    def find_cherry(self, dissimilarity_matrix):
        raise NotImplementedError(
            "find_cherry is removed in favor of the fast Cython implementation. "
            "Use cas.solver.upgma() directly."
        )

    def update_dissimilarity_map(self, dissimilarity_map, cherry, new_node):
        raise NotImplementedError(
            "update_dissimilarity_map is removed in favor of the fast Cython "
            "implementation. Use cas.solver.upgma() directly."
        )

    def setup_root_finder(self, cassiopeia_tree):
        raise NotImplementedError(
            "setup_root_finder is removed in favor of the fast Cython "
            "implementation. Use cas.solver.upgma() directly."
        )
