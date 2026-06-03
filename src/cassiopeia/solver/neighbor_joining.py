"""Neighbor-Joining solver: functional API and NeighborJoiningSolver shim."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia import dissimilarity as dissimilarity_functions
from cassiopeia.dissimilarity import _pairwise, _resolve_dissimilarity
from cassiopeia.solver import rooting, solver_utilities

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def _build_graph(
    dissimilarity_map: pd.DataFrame,
    node_name_generator: Generator[str, None, None],
) -> nx.Graph:
    """Run Cython DNJ and return a complete unrooted graph.

    Args:
        dissimilarity_map: Symmetric n×n distance DataFrame.
        node_name_generator: Generator supplying unique names for internal nodes.

    Returns:
        A complete undirected :class:`~networkx.Graph` ready for rooting.
    """
    from cassiopeia.solver import nj_solver_utilities

    sample_names = list(dissimilarity_map.index)
    n = len(sample_names)
    D = np.ascontiguousarray(dissimilarity_map.to_numpy(dtype=np.float64))
    parents, children, _, node_a, node_b = nj_solver_utilities.dnj(D)

    id_to_name: dict[int, str] = {i: sample_names[i] for i in range(n)}
    for k in range(n - 2):
        id_to_name[n + k] = next(node_name_generator)

    tree = nx.Graph()
    for name in sample_names:
        tree.add_node(name)
    for p, c in zip(parents, children, strict=False):
        tree.add_edge(id_to_name[int(p)], id_to_name[int(c)])

    # Add the final edge between the last two unmerged nodes
    tree.add_edge(id_to_name[node_a], id_to_name[node_b])
    return tree


def nj(
    tdata: CassiopeiaTree | TreeData,
    dist_key: str | None = None,
    dissimilarity: str | Callable | None = "weighted_hamming_distance",
    root: str | None = None,
    outgroup: str | None = None,
    characters_key: str | None = None,
    tree_key: str = "nj",
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> None:
    """Dynamic Neighbor-Joining (Cython O(n²) average case). Modifies tdata in-place.

    For :class:`~cassiopeia.data.CassiopeiaTree`: populates tree topology via
    ``populate_tree()``.  For :class:`~treedata.TreeData`: stores the result
    ``nx.DiGraph`` in ``tdata.obst[tree_key]``.

    When ``root='outgroup'`` with ``outgroup=None``, a synthetic all-zero leaf
    named ``'root'`` is added to the character matrix **before** computing
    distances; the NJ tree is then rooted at that leaf.  When ``outgroup`` is a
    sample name, a new root node is inserted on the branch connecting that
    sample to the rest of the tree.

    Args:
        tdata: CassiopeiaTree or TreeData to solve.
        dist_key: Key in ``tdata.obsp`` for precomputed distances (TreeData
            only, ignored when ``root='outgroup'`` and ``outgroup=None``).
        dissimilarity: Function used to compute pairwise dissimilarities.
            Accepts a callable or a string name of a function in
            :mod:`cassiopeia.solver.dissimilarity_functions`.
        root: Rooting procedure.  ``'outgroup'`` uses outgroup rooting (see
            *outgroup*).  ``None`` (default) uses ``root_sample_name`` or the
            first obs name for TreeData.
        outgroup: For ``root='outgroup'``: sample name to use as outgroup, or
            ``None`` to add a synthetic all-zero outgroup named ``'root'``.
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm``
            key (TreeData, default ``'characters'``).
        tree_key: Key in ``tdata.obst`` for the result (TreeData only).
        prior_transformation: Transformation applied to priors.
        threads: Threads for parallel dissimilarity computation.
    """
    dissimilarity_fn = _resolve_dissimilarity(dissimilarity)
    synthetic_root = root == "outgroup" and outgroup is None

    if synthetic_root:
        # Augment the character matrix with a synthetic all-zero 'root' leaf and
        # compute distances fresh from the augmented matrix.
        chars = solver_utilities._get_characters(tdata, characters_key)
        if chars is None:
            raise ValueError(
                "A character matrix is required for root='outgroup' with "
                "outgroup=None.  Provide characters_key or store characters "
                "in obsm['characters']."
            )
        root_row = pd.DataFrame(
            [np.zeros(chars.shape[1], dtype=int)],
            index=["root"],
            columns=chars.columns,
        )
        augmented = pd.concat([chars, root_row])
        missing, priors = solver_utilities._get_missing_and_priors(tdata)
        dist_df = _pairwise(
            augmented, dissimilarity_fn, missing, priors, prior_transformation, threads
        )
    else:
        dist_df = solver_utilities.get_distance_map(
            tdata,
            dissimilarity_fn,
            characters_key=characters_key,
            dist_key=dist_key,
            prior_transformation=prior_transformation,
            threads=threads,
        )

    node_gen = solver_utilities.node_name_generator()
    graph = _build_graph(dist_df, node_gen)

    rooted = rooting.apply(graph, root, tdata, outgroup=outgroup)

    solver_utilities._set_tree(tdata, rooted, characters_key, tree_key)


# ── Backward-compat shim ─────────────────────────────────────────────────────


class NeighborJoiningSolver:
    """Neighbor-Joining solver for Cassiopeia.

    Thin shim around :func:`cassiopeia.solver.nj` for backward compatibility.
    Uses the Cython DNJ implementation (Clausen 2021, O(n²) average case).

    For new code, prefer calling :func:`cassiopeia.solver.nj` directly.

    Args:
        dissimilarity_function: Function to compute the dissimilarity map.
        add_root: Whether to root the tree using a synthetic all-zero outgroup.
        prior_transformation: Transformation applied to priors.
        fast: Must be ``True``.  The generic (slow) NJ path has been removed.
        implementation: Deprecated.  Use ``fast=True`` instead.
        threads: Threads for dissimilarity map computation.
    """

    def __init__(
        self,
        dissimilarity_function: Callable[
            [np.array, np.array, int, dict[int, dict[int, float]]], float
        ]
        | None = dissimilarity_functions.weighted_hamming_distance,
        add_root: bool = False,
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
                "The generic (slow) NJ path has been removed in favor of the "
                "fast Cython implementation. "
                "Use NeighborJoiningSolver(fast=True) or cas.solver.nj()."
            )

        self._implementation = "dnj"
        self.dissimilarity_function = dissimilarity_function
        self.add_root = add_root
        self.prior_transformation = prior_transformation
        self.threads = threads

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solve the tree topology using Neighbor-Joining.

        Args:
            cassiopeia_tree: CassiopeiaTree to solve in-place.
            layer: Character matrix layer to use.
            collapse_mutationless_edges: Collapse edges with no inferred
                mutations after solving.
            logfile: Ignored (kept for API compatibility).
        """
        # Use "outgroup" when root_sample_name is absent from both the
        # character matrix and the dissimilarity map.  When it is a real
        # original sample (present in either store), use it directly.
        root = None
        if self.add_root:
            rsn = cassiopeia_tree.root_sample_name
            known_samples = set(cassiopeia_tree.character_matrix.index)
            dist_map = cassiopeia_tree.get_dissimilarity_map()
            if dist_map is not None:
                known_samples |= set(dist_map.index)
            if rsn is None or rsn not in known_samples:
                root = "outgroup"
        nj(
            cassiopeia_tree,
            dissimilarity=self.dissimilarity_function,
            root=root,
            characters_key=layer,
            prior_transformation=self.prior_transformation,
            threads=self.threads,
        )
        if collapse_mutationless_edges:
            solver_utilities.collapse_mutationless_edges(cassiopeia_tree)

    def root_tree(self, tree, root_sample, remaining_samples):
        raise NotImplementedError(
            "root_tree is removed in favor of the fast Cython implementation. "
            "Use cas.solver.nj() directly."
        )

    def find_cherry(self, dissimilarity_matrix):
        raise NotImplementedError(
            "find_cherry is removed in favor of the fast Cython implementation. "
            "Use cas.solver.nj() directly."
        )

    def update_dissimilarity_map(self, dissimilarity_map, cherry, new_node):
        raise NotImplementedError(
            "update_dissimilarity_map is removed in favor of the fast Cython "
            "implementation. Use cas.solver.nj() directly."
        )

    def setup_root_finder(self, cassiopeia_tree):
        raise NotImplementedError(
            "setup_root_finder is removed in favor of the fast Cython "
            "implementation. Use cas.solver.nj() directly."
        )
