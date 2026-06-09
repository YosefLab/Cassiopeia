"""Neighbor-Joining solver: functional API and NeighborJoiningSolver shim."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia import dissimilarity as dissimilarity_functions
from cassiopeia.dissimilarity import _pairwise, _resolve_dissimilarity, _uses_weights
from cassiopeia.solver import rooting
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
    tdata: TreeData,
    dissim_key: str | None = None,
    dissim_fn: str | Callable | None = "nonmissing_hamming",
    root: str | None = "centroid",
    outgroup: str | None = None,
    save_dissim: bool = False,
    characters_key: str | None = None,
    key_added: str = "nj",
    prior_transformation: str = "negative_log",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict[int, dict[int, float]] | bool = False,
    threads: int = 1,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct a tree with Dynamic Neighbor-Joining (Cython, O(n²) average case).

    Builds the tree from a precomputed dissimilarity map in ``tdata.obsp`` or,
    when none is available, from the character matrix in ``tdata.obsm``. The
    result is stored as an ``nx.DiGraph`` in ``tdata.obst[key_added]``.

    Rooting is selected by *root*: ``None`` uses ``root_sample_name`` or the
    first obs name; otherwise *root* names a procedure registered in
    :mod:`cassiopeia.solver.rooting` (``'outgroup'``, ``'midpoint'``,
    ``'centroid'``, ``'shared_mutation'``). When ``root='outgroup'`` with
    ``outgroup=None``, a synthetic all-unmodified leaf named ``'root'`` is added
    to the distance matrix **before** building the tree and used as the root;
    the synthetic leaf is removed from any saved dissimilarity matrix.

    Args:
        tdata: TreeData to operate on.
        dissim_key: Key in ``tdata.obsp`` for a precomputed dissimilarity map
            (ignored when ``root='outgroup'`` and ``outgroup=None``).
        dissim_fn: Dissimilarity function used when distances are not
            precomputed. Accepts a callable or a string name of a built-in
            metric in :mod:`cassiopeia.dissimilarity`.
        root: Rooting procedure name.
        outgroup: For ``root='outgroup'``: sample name to use as outgroup, or
            ``None`` to add a synthetic all-unmodified outgroup named ``'root'``.
        save_dissim: Whether to store the computed dissimilarity map in
            ``tdata.obsp[dissim_key or 'distances']``. The synthetic outgroup is
            excluded.
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
    synthetic_root = root == "outgroup" and outgroup is None

    # Use a precomputed map only when one is actually stored under *dissim_key*;
    # the synthetic-outgroup root requires distances recomputed from the
    # augmented character matrix, so *dissim_key* is then a save target only.
    use_precomputed = dissim_key is not None and not synthetic_root and dissim_key in tdata.obsp

    if use_precomputed:
        dist_df = pd.DataFrame(
            tdata.obsp[dissim_key], index=tdata.obs_names, columns=tdata.obs_names
        )
    else:
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

        if synthetic_root:
            # Augment the character matrix with a synthetic all-unmodified 'root'
            # leaf and compute distances fresh from the augmented matrix.
            # ``unmodified_state`` may be a tuple of acceptable representations
            # (e.g. the default ``(0, "0", "*")``); fill with a single scalar.
            fill = (
                unmodified_state[0]
                if isinstance(unmodified_state, (tuple, list))
                else unmodified_state
            )
            root_row = pd.DataFrame(
                [[fill] * characters.shape[1]],
                index=["root"],
                columns=characters.columns,
            )
            characters = pd.concat([characters, root_row])

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
            # Exclude the synthetic outgroup ('root') from the saved matrix.
            real = dist_df.drop(index="root", columns="root") if synthetic_root else dist_df
            _save_dissimilarity(tdata, real, dissim_key)

    node_gen = _node_name_generator()
    graph = _build_graph(dist_df, node_gen)

    rooted = _root_graph(graph, root, outgroup, tdata, characters_key)

    _set_tree(tdata, rooted, key_added)

    return tdata if copy else None


def _root_graph(
    graph: nx.Graph,
    root: str | None,
    outgroup: str | None,
    data: CassiopeiaTree | TreeData,
    characters_key: str | None,
) -> nx.DiGraph:
    """Root the complete NJ *graph* using the procedure named by *root*.

    ``root=None`` roots at the default node (``root_sample_name`` / first obs)
    via DFS; otherwise *root* selects a registered procedure in
    :mod:`cassiopeia.solver.rooting`.
    """
    if root is None:
        root_node = rooting._default_root(data)
        rooted = nx.DiGraph()
        for e in nx.dfs_edges(graph, source=root_node):
            rooted.add_edge(e[0], e[1])
        return rooted

    if root not in rooting._PROCEDURES:
        raise ValueError(
            f"Unknown rooting procedure {root!r}. Available: {sorted(rooting._PROCEDURES)}"
        )

    proc_kwargs = {}
    if root == "outgroup":
        proc_kwargs["outgroup"] = outgroup
    elif root == "shared_mutation":
        proc_kwargs["characters"] = _get_characters(data, characters_key)
    return rooting._PROCEDURES[root](graph, **proc_kwargs)


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
        | None = dissimilarity_functions.nonmissing_hamming,
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
        # Preserve legacy behavior: use priors if the tree carries them, else not.
        nj(
            cassiopeia_tree,
            dissim_fn=self.dissimilarity_function,
            root=root,
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
            "Use cas.solver.nj() directly."
        )

    def find_cherry(self, dissimilarity_matrix):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "find_cherry is removed in favor of the fast Cython implementation. "
            "Use cas.solver.nj() directly."
        )

    def update_dissimilarity_map(self, dissimilarity_map, cherry, new_node):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "update_dissimilarity_map is removed in favor of the fast Cython "
            "implementation. Use cas.solver.nj() directly."
        )

    def setup_root_finder(self, cassiopeia_tree):
        """Removed. Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "setup_root_finder is removed in favor of the fast Cython "
            "implementation. Use cas.solver.nj() directly."
        )
