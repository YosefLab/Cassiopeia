"""Hybrid solver: functional API and HybridSolver shim.

A hybrid reconstruction applies a top-down split criterion (e.g. greedy) until a
cutoff (LCA distance or number of cells) is reached, then solves each resulting
subproblem with a more precise bottom solver (e.g. ILP).  In Jones et al, the
Cassiopeia-Hybrid algorithm stacks a greedy top on an ILP bottom.

Sub-solvers may be passed as **strings** or **callables**:

- ``top_solver`` is a *split function* with signature
  ``(character_matrix, samples, weights, missing_state_indicator) -> (left, right)``,
  or the name of a registered split (e.g. ``"greedy"``).  Defaults to the vanilla
  greedy split.  Extra keyword arguments may be supplied via ``top_kwargs``.
- ``bottom_solver`` is applied to each subproblem as ``bottom_solver(sub_tdata)``,
  modifying ``sub_tdata`` in place, or the name of a registered solver
  (e.g. ``"ilp"``, ``"nj"``, ``"upgma"``, ``"greedy"``).  Extra keyword arguments
  may be supplied via ``bottom_kwargs``.
"""

from __future__ import annotations

import functools
import multiprocessing
import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from cassiopeia import dissimilarity
from cassiopeia.data import utilities as data_utilities
from cassiopeia.dissimilarity._pairwise import _encode_integer_matrix
from cassiopeia.mixins import HybridSolverError, find_duplicate_groups
from cassiopeia.utils import (
    _get_characters,
    _get_parameter,
    _node_name_generator,
    _set_tree,
    _transform_priors,
)

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def _assess_cutoff(
    samples: list[str],
    character_matrix: pd.DataFrame,
    missing_state_indicator: int,
    lca_cutoff: float | None,
    cell_cutoff: int | None,
) -> bool:
    """Return ``True`` when *samples* meet the bottom-solver cutoff."""
    if cell_cutoff is None:
        root_states = data_utilities.get_lca_characters(
            character_matrix.loc[samples].values.tolist(), missing_state_indicator
        )
        lca_distances = [
            dissimilarity.hamming(np.array(root_states), character_matrix.loc[u].values)
            for u in samples
        ]
        if np.max(lca_distances) <= lca_cutoff:
            return True
    else:
        if len(samples) <= cell_cutoff:
            return True
    return False


def _apply_top_solver(
    character_matrix: pd.DataFrame,
    samples: list[str],
    tree: nx.DiGraph,
    node_name_generator: Generator[str, None, None],
    split_fn: Callable,
    weights: dict[int, dict[int, float]] | None,
    missing_state_indicator: int,
    lca_cutoff: float | None,
    cell_cutoff: int | None,
    pbar: tqdm | None = None,
) -> tuple[str, list[tuple[str, list[str]]], nx.DiGraph]:
    """Recursively split *samples* with *split_fn* until the cutoff is reached.

    Returns the root node of this subtree, the list of subproblems
    ``[(subtree_root, subtree_samples), ...]`` to hand to the bottom solver, and
    the in-progress tree.  If *pbar* is given, it is advanced by the number of
    cells as they are resolved into terminal subproblems/polytomies.
    """
    if len(samples) == 1:
        if pbar is not None:
            pbar.update(1)
        return samples[0], [samples], tree

    clades = [
        clade
        for clade in split_fn(character_matrix, samples, weights, missing_state_indicator)
        if len(clade) != 0
    ]

    root = next(node_name_generator)
    tree.add_node(root)

    if len(clades) == 1:
        for clade in clades[0]:
            tree.add_edge(root, clade)
        if pbar is not None:
            pbar.update(len(clades[0]))
        return root, [], tree

    subproblems = []
    for clade in clades:
        if _assess_cutoff(
            clade, character_matrix, missing_state_indicator, lca_cutoff, cell_cutoff
        ):
            subproblems += [(root, clade)]
            if pbar is not None:
                pbar.update(len(clade))
        else:
            child, new_subproblems, tree = _apply_top_solver(
                character_matrix,
                clade,
                tree,
                node_name_generator,
                split_fn,
                weights,
                missing_state_indicator,
                lca_cutoff,
                cell_cutoff,
                pbar,
            )
            tree.add_edge(root, child)
            subproblems += new_subproblems

    return root, subproblems, tree


def _apply_bottom_solver(
    subproblem_character_matrix: pd.DataFrame,
    root: str,
    samples: list[str],
    missing_state_indicator: int,
    priors: dict | None,
    bottom_solver: Callable,
) -> tuple[nx.DiGraph, str]:
    """Solve a single subproblem with *bottom_solver* and attach it to *root*.

    Builds a fresh :class:`~treedata.TreeData` from the subproblem character
    matrix, calls ``bottom_solver(sub_tdata)`` (which stores the inferred tree in
    ``sub_tdata.obst``), reads it back, and connects its root to *root*.
    """
    if len(samples) == 1:
        subproblem_tree = nx.DiGraph()
        subproblem_tree.add_edge(root, samples[0])
        return subproblem_tree, root

    import treedata as td

    # The subproblem character matrix is already integer-encoded (missing ->
    # missing_state_indicator, unmodified -> 0), so set the canonical state keys
    # the bottom solver reads to avoid re-resolving (and warning about) defaults.
    uns = {"missing_state": missing_state_indicator, "unmodified_state": 0}
    if priors:
        uns["priors"] = priors
    sub_tdata = td.TreeData(
        obs=pd.DataFrame(index=list(subproblem_character_matrix.index)),
        obsm={"characters": subproblem_character_matrix},
        uns=uns,
    )

    bottom_solver(sub_tdata)

    if not sub_tdata.obst:
        raise HybridSolverError(
            "bottom_solver did not store a tree in sub_tdata.obst. The callable "
            "must solve the TreeData in place (e.g. functools.partial(ilp, ...))."
        )
    subproblem_tree = next(iter(sub_tdata.obst.values())).copy()
    subproblem_root = [n for n in subproblem_tree if subproblem_tree.in_degree(n) == 0][0]
    subproblem_tree.add_edge(root, subproblem_root)

    return subproblem_tree, root


def _apply_bottom_solver_star(args: tuple) -> tuple[nx.DiGraph, str]:
    """Unpack *args* and call :func:`_apply_bottom_solver` (for ``Pool.imap``)."""
    return _apply_bottom_solver(*args)


def _resolve_top_solver(
    top_solver: str | Callable | None,
    top_kwargs: dict | None,
) -> Callable:
    """Resolve *top_solver* (name, callable, or ``None``) to a split callable.

    ``None`` and ``"greedy"`` both resolve to the vanilla greedy split. Any
    ``top_kwargs`` are bound via :func:`functools.partial`.
    """
    if top_solver is None:
        top_solver = "greedy"
    if isinstance(top_solver, str):
        from cassiopeia.solver.greedy import _greedy_split

        registry: dict[str, Callable] = {"greedy": _greedy_split}
        if top_solver not in registry:
            raise HybridSolverError(
                f"Unknown top_solver {top_solver!r}. Available: {sorted(registry)}."
            )
        top_solver = registry[top_solver]
    if top_kwargs:
        top_solver = functools.partial(top_solver, **top_kwargs)
    return top_solver


def _resolve_bottom_solver(
    bottom_solver: str | Callable | None,
    bottom_kwargs: dict | None,
) -> Callable:
    """Resolve *bottom_solver* (name or callable) to a solver callable.

    Recognized names are ``"ilp"``, ``"nj"``, ``"upgma"`` and ``"greedy"``. Any
    ``bottom_kwargs`` are bound via :func:`functools.partial` (which stays
    picklable for ``threads > 1``).
    """
    if bottom_solver is None:
        raise HybridSolverError(
            'Please provide a bottom_solver, e.g. "ilp" or functools.partial(ilp, ...).'
        )
    if isinstance(bottom_solver, str):
        from cassiopeia.solver.greedy import greedy
        from cassiopeia.solver.ilp import ilp
        from cassiopeia.solver.neighbor_joining import nj
        from cassiopeia.solver.upgma import upgma

        registry: dict[str, Callable] = {
            "ilp": ilp,
            "nj": nj,
            "upgma": upgma,
            "greedy": greedy,
        }
        if bottom_solver not in registry:
            raise HybridSolverError(
                f"Unknown bottom_solver {bottom_solver!r}. Available: {sorted(registry)}."
            )
        bottom_solver = registry[bottom_solver]
    if bottom_kwargs:
        bottom_solver = functools.partial(bottom_solver, **bottom_kwargs)
    return bottom_solver


def _add_duplicates_to_tree_and_remove_spurious_leaves(
    tree: nx.DiGraph,
    character_matrix: pd.DataFrame,
    node_name_generator: Generator[str, None, None],
) -> nx.DiGraph:
    """Re-add duplicate samples as sisters and prune non-sample leaf lineages."""
    duplicate_mappings = find_duplicate_groups(character_matrix)

    for i in duplicate_mappings:
        new_internal_node = next(node_name_generator)
        nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
        for duplicate in duplicate_mappings[i]:
            tree.add_edge(new_internal_node, duplicate)

    # remove extant lineages that don't correspond to leaves
    to_drop = []
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    for leaf in leaves:
        if leaf not in character_matrix.index:
            to_drop.append(leaf)
            parent = list(tree.predecessors(leaf))[0]
            while tree.out_degree(parent) < 2:
                to_drop.append(parent)
                parent = list(tree.predecessors(parent))[0]

    tree.remove_nodes_from(to_drop)

    return tree


def hybrid(
    tdata: TreeData,
    top_solver: str | Callable = "greedy",
    bottom_solver: str | Callable = "ilp",
    lca_cutoff: float | None = None,
    cell_cutoff: int | None = None,
    progress_bar: bool = True,
    characters_key: str | None = "characters",
    key_added: str = "hybrid",
    prior_transformation: str = "negative_log",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict[int, dict[int, float]] | None = None,
    top_kwargs: dict | None = None,
    bottom_kwargs: dict | None = None,
    threads: int = 1,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct a tree with a hybrid (top-down split + bottom solver) approach.

    A top-down split criterion clusters cells until a cutoff (``lca_cutoff`` or
    ``cell_cutoff``) is reached, then *bottom_solver* reconstructs each
    subproblem. The character matrix is read from ``tdata.obsm`` and the result
    is stored as an ``nx.DiGraph`` in ``tdata.obst[key_added]``.

    Args:
        tdata: TreeData to operate on.
        top_solver: Split function
            ``(character_matrix, samples, weights, missing_state_indicator) ->
            (left, right)``, or the name of a registered split (``"greedy"``).
            Defaults to the vanilla greedy split.
        bottom_solver: Callable applied to each subproblem as
            ``bottom_solver(sub_tdata)``, modifying the TreeData in place
            (e.g. ``functools.partial(ilp, ...)``), or the name of a registered
            solver (``"ilp"``, ``"nj"``, ``"upgma"``, ``"greedy"``).  Must be
            picklable when ``threads > 1`` (use names, module-level functions, or
            ``partial`` — not closures).
        lca_cutoff: LCA-distance cutoff for switching to the bottom solver.
        cell_cutoff: Cell-count cutoff for switching to the bottom solver.
        progress_bar: Whether to display a progress bar over subproblems.
        characters_key: Key in ``tdata.obsm`` for the character matrix
            (default ``'characters'``).
        key_added: Key in ``tdata.obst`` for the resulting tree.
        prior_transformation: Transformation applied to priors to form weights.
        missing_state: Missing-state value (read from ``tdata.uns`` if ``None``).
        unmodified_state: Unmodified/uncut state value (read from ``tdata.uns``
            if ``None``).
        priors: Priors for character states, as a dict mapping character index
            to dicts mapping state to prior probability (read from ``tdata.uns``
            if ``None``).
        top_kwargs: Extra keyword arguments bound to *top_solver*.
        bottom_kwargs: Extra keyword arguments bound to *bottom_solver*
            (e.g. ``{"weighted": True}`` for the ILP bottom solver).
        threads: Number of subproblems to solve concurrently.
        copy: If ``True``, return a copy of *tdata*; otherwise modify in-place
            and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        HybridSolverError: If no cutoff or no bottom_solver is provided, or if no
            character matrix is found.
    """
    if lca_cutoff is None and cell_cutoff is None:
        raise HybridSolverError(
            "Please specify a cutoff, either through lca_cutoff or cell_cutoff."
        )
    top_solver = _resolve_top_solver(top_solver, top_kwargs)
    bottom_solver = _resolve_bottom_solver(bottom_solver, bottom_kwargs)

    tdata = tdata.copy() if copy else tdata
    character_matrix = _get_characters(tdata, characters_key).copy()
    missing_state_indicator = _get_parameter(tdata, "missing_state", value=missing_state)
    unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    priors = _get_parameter(tdata, "priors", value=priors)
    # Encode string/categorical states to integers so the greedy split logic and
    # the bottom solver operate on the integer convention.
    character_matrix, missing_state_indicator = _encode_integer_matrix(
        character_matrix, missing_state_indicator, unmodified_state
    )

    weights = None
    if priors:
        weights = _transform_priors(priors, prior_transformation)

    unique_character_matrix = character_matrix.drop_duplicates()

    node_name_generator = _node_name_generator()

    tree = nx.DiGraph()
    with tqdm(
        total=len(unique_character_matrix),
        desc="Top-down split",
        disable=not progress_bar,
    ) as top_pbar:
        _, subproblems, tree = _apply_top_solver(
            unique_character_matrix,
            list(unique_character_matrix.index),
            tree,
            node_name_generator,
            top_solver,
            weights,
            missing_state_indicator,
            lca_cutoff,
            cell_cutoff,
            top_pbar,
        )

    args = [
        (
            unique_character_matrix.loc[samples],
            root,
            samples,
            missing_state_indicator,
            priors,
            bottom_solver,
        )
        for (root, samples) in subproblems
    ]

    if threads > 1:
        with multiprocessing.Pool(processes=threads) as pool:
            # imap (vs. starmap) yields results as they complete so the progress
            # bar advances incrementally rather than jumping to 100% at the end.
            results = list(
                tqdm(
                    pool.imap(_apply_bottom_solver_star, args),
                    total=len(args),
                    desc="Bottom solver",
                    disable=not progress_bar,
                )
            )
    else:
        results = [
            _apply_bottom_solver(*a)
            for a in tqdm(args, total=len(args), desc="Bottom solver", disable=not progress_bar)
        ]

    for subproblem_tree, subproblem_root in results:
        # Rename overlapping (non-root) nodes so subproblem trees don't merge
        # across unrelated parts of the tree.
        existing_nodes = list(tree)
        mapping = {}
        for n in subproblem_tree:
            if n in existing_nodes and n != subproblem_root:
                mapping[n] = next(node_name_generator)
                existing_nodes.append(mapping[n])
            else:
                existing_nodes.append(n)
        subproblem_tree = nx.relabel_nodes(subproblem_tree, mapping)
        tree = nx.compose(tree, subproblem_tree)

    samples_tree = _add_duplicates_to_tree_and_remove_spurious_leaves(
        tree, character_matrix, node_name_generator
    )

    _set_tree(tdata, samples_tree, key_added)

    return tdata if copy else None


# ── Backward-compat shim ─────────────────────────────────────────────────────


class HybridSolver:
    """The Hybrid Cassiopeia solver.

    Thin shim around :func:`cassiopeia.solver.hybrid` for backward compatibility.
    Accepts solver *instances* for ``top_solver`` / ``bottom_solver`` and adapts
    them to the callables expected by :func:`hybrid`.  For new code, prefer
    calling :func:`cassiopeia.solver.hybrid` directly with callables.

    Args:
        top_solver: A greedy-style solver instance exposing ``perform_split``.
        bottom_solver: A solver instance exposing ``solve``.
        lca_cutoff: LCA-distance cutoff.
        cell_cutoff: Cell-count cutoff.
        threads: Number of subproblems to solve concurrently.
        prior_transformation: Transformation applied to priors to form weights.
        progress_bar: Whether to display a progress bar.
    """

    def __init__(
        self,
        top_solver,
        bottom_solver,
        lca_cutoff: float | None = None,
        cell_cutoff: int | None = None,
        threads: int = 1,
        prior_transformation: str = "negative_log",
        progress_bar: bool = True,
    ):
        warnings.warn(
            "HybridSolver is deprecated and will be removed in a future release. "
            "Use cassiopeia.solver.hybrid() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if lca_cutoff is None and cell_cutoff is None:
            raise HybridSolverError(
                "Please specify a cutoff, either through lca_cutoff or cell_cutoff"
            )

        self.top_solver = top_solver
        self.bottom_solver = bottom_solver
        self.top_solver.prior_transformation = prior_transformation
        self.bottom_solver.prior_transformation = prior_transformation
        self.lca_cutoff = lca_cutoff
        self.cell_cutoff = cell_cutoff
        self.threads = threads
        self.prior_transformation = prior_transformation
        self.progress_bar = progress_bar

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Run the hybrid solver in-place by delegating to :func:`hybrid`."""
        hybrid(
            cassiopeia_tree,
            top_solver=self.top_solver.perform_split,
            bottom_solver=self.bottom_solver.solve,
            lca_cutoff=self.lca_cutoff,
            cell_cutoff=self.cell_cutoff,
            threads=self.threads,
            prior_transformation=self.prior_transformation,
            progress_bar=self.progress_bar,
            characters_key=layer,
        )
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)
