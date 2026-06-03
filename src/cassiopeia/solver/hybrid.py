"""Hybrid solver: functional API and HybridSolver shim.

A hybrid reconstruction applies a top-down split criterion (e.g. greedy) until a
cutoff (LCA distance or number of cells) is reached, then solves each resulting
subproblem with a more precise bottom solver (e.g. ILP).  In Jones et al, the
Cassiopeia-Hybrid algorithm stacks a greedy top on an ILP bottom.

Sub-solvers are passed as **callables**:

- ``top_solver`` is a *split function* with signature
  ``(character_matrix, samples, weights, missing_state_indicator) -> (left, right)``.
  Defaults to the vanilla greedy split.
- ``bottom_solver`` is applied to each subproblem as ``bottom_solver(sub_tdata)``,
  modifying ``sub_tdata`` in place (e.g. ``functools.partial(ilp, ...)``).
"""

from __future__ import annotations

import multiprocessing
import warnings
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cassiopeia import dissimilarity
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import HybridSolverError, find_duplicate_groups
from cassiopeia.solver import solver_utilities

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
            dissimilarity.hamming_distance(np.array(root_states), character_matrix.loc[u].values)
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
) -> tuple[str, list[tuple[str, list[str]]], nx.DiGraph]:
    """Recursively split *samples* with *split_fn* until the cutoff is reached.

    Returns the root node of this subtree, the list of subproblems
    ``[(subtree_root, subtree_samples), ...]`` to hand to the bottom solver, and
    the in-progress tree.
    """
    if len(samples) == 1:
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
        return root, [], tree

    subproblems = []
    for clade in clades:
        if _assess_cutoff(
            clade, character_matrix, missing_state_indicator, lca_cutoff, cell_cutoff
        ):
            subproblems += [(root, clade)]
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

    uns = {"missing_state_indicator": missing_state_indicator}
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
    tdata: CassiopeiaTree | TreeData,
    top_solver: Callable | None = None,
    bottom_solver: Callable | None = None,
    lca_cutoff: float | None = None,
    cell_cutoff: int | None = None,
    threads: int = 1,
    prior_transformation: str = "negative_log",
    progress_bar: bool = True,
    characters_key: str | None = None,
    tree_key: str = "hybrid",
) -> None:
    """Hybrid (top-down split + bottom solver) reconstruction.  Modifies *tdata* in-place.

    A top-down split criterion clusters cells until a cutoff (``lca_cutoff`` or
    ``cell_cutoff``) is reached, then *bottom_solver* reconstructs each
    subproblem.  For :class:`~treedata.TreeData` the result ``nx.DiGraph`` is
    stored in ``tdata.obst[tree_key]``; for :class:`~cassiopeia.data.CassiopeiaTree`
    the topology is populated via ``populate_tree()``.

    Args:
        tdata: CassiopeiaTree or TreeData to solve.
        top_solver: Split function
            ``(character_matrix, samples, weights, missing_state_indicator) ->
            (left, right)``.  Defaults to the vanilla greedy split.
        bottom_solver: Callable applied to each subproblem as
            ``bottom_solver(sub_tdata)``, modifying the TreeData in place
            (e.g. ``functools.partial(ilp, ...)``).  Must be picklable when
            ``threads > 1`` (use module-level functions / ``partial``, not closures).
        lca_cutoff: LCA-distance cutoff for switching to the bottom solver.
        cell_cutoff: Cell-count cutoff for switching to the bottom solver.
        threads: Number of subproblems to solve concurrently.
        prior_transformation: Transformation applied to priors to form weights.
        progress_bar: Whether to display a progress bar over subproblems.
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm`` key
            (TreeData, default ``'characters'``).
        tree_key: Key in ``tdata.obst`` for the result (TreeData only).

    Raises:
        HybridSolverError: If no cutoff or no bottom_solver is provided, or if no
            character matrix is found.
    """
    if lca_cutoff is None and cell_cutoff is None:
        raise HybridSolverError(
            "Please specify a cutoff, either through lca_cutoff or cell_cutoff."
        )
    if bottom_solver is None:
        raise HybridSolverError(
            "Please provide a bottom_solver callable, e.g. functools.partial(ilp, ...)."
        )
    if top_solver is None:
        from cassiopeia.solver.greedy import _greedy_split

        top_solver = _greedy_split

    character_matrix = solver_utilities._get_characters(tdata, characters_key)
    if character_matrix is None:
        raise HybridSolverError(
            "No character matrix found; store characters in "
            f"obsm[{characters_key or 'characters'!r}] (TreeData) or set one on "
            "the CassiopeiaTree."
        )
    character_matrix = character_matrix.copy()
    missing_state_indicator, priors = solver_utilities._get_missing_and_priors(tdata)

    weights = None
    if priors:
        weights = solver_utilities.transform_priors(priors, prior_transformation)

    unique_character_matrix = character_matrix.drop_duplicates()

    node_name_generator = solver_utilities.node_name_generator()

    tree = nx.DiGraph()
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
            results = list(
                tqdm(
                    pool.starmap(_apply_bottom_solver, args),
                    total=len(args),
                    disable=not progress_bar,
                )
            )
    else:
        results = [
            _apply_bottom_solver(*a) for a in tqdm(args, total=len(args), disable=not progress_bar)
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

    solver_utilities._set_tree(tdata, samples_tree, characters_key, tree_key)


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
            solver_utilities.collapse_mutationless_edges(cassiopeia_tree)
