"""Greedy solver: functional API and VanillaGreedySolver shim.

The "vanilla" Cassiopeia-Greedy algorithm, originally proposed in Jones et al,
Genome Biology (2020), recursively splits samples into mutually exclusive groups
based on the presence or absence of the most frequently occurring mutation.

This module provides the functional :func:`greedy` entry point, the reusable
top-down split machinery (:func:`_greedy_solve`), the vanilla split criterion
(:func:`_greedy_split`), and a backward-compatible :class:`VanillaGreedySolver`.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.mixins import (
    GreedySolverError,
    find_duplicate_groups,
    is_ambiguous_state,
    unravel_ambiguous_states,
)
from cassiopeia.solver import missing_data_methods, solver_utilities

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def _compute_mutation_frequencies(
    samples: list[str],
    unique_character_matrix: pd.DataFrame,
    missing_state_indicator: int = -1,
) -> dict[int, dict[int, int]]:
    """Compute character/state mutation frequencies over a set of samples.

    Generates a dictionary mapping each character to a dictionary of
    state/frequency pairs, restricted to *samples*.  Supports ambiguous states.

    Args:
        samples: The set of relevant samples in calculating frequencies.
        unique_character_matrix: The character matrix from which to calculate
            frequencies.
        missing_state_indicator: The character representing missing values.

    Returns:
        A dictionary containing frequency information for each character/state
        pair.
    """
    subset_cm = unique_character_matrix.loc[samples, :].to_numpy()
    freq_dict = {}
    for char in range(subset_cm.shape[1]):
        char_dict = {}
        all_states = unravel_ambiguous_states(subset_cm[:, char])
        state_counts = np.unique(all_states, return_counts=True)

        for i in range(len(state_counts[0])):
            state = state_counts[0][i]
            count = state_counts[1][i]
            char_dict[state] = count
        if missing_state_indicator not in char_dict:
            char_dict[missing_state_indicator] = 0
        freq_dict[char] = char_dict

    return freq_dict


def _greedy_split(
    character_matrix: pd.DataFrame,
    samples: list[str],
    weights: dict[int, dict[int, float]] | None = None,
    missing_state_indicator: int = -1,
    missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
) -> tuple[list[str], list[str]]:
    """Partition *samples* based on the most frequent (character, state) pair.

    Splits the sample set into two partitions on the presence/absence of the
    most frequent mutation.  Samples with missing data at the chosen character
    are assigned by *missing_data_classifier*.

    Args:
        character_matrix: Character matrix (deduplicated).
        samples: A list of samples to partition.
        weights: Weighting of each (character, state) pair, typically a
            transformation of the priors.
        missing_state_indicator: Character representing missing data.
        missing_data_classifier: Function classifying samples with missing data
            at the chosen character into the left/right partition.

    Returns:
        A tuple of lists representing the left and right partition groups.
    """
    sample_indices = solver_utilities.convert_sample_names_to_indices(
        character_matrix.index, samples
    )
    mutation_frequencies = _compute_mutation_frequencies(
        samples, character_matrix, missing_state_indicator
    )

    best_frequency = 0
    chosen_character = 0
    chosen_state = 0
    for character in mutation_frequencies:
        for state in mutation_frequencies[character]:
            if state != missing_state_indicator and state != 0:
                # Avoid splitting on mutations shared by all samples
                if (
                    mutation_frequencies[character][state]
                    < len(samples) - mutation_frequencies[character][missing_state_indicator]
                ):
                    if weights:
                        if (
                            mutation_frequencies[character][state] * weights[character][state]
                            > best_frequency
                        ):
                            chosen_character, chosen_state = (character, state)
                            best_frequency = (
                                mutation_frequencies[character][state] * weights[character][state]
                            )
                    else:
                        if mutation_frequencies[character][state] > best_frequency:
                            chosen_character, chosen_state = (character, state)
                            best_frequency = mutation_frequencies[character][state]

    if chosen_state == 0:
        return samples, []

    left_set = []
    right_set = []
    missing = []

    unique_character_array = character_matrix.to_numpy()
    sample_names = list(character_matrix.index)

    ambiguous_contains = lambda query, _s: _s in query if is_ambiguous_state(query) else _s == query

    for i in sample_indices:
        observed_state = unique_character_array[i, chosen_character]
        if ambiguous_contains(observed_state, chosen_state):
            left_set.append(sample_names[i])
        elif unique_character_array[i, chosen_character] == missing_state_indicator:
            missing.append(sample_names[i])
        else:
            right_set.append(sample_names[i])

    left_set, right_set = missing_data_classifier(
        character_matrix,
        missing_state_indicator,
        left_set,
        right_set,
        missing,
        weights=weights,
    )

    return left_set, right_set


def _add_duplicates_to_tree(
    tree: nx.DiGraph,
    character_matrix: pd.DataFrame,
    node_name_generator,
) -> nx.DiGraph:
    """Place duplicate samples in *tree* as sisters of their representative.

    Samples removed during deduplication are added back as sisters to the cells
    that share their mutations.

    Args:
        tree: The tree to add duplicates to.
        character_matrix: The full (non-deduplicated) character matrix.
        node_name_generator: Generator producing unique internal node names.

    Returns:
        The tree with duplicates added.
    """
    duplicate_mappings = find_duplicate_groups(character_matrix)

    for i in duplicate_mappings:
        new_internal_node = next(node_name_generator)
        nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
        for duplicate in duplicate_mappings[i]:
            tree.add_edge(new_internal_node, duplicate)

    return tree


def _greedy_solve(
    character_matrix: pd.DataFrame,
    split_fn: Callable,
    *,
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None,
    allow_ambiguous: bool,
) -> nx.DiGraph:
    """Build a tree top-down by recursively splitting the sample set.

    Each call to *split_fn* partitions a set of samples; an ancestral node is
    created and each side of the partition is attached as a daughter clade.
    This recurses until clades are singletons.  When a split cannot be made the
    samples form a polytomy.  Duplicate samples are re-added at the end.

    Args:
        character_matrix: The character matrix (samples × characters).
        split_fn: A function ``(character_matrix, samples, weights,
            missing_state_indicator) -> (left, right)`` partitioning *samples*.
        missing_state_indicator: Character representing missing data.
        weights: Per-(character, state) weights from priors, or ``None``.
        allow_ambiguous: Whether ambiguous states are permitted.

    Returns:
        The reconstructed tree as an :class:`~networkx.DiGraph`.

    Raises:
        GreedySolverError: If the matrix contains ambiguous states and
            *allow_ambiguous* is ``False``.
    """
    node_name_generator = solver_utilities.node_name_generator()

    if (
        any(is_ambiguous_state(state) for state in character_matrix.values.flatten())
        and not allow_ambiguous
    ):
        raise GreedySolverError("Ambiguous states are not currently supported with this solver.")

    keep_rows = (
        character_matrix.apply(
            lambda x: [set(s) if is_ambiguous_state(s) else {s} for s in x.values],
            axis=0,
        )
        .apply(tuple, axis=1)
        .drop_duplicates()
        .index.values
    )
    unique_character_matrix = character_matrix.loc[keep_rows].copy()

    tree = nx.DiGraph()
    tree.add_nodes_from(list(unique_character_matrix.index))

    def _solve(samples: list[str]):
        if len(samples) == 1:
            return samples[0]
        clades = [
            clade
            for clade in split_fn(
                unique_character_matrix, samples, weights, missing_state_indicator
            )
            if len(clade) != 0
        ]
        root = next(node_name_generator)
        tree.add_node(root)

        # If unable to split, generate a polytomy and return.
        if len(clades) == 1:
            for clade in clades[0]:
                tree.add_edge(root, clade)
            return root
        for clade in clades:
            child = _solve(clade)
            tree.add_edge(root, child)
        return root

    _solve(list(unique_character_matrix.index))

    return _add_duplicates_to_tree(tree, character_matrix, node_name_generator)


def greedy(
    tdata: CassiopeiaTree | TreeData,
    characters_key: str | None = None,
    tree_key: str = "greedy",
    missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
    prior_transformation: str = "negative_log",
) -> None:
    """Vanilla Cassiopeia-Greedy reconstruction.  Modifies *tdata* in-place.

    Builds a tree top-down by recursively splitting samples on the most frequent
    mutation.  For :class:`~cassiopeia.data.CassiopeiaTree` the topology is
    populated via ``populate_tree()``; for :class:`~treedata.TreeData` the result
    ``nx.DiGraph`` is stored in ``tdata.obst[tree_key]``.

    Args:
        tdata: CassiopeiaTree or TreeData to solve.
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm`` key
            (TreeData, default ``'characters'``).
        tree_key: Key in ``tdata.obst`` for the result (TreeData only).
        missing_data_classifier: Function assigning samples with missing data at
            the split character into the left/right partition.
        prior_transformation: Transformation applied to priors to form weights.
    """
    character_matrix = solver_utilities._get_characters(tdata, characters_key)
    if character_matrix is None:
        raise ValueError(
            "No character matrix found; store characters in "
            f"obsm[{characters_key or 'characters'!r}] (TreeData) or set one on "
            "the CassiopeiaTree."
        )

    missing_state_indicator, priors = solver_utilities._get_missing_and_priors(tdata)

    weights = None
    if priors:
        weights = solver_utilities.transform_priors(priors, prior_transformation)

    split_fn = functools.partial(_greedy_split, missing_data_classifier=missing_data_classifier)

    tree = _greedy_solve(
        character_matrix,
        split_fn,
        missing_state_indicator=missing_state_indicator,
        weights=weights,
        allow_ambiguous=True,
    )

    solver_utilities._set_tree(tdata, tree, characters_key, tree_key)


# ── Backward-compat shim ─────────────────────────────────────────────────────


class VanillaGreedySolver:
    """A class for the basic Cassiopeia-Greedy solver.

    Thin shim around :func:`cassiopeia.solver.greedy` for backward
    compatibility.  For new code, prefer calling :func:`cassiopeia.solver.greedy`
    directly.

    Args:
        missing_data_classifier: A function implementing a missing-data
            imputation method.  Defaults to the "average" method.
        prior_transformation: Transformation applied to priors to form weights.
    """

    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
    ):
        warnings.warn(
            "VanillaGreedySolver is deprecated and will be removed in a future "
            "release. Use cassiopeia.solver.greedy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.missing_data_classifier = missing_data_classifier
        self.prior_transformation = prior_transformation
        self.allow_ambiguous = True

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Reconstruct the tree topology in-place using vanilla greedy.

        Args:
            cassiopeia_tree: CassiopeiaTree to solve in-place.
            layer: Character matrix layer to use.
            collapse_mutationless_edges: Collapse edges with no inferred
                mutations after solving.
            logfile: Ignored (kept for API compatibility).
        """
        greedy(
            cassiopeia_tree,
            characters_key=layer,
            missing_data_classifier=self.missing_data_classifier,
            prior_transformation=self.prior_transformation,
        )
        if collapse_mutationless_edges:
            solver_utilities.collapse_mutationless_edges(cassiopeia_tree)

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: list[str],
        weights: dict[int, dict[int, float]] | None = None,
        missing_state_indicator: int = -1,
    ) -> tuple[list[str], list[str]]:
        """Partition *samples* on the most frequent (character, state) pair.

        Retained for backward compatibility; delegates to :func:`_greedy_split`.
        """
        return _greedy_split(
            character_matrix,
            samples,
            weights,
            missing_state_indicator,
            missing_data_classifier=self.missing_data_classifier,
        )

    def compute_mutation_frequencies(
        self,
        samples: list[str],
        unique_character_matrix: pd.DataFrame,
        missing_state_indicator: int = -1,
    ) -> dict[int, dict[int, int]]:
        """Compute character/state mutation frequencies over *samples*.

        Retained for backward compatibility; delegates to
        :func:`_compute_mutation_frequencies`.
        """
        return _compute_mutation_frequencies(
            samples, unique_character_matrix, missing_state_indicator
        )
