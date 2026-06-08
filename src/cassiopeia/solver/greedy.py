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

from cassiopeia.dissimilarity._pairwise import _encode_integer_matrix, _encode_priors
from cassiopeia.mixins import (
    GreedySolverError,
    find_duplicate_groups,
    is_ambiguous_state,
    unravel_ambiguous_states,
)
from cassiopeia.utils import (
    _get_characters,
    _get_parameter,
    _node_name_generator,
    _resolve_priors,
    _set_tree,
    _transform_priors,
)

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


def convert_sample_names_to_indices(names: list[str], samples: list[str]) -> list[int]:
    """Map samples to their integer indices in a given set of names.

    Used to map sample string names to their integer positions in the index of
    the original character matrix for efficient indexing operations.

    Args:
        names: A list of sample names, represented by their string names in the
            original character matrix.
        samples: A list of sample names representing the subset to be mapped to
            integer indices.

    Returns:
        A list of samples mapped to integer indices.
    """
    name_to_index = dict(zip(names, range(len(names)), strict=False))

    return [name_to_index[x] for x in samples]


def _assign_missing_average(
    character_matrix: pd.DataFrame,
    missing_state_indicator: int,
    left_set: list[str],
    right_set: list[str],
    missing: list[str],
    weights: dict[int, dict[int, float]] | None = None,
) -> tuple[list[str], list[str]]:
    """Implement the "average" missing-data imputation method.

    An on-the-fly missing data imputation method for Cassiopeia-Greedy. It takes
    in a set of samples that have a missing value at the character chosen to
    split on in a partition. For each of these samples, it calculates the average
    number of mutations that samples on each side of the partition share with it
    and places the sample on the side with the higher value.

    Args:
        character_matrix: The character matrix containing the observed character
            states for the samples.
        missing_state_indicator: The character representing missing values.
        left_set: A list of the samples on the left of the partition, represented
            by their names in the original character matrix.
        right_set: A list of the samples on the right of the partition,
            represented by their names in the original character matrix.
        missing: A list of samples with missing data to be imputed, represented
            by their names in the original character matrix.
        weights: A set of optional weights for character/state mutation pairs.

    Returns:
        A tuple of lists, representing the left and right partitions with missing
        samples imputed.
    """
    # A helper function to calculate the number of shared character/state pairs
    # shared between a missing sample and a side of the partition
    sample_names = list(character_matrix.index)
    character_array = character_matrix.to_numpy()
    left_indices = convert_sample_names_to_indices(sample_names, left_set)
    right_indices = convert_sample_names_to_indices(sample_names, right_set)
    missing_indices = convert_sample_names_to_indices(sample_names, missing)

    def score_side(subset_character_states, query_states, weights):
        score = 0
        for char in range(len(subset_character_states)):
            query_state = [q for q in query_states[char] if q != 0 and q != missing_state_indicator]
            all_states = np.array(subset_character_states[char])
            for q in query_state:
                if weights:
                    score += weights[char][q] * np.count_nonzero(all_states == q)
                else:
                    score += np.count_nonzero(all_states == q)

        return score

    subset_character_array_left = character_array[left_indices, :]
    subset_character_array_right = character_array[right_indices, :]

    all_left_states = [
        unravel_ambiguous_states(subset_character_array_left[:, char])
        for char in range(subset_character_array_left.shape[1])
    ]
    all_right_states = [
        unravel_ambiguous_states(subset_character_array_right[:, char])
        for char in range(subset_character_array_right.shape[1])
    ]

    for sample_index in missing_indices:
        all_states_for_sample = [
            unravel_ambiguous_states([character_array[sample_index, char]])
            for char in range(character_array.shape[1])
        ]

        left_score = score_side(
            np.array(all_left_states, dtype=object),
            np.array(all_states_for_sample, dtype=object),
            weights,
        )
        right_score = score_side(
            np.array(all_right_states, dtype=object),
            np.array(all_states_for_sample, dtype=object),
            weights,
        )

        if (left_score / len(left_set)) > (right_score / len(right_set)):
            left_set.append(sample_names[sample_index])
        else:
            right_set.append(sample_names[sample_index])

    return left_set, right_set


# Registry of missing-data classifiers. Maps a name to a callable that assigns
# samples with missing data at the split character into the left/right partition.
_MISSING_DATA_CLASSIFIERS: dict[str, Callable] = {
    "average": _assign_missing_average,
}


def _resolve_missing_data_classifier(classifier: str | Callable) -> Callable:
    """Resolve a missing-data classifier name or callable to a callable.

    Args:
        classifier: The name of a registered classifier (e.g. ``"average"``) or
            a callable implementing the imputation method directly.

    Raises:
        GreedySolverError: If *classifier* is an unknown name.
    """
    if callable(classifier):
        return classifier
    if classifier not in _MISSING_DATA_CLASSIFIERS:
        raise GreedySolverError(
            f"Unknown missing_data_classifier {classifier!r}. "
            f"Available: {sorted(_MISSING_DATA_CLASSIFIERS)}"
        )
    return _MISSING_DATA_CLASSIFIERS[classifier]


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
    missing_data_classifier: str | Callable = "average",
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
        missing_data_classifier: Name of a registered missing-data classifier
            (e.g. ``"average"``) or a callable assigning samples with missing
            data at the chosen character into the left/right partition.

    Returns:
        A tuple of lists representing the left and right partition groups.
    """
    missing_data_classifier = _resolve_missing_data_classifier(missing_data_classifier)
    sample_indices = convert_sample_names_to_indices(character_matrix.index, samples)
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
    node_name_generator = _node_name_generator()

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
    tdata: TreeData,
    missing_data_classifier: str | Callable = "average",
    characters_key: str | None = None,
    key_added: str = "greedy",
    prior_transformation: str = "negative_log",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict[int, dict[int, float]] | bool = True,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct a tree with vanilla Cassiopeia-Greedy.

    Builds a tree top-down by recursively splitting samples on the most frequent
    mutation. The character matrix is read from ``tdata.obsm`` and the result is
    stored as an ``nx.DiGraph`` in ``tdata.obst[key_added]``.

    Args:
        tdata: TreeData to operate on.
        missing_data_classifier: Name of a registered missing-data classifier
            (e.g. ``"average"``) or a callable assigning samples with missing
            data at the split character into the left/right partition.
        characters_key: Key in ``tdata.obsm`` for the character matrix
            (default ``'characters'``).
        key_added: Key in ``tdata.obst`` for the resulting tree.
        prior_transformation: Transformation applied to priors to form weights.
        missing_state: Missing-state value (read from ``tdata.uns`` if ``None``).
        unmodified_state: Unmodified/uncut state value (read from ``tdata.uns``
            if ``None``).
        priors: Priors for character states. ``True`` (default) reads priors
            from ``tdata.uns["priors"]`` and raises if none are stored; ``False``
            reconstructs without priors; a dict (character index -> {state:
            probability}) is used directly.
        copy: If ``True``, return a copy of *tdata*; otherwise modify in-place
            and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.
    """
    tdata = tdata.copy() if copy else tdata
    character_matrix = _get_characters(tdata, characters_key).copy()
    missing_state = _get_parameter(tdata, "missing_state", value=missing_state)
    unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    priors = _resolve_priors(tdata, priors)
    character_matrix, missing_state, mapping = _encode_integer_matrix(
        character_matrix, missing_state, unmodified_state
    )
    # Re-key prior state values to match the integer encoding of the matrix.
    priors = _encode_priors(priors, mapping)

    weights = None
    if priors:
        weights = _transform_priors(priors, prior_transformation)

    split_fn = functools.partial(_greedy_split, missing_data_classifier=missing_data_classifier)

    tree = _greedy_solve(
        character_matrix,
        split_fn,
        missing_state_indicator=missing_state,
        weights=weights,
        allow_ambiguous=True,
    )

    _set_tree(tdata, tree, key_added)

    return tdata if copy else None


# ── Backward-compat shim ─────────────────────────────────────────────────────


class VanillaGreedySolver:
    """A class for the basic Cassiopeia-Greedy solver.

    Thin shim around :func:`cassiopeia.solver.greedy` for backward
    compatibility.  For new code, prefer calling :func:`cassiopeia.solver.greedy`
    directly.

    Args:
        missing_data_classifier: Name of a registered missing-data classifier
            (e.g. ``"average"``) or a callable implementing the imputation
            method.  Defaults to the ``"average"`` method.
        prior_transformation: Transformation applied to priors to form weights.
    """

    def __init__(
        self,
        missing_data_classifier: str | Callable = "average",
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
        # Preserve legacy behavior: use priors if the tree carries them, else not.
        greedy(
            cassiopeia_tree,
            characters_key=layer,
            missing_data_classifier=self.missing_data_classifier,
            prior_transformation=self.prior_transformation,
            priors=_get_parameter(cassiopeia_tree, "priors") or False,
        )
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)

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
