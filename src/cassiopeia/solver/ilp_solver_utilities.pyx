"""
Cython utilities for the ILPSolver.
"""
import cython
from typing import List, Tuple, Union

import numpy as np

from cassiopeia.data import utilities as data_utilities  # noqa: F401 (if unused)
from cassiopeia.mixins import logger
from cassiopeia.solver import dissimilarity_functions  # noqa: F401 (if unused)


@cython.boundscheck(False)
@cython.wraparound(False)
def infer_potential_graph_cython(
    character_array,
    pid: Union[int, str],
    maximum_lca_distance: int,
    maximum_potential_graph_layer_size: int,
    missing_state_indicator: int,
) -> list[tuple[str, str]]:
    """Cython-optimized potential graph inference (NumPy-2-safe).

    Args:
        character_array: 2D array-like of strings; items are states.
        pid: Unique process ID used for logging purposes.
        maximum_lca_distance: Maximum LCA height to use when inferring ancestors.
        maximum_potential_graph_layer_size: Max nodes in a layer of the potential graph.
        missing_state_indicator: Indicator for missing data (int), converted to "-".
    """
    logger.info(
        f"(Process: {pid}) Estimating a potential graph with a maximum layer size of "
        f"{maximum_potential_graph_layer_size} and a maximum LCA distance of {maximum_lca_distance}."
    )

    # Ensure ndarray of str
    carr = np.asarray(character_array, dtype=str)
    cdef int n_characters = carr.shape[1]

    # Cast character states to a compact string representation, normalizing missing to "-"
    character_states = np.array(
        [
            "|".join(row).replace(str(missing_state_indicator), "-")
            for row in carr
        ],
        dtype=str,
    )

    cdef int effective_threshold
    cdef int max_layer_width
    cdef list layer_sizes = []
    cdef list distance_thresholds = []
    cdef list previous_layer_edges = []
    cdef list current_layer_edges = []
    cdef int distance_threshold = 0

    while distance_threshold < (maximum_lca_distance + 1):
        current_layer_edges = []

        source_nodes = character_states
        effective_threshold = distance_threshold
        max_layer_width = 0

        while source_nodes.shape[0] > 1:
            if source_nodes.shape[0] > maximum_potential_graph_layer_size:
                logger.info(f"(Process: {pid}) Maximum layer size exceeded, returning network.")
                return previous_layer_edges

            next_layer, layer_edges = infer_layer_of_potential_graph(
                source_nodes, effective_threshold
            )

            if (next_layer.shape[0] > maximum_potential_graph_layer_size) and (len(previous_layer_edges) > 0):
                return previous_layer_edges

            # Convert concatenated "parent|...|child" strings into edge tuples
            edges_tuples = []
            for e in layer_edges:
                parts = e.split("|")
                parent = "|".join(parts[:n_characters])
                child = "|".join(parts[n_characters:])
                if parent != child:
                    edges_tuples.append((parent, child))
            current_layer_edges += edges_tuples

            if source_nodes.shape[0] > next_layer.shape[0]:
                if effective_threshold == distance_threshold:
                    effective_threshold *= 3

            source_nodes = next_layer
            if source_nodes.shape[0] > max_layer_width:
                max_layer_width = source_nodes.shape[0]

        logger.info(
            f"(Process: {pid}) LCA distance {distance_threshold} completed with a neighborhood size of {max_layer_width}."
        )

        distance_thresholds.append(distance_threshold)
        distance_threshold += 1
        layer_sizes.append(max_layer_width)
        previous_layer_edges = current_layer_edges

    return current_layer_edges


@cython.boundscheck(False)
@cython.wraparound(False)
def infer_layer_of_potential_graph(
    source_nodes,
    distance_threshold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Infer a layer of the potential graph.

    Args:
        source_nodes: 1D array-like[str], character strings ("a|b|c|").
        distance_threshold: Max Hamming distance via the ancestor.
    Returns:
        next_layer: np.ndarray[str] of ancestor node strings for the next layer.
        new_edges:  np.ndarray[str] of concatenated "parent|...|child" strings.
    """
    source_nodes = np.asarray(source_nodes, dtype=str)
    cdef int n_samples = source_nodes.shape[0]
    if n_samples == 0:
        return np.empty((0,), dtype=str), np.empty((0,), dtype=str)

    # Infer dimension from first node string
    first = str(source_nodes[0])
    cdef int dim = len(first.split("|"))
    cdef int i, j, k
    cdef int d1_a, d2_a, min_distance_to_ancestor

    cdef str ancestor, edge, edge1, edge2, parent, sample1, sample2
    cdef list top_ancestors
    cdef list distance_to_ancestors

    cdef set layer = set()
    cdef set new_edges = set()

    for i in range(0, n_samples - 1):
        sample1 = str(source_nodes[i])
        top_ancestors = []
        distance_to_ancestors = []

        for j in range(i + 1, n_samples):
            sample2 = str(source_nodes[j])

            ancestor = get_lca_characters_cython(
                np.array(sample1.split("|"), dtype=str),
                np.array(sample2.split("|"), dtype=str),
                dim,
                "-",
            )

            d1_a = simple_hamming_distance_cython(
                np.array(ancestor.split("|"), dtype=str),
                np.array(sample1.split("|"), dtype=str),
                "-"
            )

            d2_a = simple_hamming_distance_cython(
                np.array(ancestor.split("|"), dtype=str),
                np.array(sample2.split("|"), dtype=str),
                "-"
            )

            # store concatenated parent|...|child string
            edge = ancestor + "|" + sample2
            top_ancestors.append(edge)
            distance_to_ancestors.append(d1_a + d2_a)

            if d1_a + d2_a < distance_threshold:
                edge1 = ancestor + "|" + sample1
                edge2 = ancestor + "|" + sample2
                new_edges.add(edge1)
                new_edges.add(edge2)
                layer.add(ancestor)

        if distance_to_ancestors:
            # enforce at least one edge via best (closest) ancestor
            min_distance_to_ancestor = min(distance_to_ancestors)
            for k in range(len(top_ancestors)):
                if distance_to_ancestors[k] == min_distance_to_ancestor:
                    parent = "|".join(top_ancestors[k].split("|")[:dim])
                    edge2 = top_ancestors[k]
                    edge1 = parent + "|" + sample1
                    new_edges.add(edge1)
                    new_edges.add(edge2)
                    layer.add(parent)

    return np.array(list(layer), dtype=str), np.array(list(new_edges), dtype=str)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_lca_characters_cython(
    arr1,
    arr2,
    n_char: int,
    missing_state_indicator: str,
) -> str:
    """Cython-optimized LCA inference using strings."""
    # Ensure indexable sequences of str
    a1 = np.asarray(arr1, dtype=str)
    a2 = np.asarray(arr2, dtype=str)

    cdef list ancestor = ["0"] * n_char
    cdef int i

    for i in range(n_char):
        if a1[i] == a2[i]:
            ancestor[i] = a1[i]
        else:
            if a1[i] == missing_state_indicator:
                ancestor[i] = a2[i]
            elif a2[i] == missing_state_indicator:
                ancestor[i] = a1[i]
            # else leave as "0"

    return "|".join(ancestor)


@cython.boundscheck(False)
@cython.wraparound(False)
def simple_hamming_distance_cython(arr1, arr2, missing_state_indicator: str) -> int:
    """Hamming distance ignoring missing values."""
    a1 = np.asarray(arr1, dtype=str)
    a2 = np.asarray(arr2, dtype=str)

    if a1.shape[0] != a2.shape[0]:
        raise ValueError("Arrays must be the same length")

    cdef int i
    cdef int count = 0
    cdef Py_ssize_t n = a1.shape[0]

    for i in range(n):
        if (a1[i] != a2[i]) and (a1[i] != missing_state_indicator) and (a2[i] != missing_state_indicator):
            count += 1
    return count
