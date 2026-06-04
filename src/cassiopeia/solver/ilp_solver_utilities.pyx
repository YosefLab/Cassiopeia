"""Cython utilities for the ILPSolver (integer character states).

The potential graph is inferred over integer character vectors using typed-memoryview
C loops (LCA + Hamming distance).  The pairwise distances for the initial layer are
computed once and reused across the LCA-distance-threshold sweep.
"""

import cython

import numpy as np

cimport numpy as cnp

from cassiopeia.mixins import logger

ctypedef cnp.int64_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _lca(INT[:] a, INT[:] b, INT[:] out, Py_ssize_t nc, INT missing):
    """LCA of two character vectors under Camin-Sokal parsimony, into *out*."""
    cdef Py_ssize_t i
    for i in range(nc):
        if a[i] == b[i]:
            out[i] = a[i]
        elif a[i] == missing:
            out[i] = b[i]
        elif b[i] == missing:
            out[i] = a[i]
        else:
            out[i] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _hamming(INT[:] a, INT[:] b, Py_ssize_t nc, INT missing):
    """Hamming distance between two vectors, ignoring missing positions."""
    cdef Py_ssize_t i
    cdef int c = 0
    for i in range(nc):
        if a[i] != b[i] and a[i] != missing and b[i] != missing:
            c += 1
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _pairwise_distances(INT[:, ::1] s, INT[:, ::1] dist, Py_ssize_t n, Py_ssize_t nc, INT missing):
    """Fill ``dist[i, j]`` (for j > i) with ``hamming(lca(i,j), i) + hamming(lca(i,j), j)``."""
    cdef cnp.ndarray[INT, ndim=1] anc_arr = np.empty(nc, dtype=np.int64)
    cdef INT[:] anc = anc_arr
    cdef Py_ssize_t i, j
    for i in range(n - 1):
        for j in range(i + 1, n):
            _lca(s[i], s[j], anc, nc, missing)
            dist[i, j] = _hamming(anc, s[i], nc, missing) + _hamming(anc, s[j], nc, missing)


def _infer_potential_layer(source, distance_threshold, missing, precomputed_dist=None):
    """Infer one layer of ancestors and edges from integer character vectors.

    Args:
        source: 2D int array (n_samples × n_characters).
        distance_threshold: Pairs whose summed LCA distance is below this add edges.
        missing: Missing-state indicator (int).
        precomputed_dist: Optional cached ``dist[i, j]`` matrix for *source*; when
            given, the O(n²) pairwise distance computation is skipped.

    Returns:
        ``(next_layer, edges, dist)`` where *next_layer* is a 2D int array of ancestor
        vectors, *edges* is a set of ``(parent_tuple, child_tuple)`` pairs, and *dist*
        is the (cacheable) pairwise distance matrix.
    """
    cdef cnp.ndarray[INT, ndim=2] src = np.ascontiguousarray(source, dtype=np.int64)
    cdef Py_ssize_t n = src.shape[0]
    cdef Py_ssize_t nc = src.shape[1]
    cdef INT miss = missing
    cdef int threshold = distance_threshold

    if n == 0:
        return np.empty((0, nc), dtype=np.int64), set(), np.empty((0, 0), dtype=np.int64)

    cdef INT[:, ::1] s = src

    cdef cnp.ndarray[INT, ndim=2] dist_arr
    if precomputed_dist is None:
        dist_arr = np.empty((n, n), dtype=np.int64)
        _pairwise_distances(s, dist_arr, n, nc, miss)
    else:
        dist_arr = precomputed_dist
    cdef INT[:, ::1] dist = dist_arr

    cdef cnp.ndarray[INT, ndim=1] anc_arr = np.empty(nc, dtype=np.int64)
    cdef INT[:] anc = anc_arr

    source_tuples = [tuple(row) for row in src.tolist()]
    layer = set()
    edges = set()

    cdef Py_ssize_t i, j
    cdef int d, min_d
    for i in range(n - 1):
        # closest ancestor distance for sample i over all j > i
        min_d = dist[i, i + 1]
        for j in range(i + 2, n):
            if dist[i, j] < min_d:
                min_d = dist[i, j]
        # add edges via ancestors below threshold, plus the closest ancestor
        for j in range(i + 1, n):
            d = dist[i, j]
            if d < threshold or d == min_d:
                _lca(s[i], s[j], anc, nc, miss)
                a = tuple(anc_arr.tolist())
                edges.add((a, source_tuples[i]))
                edges.add((a, source_tuples[j]))
                layer.add(a)

    if layer:
        next_layer = np.array(list(layer), dtype=np.int64).reshape(-1, nc)
    else:
        next_layer = np.empty((0, nc), dtype=np.int64)
    return next_layer, edges, dist_arr


def infer_layer_of_potential_graph(source, distance_threshold, missing_state_indicator=-1):
    """Infer a single layer of the potential graph (public wrapper).

    Returns ``(next_layer, edges)`` where *edges* is a list of
    ``(parent_tuple, child_tuple)`` pairs.
    """
    next_layer, edges, _ = _infer_potential_layer(
        source, distance_threshold, missing_state_indicator
    )
    return next_layer, list(edges)


def get_lca_characters_cython(arr1, arr2, n_char, missing_state_indicator):
    """LCA of two integer character vectors (Camin-Sokal)."""
    cdef cnp.ndarray[INT, ndim=1] a = np.ascontiguousarray(arr1, dtype=np.int64)
    cdef cnp.ndarray[INT, ndim=1] b = np.ascontiguousarray(arr2, dtype=np.int64)
    cdef cnp.ndarray[INT, ndim=1] out = np.empty(n_char, dtype=np.int64)
    _lca(a, b, out, n_char, missing_state_indicator)
    return out


def simple_hamming_distance_cython(arr1, arr2, missing_state_indicator):
    """Hamming distance between two integer vectors, ignoring missing positions."""
    cdef cnp.ndarray[INT, ndim=1] a = np.ascontiguousarray(arr1, dtype=np.int64)
    cdef cnp.ndarray[INT, ndim=1] b = np.ascontiguousarray(arr2, dtype=np.int64)
    if a.shape[0] != b.shape[0]:
        raise ValueError("Arrays must be the same length")
    return _hamming(a, b, a.shape[0], missing_state_indicator)


def infer_potential_graph_cython(
    character_matrix,
    pid,
    maximum_lca_distance,
    maximum_potential_graph_layer_size,
    missing_state_indicator,
):
    """Infer the potential graph over integer character vectors.

    Args:
        character_matrix: 2D int array; rows are samples, items are states.
        pid: Process ID used for logging.
        maximum_lca_distance: Maximum LCA distance threshold to sweep over.
        maximum_potential_graph_layer_size: Max nodes allowed in a layer.
        missing_state_indicator: Missing-state indicator (int).

    Returns:
        A list of ``(parent_tuple, child_tuple)`` integer edges.
    """
    logger.info(
        f"(Process: {pid}) Estimating a potential graph with a maximum layer size of "
        f"{maximum_potential_graph_layer_size} and a maximum LCA distance of {maximum_lca_distance}."
    )

    character_states = np.ascontiguousarray(character_matrix, dtype=np.int64)
    cdef INT miss = missing_state_indicator

    previous_layer_edges = []
    current_layer_edges = []
    first_layer_dist = None  # cached pairwise distances for the initial layer (#2)

    cdef int distance_threshold = 0
    cdef int effective_threshold
    cdef int max_layer_width
    cdef bint is_first

    while distance_threshold < (maximum_lca_distance + 1):
        current_layer_edges = []
        source_nodes = character_states
        is_first = True
        effective_threshold = distance_threshold
        max_layer_width = 0

        while source_nodes.shape[0] > 1:
            if source_nodes.shape[0] > maximum_potential_graph_layer_size:
                logger.info(f"(Process: {pid}) Maximum layer size exceeded, returning network.")
                return previous_layer_edges

            if is_first:
                next_layer, layer_edges, first_layer_dist = _infer_potential_layer(
                    source_nodes, effective_threshold, miss, first_layer_dist
                )
            else:
                next_layer, layer_edges, _ = _infer_potential_layer(
                    source_nodes, effective_threshold, miss, None
                )

            if (next_layer.shape[0] > maximum_potential_graph_layer_size) and (
                len(previous_layer_edges) > 0
            ):
                return previous_layer_edges

            current_layer_edges += [(p, c) for (p, c) in layer_edges if p != c]

            if source_nodes.shape[0] > next_layer.shape[0]:
                if effective_threshold == distance_threshold:
                    effective_threshold *= 3

            source_nodes = next_layer
            is_first = False
            if source_nodes.shape[0] > max_layer_width:
                max_layer_width = source_nodes.shape[0]

        logger.info(
            f"(Process: {pid}) LCA distance {distance_threshold} completed with a "
            f"neighborhood size of {max_layer_width}."
        )

        distance_threshold += 1
        previous_layer_edges = current_layer_edges

    return current_layer_edges
