"""
Cython utilities for the ILPSolver.
"""
import cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.string cimport PyString_AsString


import logging
from typing import Dict, List, Tuple, Optional, Union

# we need to doubly-import numpy so we have access to cython-optimized numpy
# functionality
from numpy cimport ndarray
cimport numpy as cnp
import numpy as np

from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import dissimilarity_functions


# @cython.boundscheck(False)
# @cython.wraparound(False)
def infer_potential_graph_cython(
    character_array: str[:, :],
    pid: Union[int, str],
    lca_height: int,
    maximum_potential_graph_layer_size: int,
    missing_state_indicator: int,
):

    logging.info(
        f"(Process: {pid}) Estimating a potential graph with "
        "a maximum layer size of "
        f"{maximum_potential_graph_layer_size} and a maximum "
        f"LCA height of {lca_height}."
    )

    cdef int effective_threshold
    cdef int max_layer_width
    # cdef ndarray[str, ndim=1] source_nodes

    cdef list layer_sizes = []
    cdef list distance_thresholds = []
    cdef list previous_layer_edges = []
    cdef list current_layer_edges = []

    cdef int n_characters = character_array.shape[1]

    cdef int distance_threshold = 0
    
    # cast character states to strings.
    # Also, because we've converted states into strings, missing data items
    # that were negative numbers will take up two characters. Thus,
    # we convert missing data to a universal character "-"
    character_states = np.array(["|".join(arr).replace(str(missing_state_indicator), "-") for arr in np.array(character_array)])

    while distance_threshold < (lca_height + 1):

        current_layer_edges = []

        source_nodes = character_states
        effective_threshold = distance_threshold
        max_layer_width = 0

        while source_nodes.shape[0] > 1:

            if source_nodes.shape[0] > maximum_potential_graph_layer_size:
                logging.info(
                    f"(Process: {pid}) Maximum layer size "
                    "exceeded, returning network."
                )

                return previous_layer_edges

            (next_layer, layer_edges) = infer_layer_of_potential_graph(
                source_nodes, effective_threshold
            )
            
            # subset to unique values
            if next_layer.shape[0] > 0:
                next_layer = np.unique(next_layer)
                # unique_idx = fast_unique(np.array([node.split("|") for node in next_layer]).astype(int))
                # next_layer = next_layer[unique_idx]

            if (
                next_layer.shape[0] > maximum_potential_graph_layer_size
                and len(previous_layer_edges) > 0
            ):
                return previous_layer_edges

            # edges come out as rows in a numpy matrix, where the first
            # (2*n_characters-1) positions correspond to the parent and the
            # last (2*n_characters) positions correspond to the child
            layer_edges = [
                ("|".join(e.split("|")[:n_characters]), "|".join(e.split("|")[n_characters:]))
                for e in layer_edges
                if "|".join(e.split("|")[:n_characters]) != "|".join(e.split("|")[n_characters:])
            ]
            current_layer_edges += layer_edges
        
            if source_nodes.shape[0] > next_layer.shape[0]:
                if effective_threshold == distance_threshold:
                    effective_threshold *= 3
            source_nodes = next_layer

            max_layer_width = max(max_layer_width, source_nodes.shape[0])


        logging.info(
            f"(Process: {pid}) LCA distance {distance_threshold} "
            f"completed with a neighborthood size of {max_layer_width}."
        )

        distance_thresholds.append(distance_threshold)

        distance_threshold += 1

        layer_sizes.append(max_layer_width)
        
        previous_layer_edges = current_layer_edges
    
    return current_layer_edges


# @cython.boundscheck(False)
# @cython.wraparound(False)
def infer_layer_of_potential_graph(
    source_nodes: ndarray[str],
    distance_threshold: int,
) -> Tuple[np.array, List[Tuple[np.array, np.array]]]:
    """Infer a layer of the potential graph.

    This function is invoked by `infer_potential_graph` and returns a layer
    of samples that represent evolutionary ancestors of the passed in source
    nodes. Ancestors are added to the layer if the distance from a pair of 
    source nodes to the ancestor is less than the specified distance
    threshold. The function returns a set of nodes to use as the next layer
    of the potential graph as well as the edges to add to the potential
    graph.

    The edge representation here is a bit unorthodox, as a way of avoiding
    the use of tuples. The edges are returned as concatenated numpy arrays of 
    length 2M (where M is the number of characters). The first M characters
    correspond to the first node in the edge and the latter half correspond
    to the second node.

    Args:
        source_nodes: A list of samples, represented by their character
            states.
        distance_threshold: Maximum hamming distance allowed between a pair of 
            source nodes through their ancestor.

    Returns:
        A list of samples to be treated as the source nodes of the next
            layer and the edges connecting samples to one another.
    """

    cdef int i, j, k, d1_a, d2_a, min_distance_to_ancestor
    cdef int dim = len(source_nodes[0].split("|"))
    cdef int n_samples = source_nodes.shape[0]

    cdef str ancestor, edge, edge1, edge2, parent, sample1, sample2
    cdef long[:] distance_to_ancestor,
    cdef list top_ancestors

    cdef list layer = []
    cdef list new_edges = []

    for i in range(0, n_samples-1):

        sample1 = str(source_nodes[i])
        top_ancestors = []
        distance_to_ancestors = []

        for j in range(i + 1, n_samples):

            sample2 = str(source_nodes[j])
            ancestor = get_lca_characters_cython(
                np.array(sample1.split("|")),
                np.array(sample2.split("|")),
                dim,
                "-"
            )

            d1_a = simple_hamming_distance_cython(
                np.array(ancestor.split("|")),
                np.array(sample1.split("|")),
                "-"
            )

            d2_a = simple_hamming_distance_cython(
                np.array(ancestor.split("|")),
                np.array(sample2.split("|")),
                "-"
            )

            # edge = np.array([ancestor, sample2]).flatten()
            edge = (ancestor + "|" + sample2)
            top_ancestors.append(edge)
            distance_to_ancestors.append(d1_a + d2_a)

            if d1_a + d2_a < distance_threshold:
                
                edge1 = (ancestor + "|" + sample1)
                edge2 = (ancestor + "|" + sample2)
                
                new_edges.append(edge1)
                new_edges.append(edge2)

                layer.append(ancestor)

        # enforce adding at least one edge between layers for each sample:
        # find the pair of nodes that have the lowest LCA and add this
        # to the list of new nodes in the layer
        min_distance_to_ancestor = min(distance_to_ancestors)
        for k in range(len(top_ancestors)):
            if distance_to_ancestors[k] == min_distance_to_ancestor:
                
                # The first (2*dim-1) characters correspond to the parent
                parent = "|".join(top_ancestors[k].split("|")[:dim])
                edge2 = top_ancestors[k]
                edge1 = (parent + "|" + sample1)

                new_edges.append(edge1)
                new_edges.append(edge2)

                layer.append(parent)

    return np.array(layer), np.array(new_edges)

# @cython.boundscheck(False)
# @cython.wraparound(False)
def get_lca_characters_cython(
    arr1: np.array(),
    arr2: np.array(),
    n_char: int,
    missing_state_indicator: str,
) -> str:

    cdef list ancestor = ["0"]*n_char
    cdef int i

    for i in range(n_char):
        
        if arr1[i] == arr2[i]:
            ancestor[i] = arr1[i]

        else:
            if arr1[i] == missing_state_indicator:
                ancestor[i] = arr2[i]
            elif arr2[i] == missing_state_indicator:
                ancestor[i] = arr1[i]

    return "|".join(ancestor)

# @cython.boundscheck(False)
# @cython.wraparound(False)
def simple_hamming_distance_cython(
    arr1: np.array(),
    arr2: np.array(),
    missing_state_indicator: str
):

    cdef int i
    cdef int count = 0
    for i in range(len(arr1)):
        
        if (arr1[i] != arr2[i]) and (arr1[i] != missing_state_indicator) and (arr2[i] != missing_state_indicator):
            count += 1

    return count

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def fast_unique(ndarray[str, ndim=1] a):
#     cdef int i, len_before
#     cdef int nr = a.shape[0]
#     cdef int nc = len(a[0].split("|"))
#     cdef set s = set()
#     cdef ndarray[uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')
#     cdef bytes string

#     for i in range(nr):
#         len_before = len(s)
#         _string = PyBytes_FromStringAndSize(<char*>&a[i], sizeof(long) * nc)
#         s.add(_string)
#         if len(s) > len_before:
#             idx[i] = True
#     return idx

# cdef char ** to_cstring_array(list_str):
#     cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
#     for i in xrange(len(list_str)):
#         ret[i] = PyString_AsString(list_str[i])
#     return ret
