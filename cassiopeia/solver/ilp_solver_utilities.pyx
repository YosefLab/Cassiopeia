"""
Cython utilities for the ILPSolver.
"""
import cython

import logging
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import dissimilarity_functions


@cython.boundscheck(False)
def infer_potential_graph_cython(
    character_states: long[:,:],
    pid: Union[int, str],
    lca_height: int,
    maximum_potential_graph_layer_size: int,
    missing_state_indicator: int):

    logging.info(
        f"(Process: {pid}) Estimating a potential graph with "
        "a maximum layer size of "
        f"{maximum_potential_graph_layer_size} and n maximum "
        f"LCA height of {lca_height}."
    )

    cdef int effective_threshold
    cdef int max_layer_width
    cdef long[:,:] source_nodes

    cdef list layer_sizes = []
    cdef list distance_thresholds = []
    cdef list previous_layer_edges = []
    cdef list current_layer_edges = []

    cdef int n_characters = character_states.shape[1]

    cdef int distance_threshold = 0
    cdef int layer_iterator = 0

    while distance_threshold < (lca_height + 1):
        
        current_layer_edges = [-1]*1000000 # allocate sufficient memory

        source_nodes = character_states
        effective_threshold = distance_threshold
        max_layer_width = 0
        layer_iterator = 0

        while source_nodes.shape[0] > 1:

            if source_nodes.shape[0] > maximum_potential_graph_layer_size:
                logging.info(
                    f"(Process: {pid}) Maximum layer size "
                    "exceeded, returning network."
                )

                return previous_layer_edges

            (
                next_layer,
                layer_edges,
            ) = infer_layer_of_potential_graph(
                source_nodes, effective_threshold, missing_state_indicator
            )

            # subset to unique values
            if next_layer.shape[0] > 0:
                next_layer = np.unique(next_layer, axis=0)

            if (
                next_layer.shape[0] > maximum_potential_graph_layer_size
                and len(previous_layer_edges) > 0
            ):
                return previous_layer_edges

            # edges come out as rows in a numpy matrix, where the first
            # n_characters positions correspond to the parent and the
            # remaining positions correspond to the child
            layer_edges = [
                (tuple(e[:n_characters]), tuple(e[n_characters:]))
                for e in layer_edges
                if tuple(e[:n_characters]) != tuple(e[n_characters:])
            ]
            for layer_edge in layer_edges:
                current_layer_edges[layer_iterator] = layer_edge
                layer_iterator += 1

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
        
        previous_layer_edges = current_layer_edges[:layer_iterator]

    return current_layer_edges[:layer_iterator]

@cython.boundscheck(False)
@cython.wraparound(False)
def infer_layer_of_potential_graph(
    source_nodes: long[:,:], distance_threshold: int, missing_state_indicator: int
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
        missing_state_indicator: State to treat as missing.

    Returns:
        A list of samples to be treated as the source nodes of the next
            layer and the edges connecting samples to one another.
    """
    
    cdef int i, j, k, d1_a, d2_a, min_distance_to_ancestor
    cdef int dim = source_nodes.shape[1]
    cdef int n_samples = source_nodes.shape[0]
    
    cdef long[:] ancestor, distance_to_ancestor, parent, sample1, sample2
    cdef list top_ancestors

    cdef list layer = [-1]*1000000
    cdef list new_edges = [-1]*1000000
    cdef long[:] edge
    
    cdef int edge_iterator = 0
    cdef int layer_iterator = 0

    # preallocate edges
    cdef long[:] edge1 = np.array([0]*dim, long)
    cdef long[:] edge2 = np.array([0]*dim, long)
    for i in range(0, n_samples-1):

        sample1 = source_nodes[i]
        top_ancestors = []
        distance_to_ancestors = []

        for j in range(i + 1, n_samples):

            sample2 = source_nodes[j]
            ancestor = get_lca_characters_cython(sample1, sample2, len(sample1), missing_state_indicator)

            d1_a = dissimilarity_functions.hamming_distance(ancestor,
                    sample1,
                    ignore_missing_state=True,
                    missing_state_indicator=missing_state_indicator)
            d2_a = dissimilarity_functions.hamming_distance(ancestor,
                    sample2,
                    ignore_missing_state=True,
                    missing_state_indicator=missing_state_indicator)

            edge = np.concatenate((ancestor, sample2))
            top_ancestors.append(edge)
            distance_to_ancestors.append(d1_a + d2_a)

            if d1_a + d2_a <= distance_threshold:
                
                edge1 = np.concatenate((ancestor, sample1))
                edge2 = np.concatenate((ancestor, sample2))

                new_edges[edge_iterator] = edge1
                new_edges[edge_iterator+1] = edge2
                edge_iterator += 2

                layer[layer_iterator] = ancestor
                layer_iterator += 1

        # enforce adding at least one edge between layers for each sample:
        # find the pair of nodes that have the lowest LCA and add this
        # to the list of new nodes in the layer
        min_distance_to_ancestor = min(distance_to_ancestors)
        for k in range(len(top_ancestors)):
            if distance_to_ancestors[k] == min_distance_to_ancestor:

                parent = top_ancestors[k][:dim]
                edge2 = top_ancestors[k]
                edge1 = np.concatenate((parent, sample1))
                
                new_edges[edge_iterator] = edge1
                new_edges[edge_iterator+1] = edge2
                edge_iterator += 2

                layer[layer_iterator] = parent
                layer_iterator += 1

    return np.array(layer[:layer_iterator]), np.array(new_edges[:edge_iterator])

def get_lca_characters_cython(arr1: np.array(), arr2: np.array(), n_char: int, missing_state_indicator: int) -> np.array:

    cdef long[:] ancestor = np.zeros((n_char,), dtype=long)
    cdef int i

    for i in range(n_char):

        if arr1[i] == arr2[i]:
            ancestor[i] = arr1[i]

        else:
            if arr1[i] == missing_state_indicator:
                ancestor[i] = arr2[i]
            elif arr2[i] == missing_state_indicator:
                ancestor[i] = arr1[i]
    return ancestor