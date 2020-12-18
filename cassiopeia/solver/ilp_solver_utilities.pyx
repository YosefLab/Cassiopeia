"""
Cython utilities for ILPSolver.
"""
cimport cython

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import solver_utilities


@cython.boundscheck(False)
@cython.wraparound(False)
def infer_layer_of_potential_graph(
    long[:,:] source_nodes, int distance_threshold, int missing_char
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
        missing_char: State to treat as missing.

    Returns:
        A list of samples to be treated as the source nodes of the next
            layer and the edges connecting samples to one another.
    """
    
    cdef int i, j, k, d1_a, d2_a, min_distance_to_ancestor
    cdef int dim = source_nodes.shape[1]
    cdef int n_samples = source_nodes.shape[0]
    
    cdef long[:] ancestor, distance_to_ancestor, parent, sample1, sample2
    cdef list top_ancestors

    cdef list layer = []
    cdef list new_edges = []
    cdef long[:] edge
    
    for i in range(0, n_samples-1):

        sample1 = source_nodes[i]
        top_ancestors = []
        distance_to_ancestors = []

        for j in range(i + 1, n_samples):

            sample2 = source_nodes[j]
            # ancestor = np.array([sample1[k] if sample1[k] == sample2[k] else 0 for k in range(dim)])
            ancestor = np.array(solver_utilities.get_lca_characters([sample1, sample2], missing_char=missing_char))

            d1_a = dissimilarity_functions.hamming_distance(sample1, ancestor)
            d2_a = dissimilarity_functions.hamming_distance(sample2, ancestor)

            edge = np.concatenate((ancestor, sample2))
            top_ancestors.append(edge)
            distance_to_ancestors.append(d1_a + d2_a)

            if d1_a + d2_a < distance_threshold:

                edge1 = np.concatenate((ancestor, sample1))
                edge2 = np.concatenate((ancestor, sample2))

                new_edges.append(edge1)
                new_edges.append(edge2)

                layer.append(ancestor)

        # enforce adding at least one edge between layers for each sample:
        # find the pair of nodes that have the lowest LCA and add this
        # to the list of new nodes in the layer
        min_distance_to_ancestor = min(distance_to_ancestors)
        for k in range(len(top_ancestors)):
            if distance_to_ancestors[k] == min_distance_to_ancestor:

                parent = top_ancestors[k][:dim]
                edge2 = top_ancestors[k]
                edge1 = np.concatenate((parent, sample1))

                new_edges.append(edge1)
                new_edges.append(edge2)

                layer.append(parent)

    return np.array(layer), np.array(new_edges)