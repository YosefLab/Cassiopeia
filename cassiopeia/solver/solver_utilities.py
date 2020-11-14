"""This file contains general utilities to be called by functions throughout 
the solver module"""

import logging

import numpy as np
import pandas as pd
import networkx as nx

from typing import Dict, List, Optional, Tuple


class InferAncestorError(Exception):
    """An Exception class for collapsing edges, indicating a necessary argument
    was not included.
    """

    pass


def get_lca_characters(vecs: List[List[str]], missing_char: str) -> List[str]:
    """Builds the character vector of the LCA of a list of character vectors,
    obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_char: The character representing missing values

    Returns:
        A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = ["0"] * len(vecs[0])
    for i in range(k):
        chars = [vec[i] for vec in vecs]
        if len(set(chars)) == 1:
            lca_vec[i] = chars[0]
        else:
            if missing_char in chars:
                chars.remove(missing_char)
                if len(set(chars)) == 1:
                    lca_vec[i] = chars[0]
    return lca_vec


def annotate_ancestral_characters(
    T: nx.DiGraph, node: int, char_map: Dict[int, List[str]], missing_char: str
):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.

    For an internal node, annotates that node's character vector to be the LCA
    of its daughter character vectors. Annotates from the samples upwards.

    Args:
        T: A networkx DiGraph object representing the tree
        node: The node whose state is to be inferred
        char_map: A dictionary that maps nodes to their character vectors
        missing_char: The character representing missing values

    Returns:
        None, annotates char_map dictionary with node/character vector pairs


    """
    if T.out_degree(node) == 0:
        return
    vecs = []
    for i in T.successors(node):
        annotate_ancestral_characters(T, i, char_map, missing_char)
        vecs.append(char_map[i])
    lca_characters = get_lca_characters(vecs, missing_char)
    char_map[node] = lca_characters


def collapse_edges(T: nx.DiGraph, node: int, char_map: Dict[int, List[str]]):
    """A helper function to collapse mutationless edges in a tree in-place.

    Collapses an edge if the character vector of the parent node is identical
    to its daughter, removing the identical daughter and creating edges between
    the parent and the daughter's children. Does not collapse at the level of
    the samples. Can create multifurcating trees from strictly binary trees.

    Args:
        T: A networkx DiGraph object representing the tree
        node: The node whose state is to be inferred
        char_map: A dictionary that maps nodes to their character vectors

    Returns:
        None, operates on the tree destructively
    """
    if T.out_degree(node) == 0:
        return
    to_remove = []
    to_collapse = []
    for i in T.successors(node):
        to_collapse.append(i)
    for i in to_collapse:
        if T.out_degree(i) > 0:
            collapse_edges(T, i, char_map)
            if char_map[i] == char_map[node]:
                for j in T.successors(i):
                    T.add_edge(node, j)
                to_remove.append(i)
    for i in to_remove:
        T.remove_node(i)


def collapse_tree(
    T: nx.DiGraph,
    infer_ancestral_characters: bool,
    cm: pd.DataFrame = None,
    missing_char: str = None,
):
    """Collapses mutationless edges in a tree in-place.

    Uses the internal node annotations of a tree to collapse edges with no
    mutations. Either takes in a tree with internal node annotations or
    a tree without annotations and infers the annotations bottom-up from the
    samples obeying Camin-Sokal Parsimony. If ground truth internal annotations
    exist, it is suggested that they are used directly and that the annotations
    are not inferred again using the parsimony method.

    Args:
        network: A networkx DiGraph object representing the tree

    Returns:
        None, operates on the tree destructively

    """
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    root = [n for n in T if T.in_degree(n) == 0][0]
    char_map = {}

    # Populates the internal annotations using either the ground truth
    # annotations, or infers them
    if infer_ancestral_characters:
        if cm is None or missing_char is None:
            logging.info(
                "In order to infer ancestral characters, a character matrix and missing character are needed"
            )
            raise InferAncestorError()

        for i in leaves:
            char_map[i] = list(cm.iloc[i, :])
        annotate_ancestral_characters(T, root, char_map, missing_char)
    else:
        for i in T.nodes():
            char_map[i] = i.char_vec

    # Calls helper function on root, passing in the mapping dictionary
    collapse_edges(T, root, char_map)


def post_process_tree(T, cm):
    # raise NotImplementedError()
    pass
