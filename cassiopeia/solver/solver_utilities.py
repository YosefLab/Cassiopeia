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
    T: nx.DiGraph, node: int, node_to_characters: Dict[int, List[str]], missing_char: str
):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.

    For an internal node, annotates that node's character vector to be the LCA
    of its daughter character vectors. Annotates from the samples upwards.

    Args:
        T: A networkx DiGraph object representing the tree
        node: The node whose state is to be inferred
        node_to_characters: A dictionary that maps nodes to their character vectors
        missing_char: The character representing missing values

    Returns:
        None, annotates node_to_characters dictionary with node/character vector pairs


    """
    if T.out_degree(node) == 0:
        return
    vecs = []
    for i in T.successors(node):
        annotate_ancestral_characters(T, i, node_to_characters, missing_char)
        vecs.append(node_to_characters[i])
    lca_characters = get_lca_characters(vecs, missing_char)
    node_to_characters[node] = lca_characters
    T.nodes[node]['characters'] = lca_characters


def collapse_edges(T: nx.DiGraph, node: int, node_to_characters: Dict[int, List[str]]):
    """A helper function to collapse mutationless edges in a tree in-place.

    Collapses an edge if the character vector of the parent node is identical
    to its daughter, removing the identical daughter and creating edges between
    the parent and the daughter's children. Does not collapse at the level of
    the samples. Can create multifurcating trees from strictly binary trees.

    Args:
        T: A networkx DiGraph object representing the tree
        node: The node whose state is to be inferred
        node_to_characters: A dictionary that maps nodes to their character vectors

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
            collapse_edges(T, i, node_to_characters)
            if node_to_characters[i] == node_to_characters[node]:
                for j in T.successors(i):
                    T.add_edge(node, j)
                to_remove.append(i)
    for i in to_remove:
        T.remove_node(i)


def collapse_tree(
    T: nx.DiGraph,
    infer_ancestral_characters: bool,
    character_matrix: pd.DataFrame = None,
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
        T: A networkx DiGraph object representing the tree
        infer_ancestral_characters: Infer the ancestral characters states of
            the tree
        character_matrix: A character matrix storing character states for each
            leaf
        missing_char: Character state indicating missing data

    Returns:
        None, operates on the tree destructively

    """
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    root = [n for n in T if T.in_degree(n) == 0][0]
    node_to_characters = {}

    # Populates the internal annotations using either the ground truth
    # annotations, or infers them
    if infer_ancestral_characters:
        if character_matrix is None or missing_char is None:
            logging.info(
                "In order to infer ancestral characters, a character matrix and missing character are needed"
            )
            raise InferAncestorError()

        for i in leaves:
            node_to_characters[i] = list(character_matrix.iloc[i, :])
            T.nodes[i]['characters'] = list(character_matrix.iloc[i, :])
        annotate_ancestral_characters(T, root, node_to_characters, missing_char)
    else:
        for i in T.nodes():
            node_to_characters[i] = T.nodes[i]['characters']

    # Calls helper function on root, passing in the mapping dictionary
    collapse_edges(T, root, node_to_characters)


def to_newick(tree: nx.DiGraph) -> str:
    """Converts a networkx graph to a newick string.
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        _name = node

        return (
            "%s" % (_name,)
            if is_leaf
            else (
                "("
                + ",".join(_to_newick_str(g, child) for child in g.successors(node))
                + ")"
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"

def post_process_tree(T, cm):
    # raise NotImplementedError()
    pass
