"""This file contains general utilities to be called by functions throughout 
the solver module"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import ete3
import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import utilities as data_utilities


class InferAncestorError(Exception):
    """An Exception class for collapsing edges, indicating a necessary argument
    was not included.
    """

    pass


class PriorTransformationError(Exception):
    """An Exception class for generating weights from priors."""

    pass


def annotate_ancestral_characters(
    T: nx.DiGraph,
    node: int,
    node_to_characters: Dict[int, List[int]],
    missing_state_indicator: int,
):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.

    For an internal node, annotates that node's character vector to be the LCA
    of its daughter character vectors. Annotates from the samples upwards.

    Args:
        T: A networkx DiGraph object representing the tree
        node: The node whose state is to be inferred
        node_to_characters: A dictionary that maps nodes to their character vectors
        missing_state_indicator: The character representing missing values

    Returns:
        None, annotates node_to_characters dictionary with node/character vector pairs


    """
    if T.out_degree(node) == 0:
        return
    vectors = []
    for i in T.successors(node):
        annotate_ancestral_characters(
            T, i, node_to_characters, missing_state_indicator
        )
        vectors.append(node_to_characters[i])
    lca_characters = data_utilities.get_lca_characters(
        vectors, missing_state_indicator
    )
    node_to_characters[node] = lca_characters
    T.nodes[node]["characters"] = lca_characters


def collapse_edges(
    T: nx.DiGraph, node: int, node_to_characters: Dict[int, List[int]]
):
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
    tree: nx.DiGraph,
    infer_ancestral_characters: bool,
    character_matrix: Optional[pd.DataFrame] = None,
    missing_state_indicator: Optional[int] = None,
):
    """Collapses mutationless edges in a tree in-place.

    Uses the internal node annotations of a tree to collapse edges with no
    mutations. Either takes in a tree with internal node annotations or
    a tree without annotations and infers the annotations bottom-up from the
    samples obeying Camin-Sokal Parsimony. If ground truth internal annotations
    exist, it is suggested that they are used directly and that the annotations
    are not inferred again using the parsimony method.

    Args:
        tree: A networkx DiGraph object representing the tree
        infer_ancestral_characters: Infer the ancestral characters states of
            the tree
        character_matrix: A character matrix storing character states for each
            leaf
        missing_state_indicator: Character state indicating missing data

    Returns:
        A collapsed tree

    """
    leaves = [
        n for n in tree if tree.out_degree(n) == 0 and tree.in_degree(n) == 1
    ]
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    node_to_characters = {}

    # Populates the internal annotations using either the ground truth
    # annotations, or infers them
    if infer_ancestral_characters:
        if character_matrix is None or missing_state_indicator is None:
            raise InferAncestorError(
                "In order to infer ancestral characters, a character matrix and missing character are needed"
            )

        name_to_index = dict(
            zip(character_matrix.index, range(character_matrix.shape[0]))
        )
        character_matrix_np = character_matrix.to_numpy()

        for i in leaves:
            node_to_characters[i] = tree.nodes[i]["characters"] = list(
                character_matrix_np[name_to_index[i], :]
            )

        annotate_ancestral_characters(
            tree, root, node_to_characters, missing_state_indicator
        )

    else:
        for i in tree.nodes():
            node_to_characters[i] = tree.nodes[i]["characters"]

    # Calls helper function on root, passing in the mapping dictionary
    collapse_edges(tree, root, node_to_characters)
    return tree


def collapse_unifurcations(tree: ete3.Tree) -> ete3.Tree:
    """Collapse unifurcations.
    Collapse all unifurcations in the tree, namely any node with only one child
    should be removed and all children should be connected to the parent node.
    Args:
        tree: tree to be collapsed
    Returns:
        A collapsed tree.
    """

    collapse_fn = lambda x: (len(x.children) == 1)

    collapsed_tree = tree.copy()
    to_collapse = [n for n in collapsed_tree.traverse() if collapse_fn(n)]

    for n in to_collapse:
        n.delete()

    return collapsed_tree


def transform_priors(
    priors: Optional[Dict[int, Dict[int, float]]],
    prior_transformation: str = "negative_log",
) -> Dict[int, Dict[int, float]]:
    """Generates a dictionary of weights from priors.

    Generates a dictionary of weights from given priors for each character/state
    pair for use in algorithms that inherit the GreedySolver. Supported
    transformations include negative log, negative log square root, and inverse.

    Args:
        priors: A dictionary of prior probabilities for each character/state
            pair
        prior_transformation: A function defining a transformation on the priors
            in forming weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative log
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Returns:
        A dictionary of weights for each character/state pair
    """
    if prior_transformation not in [
        "negative_log",
        "inverse",
        "square_root_inverse",
    ]:
        raise PriorTransformationError(
            "Please select one of the supported prior transformations."
        )

    prior_function = lambda x: -np.log(x)

    if prior_transformation == "square_root_inverse":
        prior_function = lambda x: (np.sqrt(1 / x))
    if prior_transformation == "inverse":
        prior_function = lambda x: 1 / x

    weights = {}
    for character in priors:
        state_weights = {}
        for state in priors[character]:
            p = priors[character][state]
            if p <= 0.0 or p > 1.0:
                raise PriorTransformationError(
                    "Please make sure all priors have a positive value less than 1 and greater than 0"
                )
            state_weights[state] = prior_function(p)
        weights[character] = state_weights
    return weights


def convert_sample_names_to_indices(
    names: List[str], samples: List[str]
) -> List[int]:
    """Maps samples to their integer indices in a given set of names.

    Used to map sample string names to the their integer positions in the index
    of the original character matrix for efficient indexing operations.

    Args:
        names: A list of sample names, represented by their string names in the
            original character matrix
        samples: A list of sample names representing the subset to be mapped to
            integer indices

    Returns:
        A list of samples mapped to integer indices
    """
    name_to_index = dict(zip(names, range(len(names))))

    return list(map(lambda x: name_to_index[x], samples))
