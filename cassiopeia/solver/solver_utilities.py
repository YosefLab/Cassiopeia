import numpy as np
import pandas as pd
import networkx as nx

from typing import Dict, List, Optional, Tuple


def get_lca_characters(vec1: List[str], vec2: List[str], missing_char: str) -> List[str]:
    """Builds the character vector of the LCA of two character vectors, obeying
    Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing 
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if both have a missing value at that index.
    
    Args:
        vec1: The first character vector
        vec2: The second character vector
        missing_char: The character representing missing values

    Returns:
        A list representing the character vector of the lca

    """
    assert len(vec1) == len(vec2)
    lca_vec = [0] * len(vec1)
    for i in range(len(vec1)):
        if vec1[i] == missing_char and vec2[i] != missing_char:
            lca_vec[i] = vec2[i]
        if vec2[i] == missing_char and vec1[i] != missing_char:
            lca_vec[i] = vec1[i]
        if vec1[i] == vec2[i] and vec1[i] != "0":
            lca_vec[i] = vec1[i]
    return lca_vec


def infer_ancestral_characters(T: nx.DiGraph, node: int, char_map: Dict[int, List[str]], missing_char: str):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.

    For an internal node, annotates that node's character vector to be the LCA
    of its daughters character vectors. Annotates from the samples.

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
        infer_ancestral_characters(T, i, char_map, missing_char)
        vecs.append(char_map[i])
    assert len(vecs) == 2
    lca_characters = get_lca_characters(vecs[0], vecs[1], missing_char)
    char_map[node] = lca_characters
    return


def collapse_edges(T: nx.DiGraph, node: int, char_map: Dict[int, List[str]]):
    """A helper function to help collapse mutationless edges in a tree.

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
        collapse_edges(T, i, char_map)
        if char_map[i] == char_map[node]:
            for j in T.successors(i):
                T.add_edge(node, j)
            to_remove.append(i)
    for i in to_remove:
        T.remove_node(i)
    return


def collapse_tree(T: nx.DiGraph, infer_ancestral_characters: bool, cm: pd.DataFrame = None, missing_char: str = None):
    """Collapses mutationless edges in a tree. 
    
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
        for i in leaves:
            char_map[i] = list(cm.iloc[i, :])
        infer_ancestral_characters(T, root, char_map, missing_char)
    else:
        for i in T.nodes():
            char_map[i] = i.char_vec

    # Calls helper function on root, passing in the mapping dictionary
    collapse_edges(T, root, char_map)
    

def post_process_tree(T, cm):
    raise NotImplementedError()


def compute_mutation_frequencies(
    cm: pd.DataFrame, missing_char: str, samples: List[int] = None
) -> Dict[int, Dict[int, int]]:
    """Computes the number of samples in a character matrix that have each
    character/state mutation.

    Generates a dictionary that maps each character to a dictionary of state/
    sample frequency pairs, allowing quick lookup. Subsets the character matrix
    to only include the samples in the sample set.

    Args:
        cm: The character matrix from which to calculate frequencies
        missing_char: The character representing missing values
        samples: The set of relevant samples in calculating frequencies

    Returns:
        A dictionary containing frequency information for each character/state
        pair

    """
    if samples:
        cm = cm.iloc[samples, :]
    freq_dict = {}
    for char in range(cm.shape[1]):
        char_dict = {}
        state_counts = np.unique(cm.iloc[:, char], return_counts=True)
        for i in range(len(state_counts[0])):
            state = state_counts[0][i]
            count = state_counts[1][i]
            char_dict[state] = count
        if missing_char not in char_dict:
            char_dict[missing_char] = 0
        freq_dict[char] = char_dict
    return freq_dict
