import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# def map_states(cm, unedit_char, missing_char) -> Tuple[pd.DataFrame, Dict[int, str]]:
#     """Maps the characters in a character matrix to the effective states.
#     """
#     state_map = {}
#     index = 1
#     unique_states = sorted(pd.unique(cm.values.ravel()))
#     if missing_char in unique_states:
#         unique_states.remove(missing_char)
#         state_map[-1] = missing_char
#         cm = cm.replace(missing_char, -1)
#     unique_states.remove(unedit_char)
#     state_map[0] = unedit_char
#     cm = cm.replace(unedit_char, 0)
#     for i in unique_states:
#         state_map[index] = i
#         cm = cm.replace(i, index)
#         index += 1

#     return cm, state_map

def get_LCA_vec(vec1, vec2):
    """Builds a consensus vector from two, obeying Camin-Sokal Parsimony.
    """
    assert(len(vec1) == len(vec2))
    consensus = [0] * len(vec1)
    for i in range(len(vec1)):
        if vec1[i] == -1 and vec2[i] != -1:
            consensus[i] = vec2[i]
        if vec2[i] == -1 and vec1[i] != -1:
            consensus[i] = vec1[i]
        if vec1[i] == vec2[i] and vec1[i] != 0:
            consensus[i] = vec1[i]
    return consensus

def infer_ancestral_characters(network, node, char_map):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.
    """
    if network.out_degree(node) == 0:
        return
    vecs = []
    for i in network.successors(node):
        infer_ancestral_characters(network, i, char_map)
        vecs.append(char_map[i])
    assert(len(vecs) == 2)
    LCA_vec = get_LCA_vec(vecs[0], vecs[1])
    char_map[node] = LCA_vec
    return

def collapse_edges(network, node, char_map):
    """A helper function to help collapse edges in a tree.
    """
    if network.out_degree(node) == 0:
        return
    to_remove = []
    to_collapse = []
    for i in network.successors(node):
        to_collapse.append(i)
    for i in to_collapse:
        collapse_edges(network, i, char_map)
        if char_map[i] == char_map[node]:
            for j in network.successors(i):
                network.add_edge(node, j)
            to_remove.append(i)
    for i in to_remove:
        network.remove_node(i)
    return

def collapse_tree(T, cm, infer_ancestral_characters):
    """Collapses non-informative edges (edges with 0 mutations) in a tree.
    """
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    root = [n for n in T if T.in_degree(n) == 0][0]
    char_map = {}
    
    if infer_ancestral_characters:
        for i in leaves:
            char_map[i] = list(cm.iloc[i,:])
        infer_ancestral_characters(T, root, char_map)
    else:
        for i in T.nodes():
            char_map[i] = i.char_vec
    collapse_edges(T, root, char_map)

def post_process_tree(T, cm):
    raise NotImplementedError()

def compute_mutation_frequencies(cm, samples: List[int] = None) -> Dict[int, Dict[int, int]]:
    freq_dict = {}
    subset_cm = cm.iloc[samples, :]
    for char in range(subset_cm.shape[1]):
        char_dict = {}
        state_counts = np.unique(subset_cm.iloc[:,char], return_counts = True)
        for i in range(len(state_counts[0])):
            state = state_counts[0][i]
            count = state_counts[1][i]
            char_dict[state] = count
        freq_dict[char] = char_dict
    return freq_dict

# def compute_mutation_frequencies(self, samples: List[int] = None) -> pd.DataFrame:
#     """Computes the frequency of character/state pairs in the samples.

#     Args:
#         samples: A list of samples

#     Returns:
#         A dataframe mapping character/state pairs to frequencies
#     """
#     cm = self.character_matrix
#     k = cm.shape[1]
#     m = max(cm.max()) + 1
#     F = np.zeros((k,m), dtype=int)
#     if not samples:
#         samples = list(range(cm.shape[0]))
#     for i in samples:
#         for j in range(k):
#             F[j][cm.iloc(i, j)] += 1
#     return F

