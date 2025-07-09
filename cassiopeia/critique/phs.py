"""
Pairwise Homoplasy Score (PHS).
"""
import warnings
from itertools import combinations

import networkx as nx
import treedata as td
import numpy as np
from scipy.stats import binom

from cassiopeia.data import CassiopeiaTree


def _cPHS(tree,lam, q, depth_key = "time",characters_key = "characters",
          missing_state = -1,unedited_state = 0):
    """Calculate the pairwise homoplasy score (PHS) for a tree."""
    # Setup
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    k = len(tree.nodes[leaves[0]][characters_key])
    # check that all leaves have the same depth
    leaf_depths = [tree.nodes[leaf][depth_key] for leaf in leaves]
    if not all(depth == leaf_depths[0] for depth in leaf_depths):
        raise ValueError("All leaves should have a depth to calculate PHS."
                        " Did you forget to perform branch length estimation?")
    # Calculate PHS score and LCA height for all pairs
    phs = []
    lca_heights = []
    for (l1,l2), lca in nx.all_pairs_lowest_common_ancestor(tree, combinations(leaves, 2)):
        phs.append(sum(1
            for i in range(k)
            if (
                tree.nodes[lca][characters_key][i] == unedited_state
                and tree.nodes[l1][characters_key][i] not in (missing_state, unedited_state)
                and tree.nodes[l1][characters_key][i] == tree.nodes[l2][characters_key][i]
            )
        ))
        lca_heights.append(tree.nodes[lca][depth_key])
    phs = np.array(phs)
    lca_heights = np.array(lca_heights)
    # Calculate p-values
    alpha = np.exp(-lam * lca_heights)
    beta =  (1 - np.exp(-lam * (1 - lca_heights)))
    prob = alpha * beta**2 * q
    prob[lca_heights == 1] = 1
    pvalues = 1 - binom.cdf(phs - 1, k, prob)
    pvalues[pvalues == 0] = np.finfo(float).eps # zeros cannot be real zeros
    # Adjust p-values for multiple testing
    pvalues_sorted = np.sort(pvalues)
    adjusted_pvalues = pvalues_sorted * len(pvalues) / np.arange(1, len(pvalues)+1)
    return min(adjusted_pvalues)


def cPHS(
    tdata: td.TreeData | CassiopeiaTree, 
    priors: dict | dict[int, dict] | None = None, 
    depth_key: str = "time", 
    characters_key: str = "characters", 
    missing_state: int = -1, 
    unedited_state: int = 0,
    mutation_rate: float | None = None,
    collision_probability: float | None = None,
    tree: str = "tree"
) -> float:
    """Calculate the Pairwise Homoplasy Score (PHS) for a tree.

    Given a tree with inferred branch lengths and ancestral character states,
    this function calculates the cPHS statistic as described in Zilber et al. (2026).
    The cPHS statistic uses a homoplasy-based approach to assess the accuracy of tree
    reconstructions by quantifying the likelihood of observed homoplasies under a 
    specified mutation model.

    Args:
        tdata: Input tree, either a TreeData or CassiopeiaTree object.
        priors: Prior probabilities for character states. Can be a dict or a dict of dicts.
        depth_key: Node attribute in the tree where depth is stored.
        characters_key: Node attribute in the tree where character states are stored.
        missing_state: Value representing missing data in the character matrix.
        unedited_state: Value representing unedited state in the character matrix.
        mutation_rate: Estimated mutation rate. If None, it will be estimated from the data.
        collision_probability: Estimated collision probability. If None, it will be estimated from priors or data.
        tree: Name of the tree to analyze.

    Returns:
        The cPHS score for the tree.
    """
    # Setup
    if isinstance(tdata, CassiopeiaTree):
        t = tdata.get_tree_topology()
        priors = tdata.priors
        characters = tdata.character_matrix
        characters_key = "character_states"
    elif isinstance(tdata, td.TreeData):
        t = tdata.obst[tree]
        characters = tdata.obsm[characters_key]
    else:
        raise ValueError(f"Unsupported type: {type(tdata)}."
                         "Expected TreeData or CassiopeiaTree.")
    # Estimate mutation rate
    if mutation_rate is None:
        proportion_mutated = (
            np.sum(~np.isin(characters.values, (missing_state, unedited_state))) /
            np.sum(characters.values != missing_state)
        )
        mutation_rate = -np.log(1.0 - proportion_mutated)
    # Get collision probability
    if collision_probability is None:
        if priors is None:
            collision_probability = 1/characters.stack().nunique()
            warnings.warn(
                "Collision probability or priors are not provided, using 1/m as default.",
                stacklevel=2)
        elif isinstance(list(priors.values())[0], dict): 
            collision_probability =  np.sum(np.array(tree.priors[0].values())**2)
        elif isinstance(priors, dict):
            collision_probability = np.sum(np.array(list(priors.values()))**2)
        else:
            raise ValueError("Unsupported type of priors. Expected dict or dict[int, dict].")
    # Calculate cPHS
    return _cPHS(t, mutation_rate, collision_probability, depth_key, 
                 characters_key, missing_state, unedited_state)
