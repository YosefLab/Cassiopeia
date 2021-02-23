"""
General utilities for the datasets encountered in Cassiopeia.
"""
from typing import Callable, Dict, List, Optional, Tuple

import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd
import re

from cassiopeia.preprocess import utilities as preprocessing_utilities


def get_lca_characters(
    vecs: List[List[int]], missing_state_indicator: int
) -> List[int]:
    """Builds the character vector of the LCA of a list of character vectors,
    obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_state_indicator: The character representing missing values

    Returns:
        A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [0] * len(vecs[0])
    for i in range(k):
        chars = set([vec[i] for vec in vecs])
        if len(chars) == 1:
            lca_vec[i] = list(chars)[0]
        else:
            if missing_state_indicator in chars:
                chars.remove(missing_state_indicator)
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
    return lca_vec


def newick_to_networkx(newick_string: str) -> nx.DiGraph:
    """Converts a newick string to a networkx DiGraph.

    Args:
        newick_string: A newick string.

    Returns:
        A networkx DiGraph.
    """

    tree = ete3.Tree(newick_string, 1)
    return ete3_to_networkx(tree)


def ete3_to_networkx(tree: ete3.Tree) -> nx.DiGraph:
    """Converts an ete3 Tree to a networkx DiGraph.

    Args:
        tree: an ete3 Tree object

    Returns:
        a networkx DiGraph
    """

    g = nx.DiGraph()
    internal_node_iter = 0
    for n in tree.traverse():

        if n.is_root():
            if n.name == "":
                n.name = f"node{internal_node_iter}"
                internal_node_iter += 1
            continue

        if n.name == "":
            n.name = f"node{internal_node_iter}"
            internal_node_iter += 1

        g.add_edge(n.up.name, n.name)

    return g


def to_newick(tree: nx.DiGraph) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        _name = str(node)
        return (
            "%s" % (_name,)
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


def compute_dissimilarity_map(
    cm: np.array,
    C: int,
    dissimilarity_function: Callable,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
    missing_state_indicator: int = -1,
) -> np.array:
    """Compute the dissimilarity between all samples

    An optimized function for computing pairwise dissimilarities between
    samples in a character matrix according to the dissimilarity function.

    Args:
        cm: Character matrix
        C: Number of samples
        weights: Weights to use for comparing states.
        missing_state_indicator: State indicating missing data

    Returns:
        A dissimilarity mapping as a flattened array.
    """

    nb_dissimilarity = numba.jit(dissimilarity_function, nopython=True)

    nb_weights = numba.typed.Dict.empty(
        numba.types.int64,
        numba.types.DictType(numba.types.int64, numba.types.float64),
    )
    if weights:

        for k, v in weights.items():
            nb_char_weights = numba.typed.Dict.empty(
                numba.types.int64, numba.types.float64
            )
            for state, prior in v.items():
                nb_char_weights[state] = prior
            nb_weights[k] = nb_char_weights

    @numba.jit(nopython=True)
    def _compute_dissimilarity_map(cm, C, missing_state_indicator, nb_weights):

        dm = np.zeros(C * (C - 1) // 2, dtype=numba.float64)
        k = 0
        for i in range(C - 1):
            for j in range(i + 1, C):

                s1 = cm[i, :]
                s2 = cm[j, :]
                dm[k] = nb_dissimilarity(
                    s1, s2, missing_state_indicator, nb_weights
                )
                k += 1

        return dm

    return _compute_dissimilarity_map(
        cm, C, missing_state_indicator, nb_weights
    )


def sample_bootstrap_character_matrices(
    character_matrix: pd.DataFrame,
    prior_probabilities: Optional[Dict[int, Dict[int, float]]] = None,
    B: int = 10,
    random_state: Optional[np.random.RandomState] = None,
) -> List[Tuple[pd.DataFrame, Dict[int, Dict[int, float]]]]:
    """Generates bootstrapped character matrices from a character matrix.

    Ingests a character matrix and randomly creates bootstrap samples by
    sampling with replacement characters. Each bootstrapped character matrix,
    then, retains the same number of characters but some will be repeated and
    some will be ignored. If a prior proability dictionary is also passed in,
    then a new priors dictionary will be created for each bootstrapped character
    matrix.

    Args:
        character_matrix: Character matrix
        prior_probabilities: Probabilities of each (character, state) pair.
        B: Number of bootstrap samples to create.
        random_state: A numpy random state to draw samples from

    Returns:
        A list of bootstrap samples in the form
            (bootstrap_character_matrix, bootstrap_priors).
    """

    bootstrap_samples = []
    M = character_matrix.shape[1]
    for _ in range(B):

        if random_state:
            sampled_cut_sites = random_state.choice(M, M, replace=True)
        else:
            sampled_cut_sites = np.random.choice(M, M, replace=True)

        bootstrapped_character_matrix = character_matrix.iloc[
            :, sampled_cut_sites
        ]
        bootstrapped_character_matrix.columns = [
            f"random_character{i}" for i in range(M)
        ]

        new_priors = None
        if prior_probabilities:
            new_priors = {}
            for i, cut_site in zip(range(M), sampled_cut_sites):
                new_priors[i] = prior_probabilities[cut_site]

        bootstrap_samples.append((bootstrapped_character_matrix, new_priors))

    return bootstrap_samples


def sample_bootstrap_allele_tables(
    allele_table: pd.DataFrame,
    indel_priors: Optional[pd.DataFrame] = None,
    B: int = 10,
    random_state: Optional[np.random.RandomState] = None,
):
    """Generates bootstrap character matrices from an allele table.

    This function will take in an allele table, generated with the Cassiopeia
    preprocess pipeline and produce several bootstrap character matrices with
    respect to intBCs rather than individual cut-sites as in
    `sample_bootstrap_character_matrices`. This is useful because oftentimes
    there are dependencies between cut-sites on the same intBC TargetSite.

    Args:
        allele_table: AlleleTable from the Cassiopeia preprocessing pipeline
        indel_priors: A dataframe mapping indel identities to prior
            probabilities
        B: number of bootstrap samples to create
        random_state: A numpy random state for reproducibility.
    """

    cut_sites = [
        column
        for column in allele_table.columns
        if bool(re.search(r"r\d", column))
    ]
    lineage_profile = preprocessing_utilities.convert_alleletable_to_lineage_profile(
        allele_table
    )

    intbcs = allele_table["intBC"].unique()
    M = len(intbcs)

    bootstrap_samples = []

    for _ in range(B):

        if random_state:
            sampled_intbcs = random_state.choice(intbcs, M, replace=True)
        else:
            sampled_intbcs = np.random.choice(intbcs, M, replace=True)

        bootstrap_intbcs = sum(
            [
                [intbc + f"_{cut_site}" for cut_site in cut_sites]
                for intbc in sampled_intbcs
            ],
            [],
        )
        b_sample = lineage_profile[bootstrap_intbcs]

        (
            bootstrapped_character_matrix,
            priors,
            state_to_indel,
        ) = preprocessing_utilities.convert_lineage_profile_to_character_matrix(
            b_sample, indel_priors=indel_priors
        )

        bootstrap_samples.append(
            (
                bootstrapped_character_matrix,
                priors,
                state_to_indel,
                bootstrap_intbcs,
            )
        )

    return bootstrap_samples
