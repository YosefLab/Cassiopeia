"""
General utilities for the datasets encountered in Cassiopeia.
"""
import collections
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd
import re

from cassiopeia.preprocess import utilities as preprocessing_utilities

# Can't import from CassiopeiaTree.py due to circular imports
class CassiopeiaTreeWarning(UserWarning):
    """A Warning for the CassiopeiaTree class."""

    pass


def get_lca_characters(
    vecs: List[Union[List[int], List[Tuple[int, ...]]]],
    missing_state_indicator: int,
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
        chars = set()
        for vec in vecs:
            if is_ambiguous_state(vec[i]):
                chars = chars.union(vec[i])
            else:
                chars.add(vec[i])
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


def to_newick(tree: nx.DiGraph, record_branch_lengths: bool = False) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree
        record_branch_lengths: Whether to record branch lengths on the tree in
            the newick string

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        weight_string = ""

        if record_branch_lengths and g.in_degree(node) > 0:
            parent = list(g.predecessors(node))[0]
            weight_string = ":" + str(g[parent][node]["length"])

        _name = str(node)
        return (
            "%s" % (_name,) + weight_string
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
                + weight_string
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
    # Try to numbaize the dissimilarity function, but fallback to python
    numbaize = True
    try:
        dissimilarity_func = numba.jit(dissimilarity_function, nopython=True)
    # When cluster_dissimilarity is used, the dissimilarity_function is wrapped
    # in a partial, which raises a TypeError when trying to numbaize.
    except TypeError:
        warnings.warn(
            "Failed to numbaize dissimilarity function. "
            "Falling back to Python.",
            CassiopeiaTreeWarning,
        )
        numbaize = False
        dissimilarity_func = dissimilarity_function

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

    def _compute_dissimilarity_map(cm, C, missing_state_indicator, nb_weights):

        dm = np.zeros(C * (C - 1) // 2, dtype=np.float64)
        k = 0
        for i in range(C - 1):
            for j in range(i + 1, C):

                s1 = cm[i, :]
                s2 = cm[j, :]
                dm[k] = dissimilarity_func(
                    s1, s2, missing_state_indicator, nb_weights
                )
                k += 1

        return dm

    # Don't numbaize _compute_dissimilarity_map when we failed to numbaize
    # the dissimilarity function. Otherwise, if we try to call a Python
    # function from a numbaized (in object mode) a LOT of warnings are raised.
    if numbaize:
        _compute_dissimilarity_map = numba.jit(
            _compute_dissimilarity_map, nopython=True
        )

    return _compute_dissimilarity_map(
        cm, C, missing_state_indicator, nb_weights
    )


def sample_bootstrap_character_matrices(
    character_matrix: pd.DataFrame,
    prior_probabilities: Optional[Dict[int, Dict[int, float]]] = None,
    num_bootstraps: int = 10,
    random_state: Optional[np.random.RandomState] = None,
) -> List[Tuple[pd.DataFrame, Dict[int, Dict[int, float]]]]:
    """Generates bootstrapped character matrices from a character matrix.

    Ingests a character matrix and randomly creates bootstrap samples by
    sampling characters with replacement. Each bootstrapped character matrix,
    then, retains the same number of characters but some will be repeated and
    some will be ignored. If a prior proability dictionary is also passed in,
    then a new priors dictionary will be created for each bootstrapped character
    matrix.

    Args:
        character_matrix: Character matrix
        prior_probabilities: Probabilities of each (character, state) pair.
        num_bootstraps: Number of bootstrap samples to create.
        random_state: A numpy random state to from which to draw samples

    Returns:
        A list of bootstrap samples in the form
            (bootstrap_character_matrix, bootstrap_priors).
    """

    bootstrap_samples = []
    M = character_matrix.shape[1]
    for _ in range(num_bootstraps):

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

        new_priors = {}
        if prior_probabilities:
            for i, cut_site in zip(range(M), sampled_cut_sites):
                new_priors[i] = prior_probabilities[cut_site]

        bootstrap_samples.append((bootstrapped_character_matrix, new_priors))

    return bootstrap_samples


def sample_bootstrap_allele_tables(
    allele_table: pd.DataFrame,
    indel_priors: Optional[pd.DataFrame] = None,
    num_bootstraps: int = 10,
    random_state: Optional[np.random.RandomState] = None,
    cut_sites: Optional[List[str]] = None,
) -> List[
    Tuple[
        pd.DataFrame,
        Dict[int, Dict[int, float]],
        Dict[int, Dict[int, str]],
        List[str],
    ]
]:
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
        num_bootstraps: number of bootstrap samples to create
        random_state: A numpy random state for reproducibility.
        cut_sites: Columns in the AlleleTable to treat as cut sites. If None,
            we assume that the cut-sites are denoted by columns of the form
            "r{int}" (e.g. "r1")
    Returns:
        A list of bootstrap samples in the form of tuples
            (bootstrapped character matrix, prior dictionary,
            state to indel mapping, bootstrapped intBC set)
    """

    if cut_sites is None:
        cut_sites = preprocessing_utilities.get_default_cut_site_columns(
            allele_table
        )

    lineage_profile = (
        preprocessing_utilities.convert_alleletable_to_lineage_profile(
            allele_table, cut_sites
        )
    )

    intbcs = allele_table["intBC"].unique()
    M = len(intbcs)

    bootstrap_samples = []

    for _ in range(num_bootstraps):

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


def is_ambiguous_state(state: Union[int, Tuple[int, ...]]) -> bool:
    """Determine whether the provided state is ambiguous.

    Note that this function operates on a single (indel) state.

    Args:
        state: Single, possibly ambiguous, character state

    Returns:
        True if the state is ambiguous, False otherwise.
    """
    return isinstance(state, tuple)


def resolve_most_abundant(state: Tuple[int, ...]) -> int:
    """Resolve an ambiguous character by selecting the most abundant.

    This function is designed to be used with
    :func:`CassiopeiaTree.resolve_ambiguous_characters`. It resolves an ambiguous
    character, represented as a tuple of integers, by selecting the most abundant,
    where ties are resolved randomly.

    Args:
        state: Ambiguous state as a tuple of integers

    Returns:
        Selected state as a single integer
    """
    most_common = collections.Counter(state).most_common()
    return np.random.choice(
        [state for state, count in most_common if count == most_common[0][1]]
    )
