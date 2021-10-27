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

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import CassiopeiaTreeWarning, is_ambiguous_state
from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.preprocess import utilities as preprocessing_utilities


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
        if n.name == "":
            n.name = f"cassiopeia_internal_node{internal_node_iter}"
            internal_node_iter += 1

        if n.is_root():
            continue

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

    # Numbaize _compute_dissimilarity_map in nopython mode only if the
    # dissimilarity function has been successfully numbaized. Otherwise,
    # numbaize in object mode.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=numba.NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=numba.NumbaWarning)
        _compute_dissimilarity_map = numba.jit(
            _compute_dissimilarity_map, nopython=numbaize
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

    lineage_profile = preprocessing_utilities.convert_alleletable_to_lineage_profile(
        allele_table, cut_sites
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


def compute_phylogenetic_weight_matrix(
    tree: CassiopeiaTree,
    inverse: bool = False,
    inverse_fn: Callable[[Union[int, float]], float] = lambda x: 1 / x,
) -> pd.DataFrame:
    """Computes the phylogenetic weight matrix.

    Computes the distances between all leaves in a tree. The user has the option
    to return the inverse matrix, (i.e., transform distances to proximities) and
    specify an appropriate inverse function.

    This function computes the phylogenetic weight matrix in O(n^2 logn) time.

    An NxN weight matrix is returned.

    Args:
        tree: CassiopeiaTree
        inverse: Convert distances to proximities
        inverse_fn: Inverse function (default = 1 / x)

    Returns:
        An NxN phylogenetic weight matrix
    """
    N = tree.n_cell
    W = pd.DataFrame(np.zeros((N, N)), index=tree.leaves, columns=tree.leaves)

    for leaf1 in tree.leaves:

        distances = tree.get_distances(leaf1, leaves_only=True)
        for leaf2, _d in distances.items():

            if inverse:
                _d = inverse_fn(_d) if _d > 0 else np.inf

            W.loc[leaf1, leaf2] = W.loc[leaf2, leaf1] = _d

    np.fill_diagonal(W.values, 0)

    return W
    
@numba.jit(nopython=True)
def net_relatedness_index(
    dissimilarity_map: np.array, indices_1: np.array, indices_2: np.array
) -> float:
    """Computes the net relatedness index between indices.

    Using the dissimilarity map specified and the indices of samples, compute
    the net relatedness index, defined as:

    sum(distances over i,j in indices_1,indices_2) / (|indices_1| x |indices_2|)

    Args:
        dissimilarity_map: Dissimilarity map between all samples.
        indices_1: Indices corresponding to the first group.
        indices_2: Indices corresponding to the second group.

    Returns:
        The Net Relatedness Index (NRI)
    """

    nri = 0
    for i in indices_1:
        for j in indices_2:
            nri += dissimilarity_map[i, j]

    return nri / (len(indices_1) * len(indices_2))

def compute_inter_cluster_distances(
    tree: CassiopeiaTree,
    meta_item: Optional[str] = None,
    meta_data: Optional[pd.DataFrame] = None,
    dissimilarity_map: Optional[pd.DataFrame] = None,
    distance_function: Callable = net_relatedness_index,
    **kwargs,
) -> pd.DataFrame:
    """Computes mean distance between clusters.

    Compute the mean distance between categories in a categorical variable. By
    default, the phylogenetic weight matrix will be computed and used for this
    distance calculation, but a user can optionally provide a dissimilarity
    map instead.

    This function performs the computation in O(K^2)*O(distance_function) time
    for a variable with K categories.

    Args:
        tree: CassiopeiaTree
        meta_item: Column in the cell meta data of the tree. If `meta_data` is
            specified, this is ignored.
        meta_data: Meta data to use for this calculation. This argument takes
            priority over meta_item.
        dissimilarity_map: Dissimilarity map to use for distances. If this is
            specified, the phylogenetic weight matrix is not computed.
        number_of_neighbors: Number of nearest neighbors to use for computing
            the mean distances. If this is not specified, then all cells are
            used.
        **kwargs: Arguments to pass to the distance function.

    Returns:
        A K x K distance matrix.
    """
    meta_data = tree.cell_meta[meta_item] if (meta_data is None) else meta_data

    # ensure that the meta data is categorical
    if not pd.api.types.is_string_dtype(meta_data):
        raise CassiopeiaError("Meta data must be categorical or a string.")

    D = (
        compute_phylogenetic_weight_matrix(tree)
        if (dissimilarity_map is None)
        else dissimilarity_map
    )

    unique_states = meta_data.unique()
    K = len(unique_states)
    inter_cluster_distances = pd.DataFrame(
        np.zeros((K, K)), index=unique_states, columns=unique_states
    )

    # align distance matrix and meta_data
    D = D.loc[meta_data.index.values, meta_data.index.values]

    for state1 in unique_states:
        indices_1 = np.where(np.array(meta_data) == state1)[0]
        for state2 in unique_states:
            indices_2 = np.where(np.array(meta_data) == state2)[0]

            distance = distance_function(
                D.values, indices_1, indices_2, **kwargs
            )
            inter_cluster_distances.loc[state1, state2] = distance

    return inter_cluster_distances