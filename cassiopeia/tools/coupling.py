"""
File storing functionality for computing coupling statistics between meta
variables on a tree.
"""
from typing import Optional

from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities


def compute_evolutionary_coupling(
    tree: CassiopeiaTree,
    meta_variable: str,
    minimum_proportion: float = 0.05,
    number_of_shuffles: int = 500,
    random_state: Optional[np.random.RandomState] = None,
    dissimilarity_map: Optional[pd.DataFrame] = None,
    cluster_comparison_function: Optional[
        Callable
    ] = data_utilitiesnet_relatedness_index,
    **comparison_kwargs,
) -> pd.DataFrame:
    """Computes Evolutionary Coupling of categorical variables.

    Using the methodology described in Yang, Jones et al, BioRxiv (2021), this
    function will compute the "evolutionary coupling" statistic between values
    that a categorical variable can take on with the tree. For example, this
    categorical variable can be a "cell type", and this function will compute
    the evolutionary couplings between all types of cell types. This indicates
    how closely related these cell types are to one another.

    Briefly, this statistic is the Z-normalized mean distance between categories
    in the specified categorical variable.

    The computational complexity of this function is O(n^2 log n + Bk^2) for a
    tree with n leaves, a variable with k categories, and B random shuffles.

    Args:
        tree: CassiopeiaTree
        meta_variable: Column in `tree.cell_meta` that stores a categorical
            variable with K categories.
        minimum_proportion: Minimum proportion of cells that a category needs
            to appear in to be considered.
        number_of_shuffles: Number of times to shuffle the data to compute the
            empirical Z score.
        random_state: Numpy random state to parameterize the shuffling.
        dissimilarity_map: A precomputed dissimilarity map between all leaves.
        cluster_comparison_function: A function for comparing the mean distance
            between groups. By default, this is the Net Relatedness Index.
        **comparison_kwargs: Extra arguments to pass to the cluster comparison
            function.

    Returns:
        A K x K evolutionary coupling dataframe. 
    """

    W = (
        data_utilities.compute_phylogenetic_weight_matrix(tree)
        if (dissimilarity_map is None)
        else dissimilarity_map
    )

    meta_data = tree.cell_meta[meta_variable]

    # subset meta data by minimum proportion
    if minimum_proportion > 0:
        filter_threshold = int(len(tree.leaves) * minimum_proportion)
        category_frequencies = meta_data.value_counts()
        passing_categories = category_frequencies[
            category_frequencies > filter_threshold
        ].index.values
        meta_data = meta_data[meta_data[meta_variable].isin(passing_categories)]
        W = W.loc[meta_data.index.values, meta_data.index.values]

    # compute inter-cluster distances
    inter_cluster_distances = data_utilities.compute_inter_cluster_distances(
        tree,
        meta_data=meta_data,
        dissimilarity_map=W,
        distance_function=cluster_comparison_function,
        **comparison_kwargs,
    )

    # compute background for Z-scoring
    background = defaultdict(list)
    for _ in tqdm(number_of_shuffles, desc="Creating empirical background"):

        permuted_assignments = meta_data.copy()
        if random_state:
            permuted_assignments.index = random_state.permutation(
                meta_data.index.values
            )
        else:
            permuted_assignments.index = np.random.permutation(
                meta_data.index.values
            )

        background_distances = data_utilities.compute_inter_cluster_distances(
            tree,
            permuted_assignments,
            dissimilarity_map=W,
            distance_function=cluster_comparison_function,
            **comparison_kwargs,
        )

        for s1 in background_distances.index:
            for s2 in background_distances.columns:
                background[(s1, s2)].append(background_distances.loc[s1, s2])

    Z_scores = inter_cluster_distances.copy()
    for s1 in Z_scores.index:
        for s2 in Z_scores.columns:
            mean = np.mean(background[(s1, s2)])
            sd = np.std(background[(s1, s2)])

            Z_scores.loc[s1, s2] = (
                inter_cluster_distances.loc[s1, s2] - mean
            ) / sd

    Z_scores.fillna(0, inplace=True)

    return Z_scores
