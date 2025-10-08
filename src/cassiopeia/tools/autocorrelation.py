"""Utility file for computing autocorrelation statistics on trees."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree, utilities
from cassiopeia.mixins import AutocorrelationError


def compute_morans_i(
    tree: CassiopeiaTree,
    meta_columns: list | None = None,
    X: pd.DataFrame | None = None,
    W: pd.DataFrame | None = None,
    inverse_weight_fn: Callable[[int | float], float] = lambda x: 1.0 / x,
) -> float | pd.DataFrame:
    """Computes Moran's I statistic.

    Using the cross-correlation between leaves as specified on the tree, compute
    the Moran's I statistic for each of the data items specified. This will
    only work for numerical data, and will thrown an error otherwise.

    Generally, this statistic takes in a weight matrix (which can be computed
    directly from a phylogenetic tree) and a set of numerical observations that
    are centered and standardized (i.e., mean 0 and population standard deviation
    of 1). Then, the Moran's I statistic is:

    I = X' * Wn * X

    where X' denotes a tranpose, * denotes the matrix multiplier, and Wn is the
    normalized weight matrix such that sum([w_i,j for all i,j]) = 1.

    Inspired from the tools and code used in Chaligne et al, Nature Genetics
    2021.

    The mathematical details of the statistic can be found in:
        Wartenberg, "Multivariate Spatial Correlation: A Method for Exploratory
        Geographical Analysis", Geographical Analysis (1985)

    Args:
        tree: CassiopeiaTree
        meta_columns: Columns in the Cassiopeia Tree :attr:cell_meta object
            for which to compute autocorrelations
        X: Extra data matrix for computing autocorrelations.
        W: Phylogenetic weight matrix. If this is not specified, then the
            weight matrix will be computed within the function.
        inverse_weight_fn: Inverse function to apply to the weights, if the
            weight matrix must be computed.

    Returns
    -------
        Moran's I statistic
    """
    if X is None and meta_columns is None:
        raise AutocorrelationError("Specify data for computing autocorrelations.")

    _X = None
    if meta_columns is not None:
        _X = tree.cell_meta[meta_columns]

    if X is not None:
        if len(np.intersect1d(tree.leaves, X.index)) != tree.n_cell:
            raise AutocorrelationError(
                "Specified argument X must be a dataframe with identical indices to the leaves of the CassiopeiaTree."
            )

        _X = pd.concat([_X, X], axis=0)

    # check to make sure all values are numerical
    if not np.all(_X.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all())):
        raise AutocorrelationError("There are some columns that are not numeric in the specified data.")

    # cast to numeric
    _X = _X.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # instantiate the weight matrix if None is specified
    if W is None:
        W = utilities.compute_phylogenetic_weight_matrix(tree, inverse=True, inverse_fn=inverse_weight_fn)

    # make sure that W has the correct indices
    if len(np.intersect1d(tree.leaves, W.index)) != tree.n_cell:
        raise AutocorrelationError("Weight matrix does not have the same leaves as the tree.")

    # normalize W to 1
    _W = W / W.sum().sum()

    # center and standardize _X
    _X = (_X - _X.mean()) / _X.std(axis=0, ddof=0)

    I = _X.T.dot(_W).dot(_X)

    # if we're only testing one variable, return a float
    if _X.shape[1] == 1:
        I = I.iloc[0, 0]

    return I
