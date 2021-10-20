"""
Utility file for computing autocorrelation statistics on trees.
"""
from typing import Callable, List, Optional, Union
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import AutocorrelationError
from cassiopeia.data import utilities


def compute_morans_i(
    tree: CassiopeiaTree,
    meta_columns: Optional[List] = None,
    X: Optional[pd.DataFrame] = None,
    W: Optional[pd.DataFrame] = None,
    inverse_weight_fn: Callable[[Union[int, float]], float] = lambda x: 1.0 / x,
) -> Union[float, pd.DataFrame]:
    """Computes Moran's I statistic.

    Using the cross-correlation between leaves as specified on the tree, compute
    the Moran's I statistic for each fo the data items specified. This will
    only work for numerical data, and will thrown an error otherwise.

    Inspired from the tools and code used in Chaligne et al, Nature Genetics
    2021.

    Args:
        tree: CassiopeiaTree
        meta_columns: Columns in the Cassioepia Tree :attr:cell_meta object
            for which to compute autocorrelations
        X: Extra data matrix for computing autocorrelations.
        W: Phylogenetic weight matrix. If this is not specified, then the
            weight matrix will be computed within the function.
        inverse_weight_fn: Inverse function to apply to the weights, if the
            weight matrix must be computed.

    Returns:
        Moran's I statistic
    """

    if X is None and meta_columns is None:
        raise AutocorrelationError(
            "Specify data for computing" " autocorrelations."
        )

    _X = None
    if meta_columns is not None:
        _X = tree.cell_meta[meta_columns]

    if X is not None:
        if len(np.intersect1d(tree.leaves, X.index)) != tree.n_cell:
            raise AutocorrelationError(
                "Specified argument X must be a"
                " dataframe with identical indices to the leaves of"
                " the CassiopeiaTree"
            )

        _X = pd.concat([_X, X], axis=0)

    # check to make sure all values are numerical
    if not np.all(
        _X.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all())
    ):
        raise AutocorrelationError(
            "There are some columns that are not numeric"
            " in the specified data."
        )

    # instantiate the weight matrix if None is specified
    if W is None:
        W = utilities.compute_phylogenetic_weight_matrix(
            tree, inverse=True, inverse_fn=inverse_weight_fn
        )

    N = tree.n_cell

    # normalize W to 1
    _W = W / W.sum().sum()

    # center
    _X = (_X - _X.mean())

    # compute covariance
    R = _X.T.dot(_X) / N
    
    # compute total variance
    total_variance = np.diag(R).dot(np.diag(R))
    
    I = _X.T.dot(_W).dot(_X) / np.sqrt(total_variance)
     
    # if we're only testing one variable, return a float
    if _X.shape[1] == 1:
        I = I.iloc[0, 0]

    return I