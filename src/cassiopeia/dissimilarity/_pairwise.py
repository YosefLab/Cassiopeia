"""Compute and store the full pairwise dissimilarity map for a tree.

When the character matrix is integer-castable, it (and the missing-state
indicator) are coerced to integers so the metric is JIT-compiled with ``numba``
(the fast ``nopython`` path); non-numeric or ambiguous matrices fall back to the
slower object path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cassiopeia.dissimilarity._compute import compute_dissimilarity_map
from cassiopeia.dissimilarity._metrics import _resolve_dissimilarity

if TYPE_CHECKING:
    from treedata import TreeData


def _encode_integer_matrix(
    arr: np.ndarray, missing_state_indicator, unmodified_state
) -> tuple[np.ndarray, int]:
    """Encode a (string/categorical) character matrix to a contiguous int64 array.

    Maps the unmodified state to ``0``, the missing state to ``-1``, and every
    other distinct value to a distinct positive integer.  This preserves the
    semantics the dissimilarity metrics rely on (equality, missing detection, and
    the special ``0``/unmodified handling of
    :func:`~cassiopeia.dissimilarity.weighted_hamming`) while enabling the fast
    ``numba`` (``nopython``) path.  States are compared by string so
    integer/string conventions (e.g. ``0`` vs ``"0"``) are matched.

    Returns ``(int64_array, -1)`` (the second value is the integer missing
    indicator to use with the encoded matrix).
    """
    missing_s = str(missing_state_indicator)
    unmodified_s = str(unmodified_state)
    mapping: dict = {}
    nxt = 1
    for value in pd.unique(arr.ravel()):
        vs = str(value)
        if vs == unmodified_s:
            mapping[value] = 0
        elif vs == missing_s:
            mapping[value] = -1
        else:
            mapping[value] = nxt
            nxt += 1

    out = np.empty(arr.shape, dtype=np.int64)
    for value, code in mapping.items():
        out[arr == value] = code
    return np.ascontiguousarray(out), -1


def _prepare_integer_matrix(
    arr: np.ndarray, missing_state_indicator, unmodified_state, weights
) -> tuple[np.ndarray, int | float]:
    """Return ``(array, missing_state_indicator)`` ready for the numba path.

    Integer matrices are used as-is.  When weights are *not* provided, non-integer
    (string/categorical) matrices are encoded to integers via
    :func:`_encode_integer_matrix` for the fast path.  When weights *are*
    provided (state-keyed), only a lossless numeric cast is attempted so the
    weight keys stay valid; otherwise the matrix is left as-is (slower object
    path).
    """
    if np.issubdtype(arr.dtype, np.integer):
        return arr, missing_state_indicator
    if weights is None:
        return _encode_integer_matrix(arr, missing_state_indicator, unmodified_state)
    try:
        converted = np.ascontiguousarray(arr.astype(np.int64))
        return converted, int(missing_state_indicator)
    except (ValueError, TypeError):
        return arr, missing_state_indicator


def _pairwise(
    characters: pd.DataFrame,
    method: str | Callable | None,
    missing_state_indicator: int = -1,
    priors: dict | None = None,
    prior_transformation: str = "negative_log",
    threads: int = 1,
    unmodified_state=0,
) -> pd.DataFrame:
    """Compute a symmetric n×n pairwise dissimilarity map from a character matrix.

    A pure helper that depends only on a character ``pd.DataFrame`` (no
    :class:`~cassiopeia.data.CassiopeiaTree` or :class:`~treedata.TreeData`).
    Dissimilarities are computed over *unique* states for efficiency and then
    expanded back to all samples.

    Args:
        characters: Character matrix (samples × characters), indexed by sample
            name.
        method: Dissimilarity function: a callable or the string name of a
            built-in metric.
        missing_state_indicator: Missing state indicator value.
        priors: Character priors dict, or ``None``.
        prior_transformation: Transformation applied to priors to form weights.
        threads: Threads for parallel computation.
        unmodified_state: Value representing the unmodified (uncut) state, used
            when encoding string/categorical matrices to integers.

    Returns:
        Symmetric pairwise distance ``pd.DataFrame`` indexed (and columned) by
        the rows of *characters*, in the original order.

    Raises:
        DistanceSolverError: If *method* resolves to ``None``.
    """
    import scipy.spatial.distance

    from cassiopeia.solver import solver_utilities

    fn = _resolve_dissimilarity(method)
    if fn is None:
        from cassiopeia.mixins import DistanceSolverError

        raise DistanceSolverError(
            "Please provide a dissimilarity_function or a precomputed dissimilarity map."
        )

    weights = None
    if priors:
        weights = solver_utilities.transform_priors(priors, prior_transformation)

    # Only compute dissimilarities between *unique* states to save runtime.
    cell_to_state = characters.astype(str).apply("|".join, axis=1)
    state_to_cells = characters.index.groupby(cell_to_state)
    dedup_character_matrix = characters.drop_duplicates()

    N = dedup_character_matrix.shape[0]
    # Coerce to integers so the metric takes the fast numba (nopython) path.
    char_array, missing = _prepare_integer_matrix(
        dedup_character_matrix.to_numpy(), missing_state_indicator, unmodified_state, weights
    )
    condensed = compute_dissimilarity_map(
        char_array,
        N,
        fn,
        weights,
        missing,
        threads=threads,
    )
    square = scipy.spatial.distance.squareform(condensed)

    # Expand the deduplicated map back to all cells.
    full = np.pad(square, (0, characters.shape[0] - N))
    cells = list(dedup_character_matrix.index)
    j = N
    for i, dedup_cell in enumerate(dedup_character_matrix.index):
        for cell in state_to_cells[cell_to_state[dedup_cell]]:
            if dedup_cell == cell:
                continue
            dissimilarities = full[i, :]
            full[j, :] = dissimilarities
            full[:, j] = dissimilarities
            cells.append(cell)
            j += 1

    result = pd.DataFrame(full, index=cells, columns=cells)
    # Restore the original sample order.
    order = list(characters.index)
    return result.loc[order, order]


def pairwise(
    tdata: TreeData,
    method: str | Callable = "nonmissing_hamming",
    characters_key: str | None = None,
    key_added: str = "distances",
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> None:
    """Compute pairwise dissimilarities between all leaves and store in *tdata*.

    Analogous to :func:`scipy.spatial.distance.pdist`, but using a phylogenetic
    dissimilarity metric and storing the full symmetric n×n map on the
    :class:`~treedata.TreeData` object.  The map is computed from
    ``tdata.obsm[characters_key or 'characters']`` and stored as a dense
    ``numpy`` array in ``tdata.obsp[key_added]``.

    Args:
        tdata: TreeData to modify in-place.
        method: Dissimilarity function.  Accepts a callable or a string name
            of a built-in metric (e.g. ``'nonmissing_hamming'``,
            ``'weighted_hamming'``, ``'hamming'``).
        characters_key: ``obsm`` key for the character matrix (default
            ``'characters'``).
        key_added: ``obsp`` key under which the result is stored.
        prior_transformation: Transformation applied to priors when computing
            dissimilarity weights.
        threads: Threads for parallel computation.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        ValueError: If no character matrix is found.
    """
    from treedata import TreeData

    if not isinstance(tdata, TreeData):
        raise TypeError(
            "pairwise() operates on TreeData. For a CassiopeiaTree, convert with "
            "CassiopeiaTree.to_treedata()."
        )

    from cassiopeia.solver import solver_utilities

    chars = solver_utilities._get_characters(tdata, characters_key)
    if chars is None:
        raise ValueError(
            "TreeData has no character matrix; store characters in "
            f"obsm[{characters_key or 'characters'!r}]."
        )
    missing = tdata.uns.get("missing_state_indicator", -1)
    unmodified = tdata.uns.get("unmodified_state", 0)
    priors = tdata.uns.get("priors", None)
    result = _pairwise(
        chars,
        method,
        missing,
        priors,
        prior_transformation,
        threads,
        unmodified_state=unmodified,
    )
    tdata.obsp[key_added] = result.to_numpy()
