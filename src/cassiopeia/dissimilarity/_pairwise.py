"""Compute and store the full pairwise dissimilarity map for a tree.

When the character matrix is integer-castable, it (and the missing-state
indicator) are coerced to integers so the metric is JIT-compiled with ``numba``
(the fast ``nopython`` path); non-numeric or ambiguous matrices fall back to the
slower object path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cassiopeia.dissimilarity._compute import compute_dissimilarity_map
from cassiopeia.dissimilarity._metrics import _resolve_dissimilarity, _uses_weights
from cassiopeia.utils import _get_characters, _get_parameter, _resolve_priors

if TYPE_CHECKING:
    from treedata import TreeData


def _int_missing(missing_state_indicator) -> int:
    """Resolve a single integer missing-state indicator for the numba path.

    The metric ``numba`` kernels compare against a scalar integer, so a parameter
    that may be a tuple of acceptable representations (e.g. the default
    ``(-1, "-1", "NA", "-")``) is collapsed to its integer member (``-1`` if none
    is found).
    """
    if isinstance(missing_state_indicator, (tuple, list, set)):
        for v in missing_state_indicator:
            if isinstance(v, (int, np.integer)):
                return int(v)
        return -1
    return missing_state_indicator


def _as_str_set(value) -> set[str]:
    """Return the set of string representations a parameter may take.

    A parameter resolved by :func:`cassiopeia.utils._get_parameter` may be a
    single value or a tuple of acceptable representations (e.g. the default
    ``(0, "0", "*")`` for the unmodified state).  Any of those representations
    should be treated as equivalent when encoding a character matrix.
    """
    if isinstance(value, (tuple, list, set)):
        return {str(v) for v in value}
    return {str(value)}


def _encode_integer_matrix(
    arr: pd.DataFrame, missing_state_indicator, unmodified_state
) -> tuple[pd.DataFrame, int, dict | None]:
    """Encode a (string/categorical) character matrix to a contiguous int64 frame.

    Maps the unmodified state to ``0``, the missing state to ``-1``, and every
    other distinct value to a distinct positive integer.  This preserves the
    semantics the dissimilarity metrics rely on (equality, missing detection, and
    the special ``0``/unmodified handling of
    :func:`~cassiopeia.dissimilarity.weighted_hamming`) while enabling the fast
    ``numba`` (``nopython``) path.  States are compared by string so
    integer/string conventions (e.g. ``0`` vs ``"0"``) are matched.  Either
    argument may be a single value or a tuple of acceptable representations.

    Returns ``(int64_dataframe, -1, mapping)`` where *mapping* maps each original
    state value to its integer code (use :func:`_encode_priors` to re-key priors
    accordingly). Integer matrices are returned unchanged with ``mapping=None``.
    Ambiguous (tuple) states are encoded state by state into tuples of integer
    codes, so the returned frame has ``object`` dtype.
    """
    values = arr.to_numpy()
    # Integer matrices already use the integer convention; return them unchanged
    # (remapping would break priors keyed by the original state values).
    if np.issubdtype(values.dtype, np.integer):
        return arr, _int_missing(missing_state_indicator), None
    flat = values.ravel()
    ambiguous = values.dtype == object and any(isinstance(v, tuple) for v in flat)
    if ambiguous:
        states = dict.fromkeys(s for v in flat for s in (v if isinstance(v, tuple) else (v,)))
    else:
        states = pd.unique(flat)
    missing_set = _as_str_set(missing_state_indicator)
    unmodified_set = _as_str_set(unmodified_state)
    mapping: dict = {}
    nxt = 1
    for value in states:
        vs = str(value)
        if vs in unmodified_set:
            mapping[value] = 0
        elif vs in missing_set:
            mapping[value] = -1
        else:
            mapping[value] = nxt
            nxt += 1

    if ambiguous:
        out = np.empty(len(flat), dtype=object)
        out[:] = [
            tuple(dict.fromkeys(mapping[s] for s in v)) if isinstance(v, tuple) else mapping[v]
            for v in flat
        ]
        out = out.reshape(arr.shape)
    else:
        out = np.empty(arr.shape, dtype=np.int64)
        for value, code in mapping.items():
            out[values == value] = code
    return pd.DataFrame(out, index=arr.index, columns=arr.columns), -1, mapping


def _encode_priors(priors: dict[int, dict] | None, mapping: dict | None) -> dict[int, dict] | None:
    """Re-key priors' state keys to match an integer-encoded character matrix.

    ``tdata.obsm[characters_key]`` may hold string or integer states, and
    :func:`_encode_integer_matrix` remaps string/categorical states to integers.
    Priors (``{character_index: {state: probability}}``) carry state keys in the
    *original* representation, so they must be remapped with the same *mapping*
    to line up with the encoded matrix.

    Args:
        priors: Priors keyed by character index, or ``None``.
        mapping: Original-state -> integer-code map from
            :func:`_encode_integer_matrix`. ``None`` means the matrix was already
            integer (no remapping), so priors are returned unchanged.

    Returns:
        Priors with inner state keys remapped to the encoded integers. States not
        present in the matrix (absent from *mapping*) are dropped, since they can
        never be looked up.
    """
    if mapping is None or not priors:
        return priors
    str_to_code = {str(orig): code for orig, code in mapping.items()}
    return {
        character: {
            str_to_code[str(state)]: probability
            for state, probability in state_priors.items()
            if str(state) in str_to_code
        }
        for character, state_priors in priors.items()
    }


def _prepare_integer_matrix(
    arr: pd.DataFrame, missing_state_indicator, unmodified_state
) -> tuple[np.array, int, dict | None]:
    """Return ``(array, missing_state_indicator, mapping)`` ready for the numba path.

    Integer matrices are used as-is (``mapping=None``). Other (string or
    categorical) matrices are encoded to integers via
    :func:`_encode_integer_matrix`; re-key priors with :func:`_encode_priors`
    using the returned *mapping*. Matrices with ambiguous (tuple) states are
    encoded to an ``object`` array of integers and tuples of integers (object
    path), so ambiguity is preserved for the cluster metrics and the unmodified
    state is ``0`` as they expect.
    """
    values = arr.to_numpy()
    if np.issubdtype(values.dtype, np.integer):
        return np.ascontiguousarray(values), _int_missing(missing_state_indicator), None
    df, missing, mapping = _encode_integer_matrix(arr, missing_state_indicator, unmodified_state)
    out = df.to_numpy()
    return (out if out.dtype == object else np.ascontiguousarray(out)), missing, mapping


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

    from cassiopeia.utils import _transform_priors

    fn = _resolve_dissimilarity(method)
    if fn is None:
        from cassiopeia.mixins import DistanceSolverError

        raise DistanceSolverError(
            "Please provide a dissimilarity_function or a precomputed dissimilarity map."
        )

    # Only compute dissimilarities between *unique* states to save runtime.
    cell_to_state = characters.astype(str).apply("|".join, axis=1)
    state_to_cells = characters.index.groupby(cell_to_state)
    dedup_character_matrix = characters.drop_duplicates()

    N = dedup_character_matrix.shape[0]
    # Coerce to integers so the metric takes the fast numba (nopython) path.
    char_array, missing, mapping = _prepare_integer_matrix(
        dedup_character_matrix, missing_state_indicator, unmodified_state
    )
    weights = None
    if priors:
        weights = _transform_priors(_encode_priors(priors, mapping), prior_transformation)
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


def _resolve_metric_priors(
    tdata: TreeData,
    priors: dict | list | bool | None,
    method: str | Callable | None,
    characters: pd.DataFrame | None = None,
) -> dict[int, dict] | None:
    """Resolve *priors* for a pairwise dissimilarity *method*.

    Follows :func:`cassiopeia.utils._resolve_priors` (states matched to
    *characters*), except that stored priors are not picked up implicitly
    (``priors=None``) for metrics that ignore weights, and explicitly requested
    priors with such a metric emit a warning.
    """
    if _uses_weights(method):
        return _resolve_priors(tdata, priors, characters)
    if priors is None or priors is False:
        return None
    resolved = _resolve_priors(tdata, priors, characters)
    if resolved:
        warnings.warn(
            f"priors were provided but the dissimilarity function {method!r} "
            "does not use weights, so the priors will be ignored. Use a "
            "weighted metric (e.g. 'weighted_hamming') to make use of priors.",
            UserWarning,
            stacklevel=3,
        )
    return resolved


def pairwise(
    tdata: TreeData,
    method: str | Callable = "nonmissing_hamming",
    characters_key: str = "characters",
    key_added: str = "distances",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict | list | bool | None = None,
    prior_transformation: str = "negative_log",
    threads: int = 1,
    copy: bool = False,
) -> pd.DataFrame | None:
    """Compute pairwise dissimilarities between all cells.

    Analogous to :func:`scipy.spatial.distance.pdist`, but using a phylogenetic
    dissimilarity metric on the character matrix in
    ``tdata.obsm[characters_key]``. By default the full symmetric n×n map is
    stored as a dense ``numpy`` array in ``tdata.obsp[key_added]``.

    Args:
        tdata: TreeData to operate on.
        method: Dissimilarity function.  Accepts a callable or a string name
            of a built-in metric (e.g. ``'nonmissing_hamming'``,
            ``'weighted_hamming'``, ``'hamming'``).
        characters_key: Key in ``tdata.obsm`` for the character matrix.
        key_added: Key in ``tdata.obsp`` under which the result is stored.
        missing_state: Missing-state value (read from ``tdata.uns`` if ``None``).
        unmodified_state: Unmodified/uncut state value (read from ``tdata.uns``
            if ``None``).
        priors: Priors for character states, used by weighted metrics. ``None``
            (default) uses ``tdata.uns["priors"]`` if present; ``True`` requires
            stored priors; ``False`` disables priors; a dict keyed by character
            name or index (``{character: {state: probability}}``), or a list of
            per-character dicts, is used directly.
        prior_transformation: Transformation applied to priors to form weights.
        threads: Threads for parallel computation.
        copy: If ``True``, return the dissimilarity map as a
            :class:`~pandas.DataFrame` indexed by ``tdata.obs_names`` instead of
            storing it in ``tdata.obsp``.

    Returns:
        The dissimilarity map if ``copy=True``, else ``None``.

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

    chars = _get_characters(tdata, characters_key, missing_state)
    missing_state = _get_parameter(tdata, "missing_state", value=missing_state)
    unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    priors = _resolve_metric_priors(tdata, priors, method, chars)
    result = _pairwise(
        chars,
        method,
        missing_state,
        priors,
        prior_transformation,
        threads,
        unmodified_state=unmodified_state,
    )
    result = result.loc[tdata.obs_names, tdata.obs_names]
    if copy:
        return result
    tdata.obsp[key_added] = result.to_numpy()
    return None
