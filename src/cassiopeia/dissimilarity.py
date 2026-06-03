"""Pairwise dissimilarity metrics and computation for phylogenetic samples.

This module contains the dissimilarity metric functions used to compare
phylogenetic samples (formerly :mod:`cassiopeia.solver.dissimilarity_functions`)
as well as :func:`pairwise`, which computes and stores the full pairwise
dissimilarity map for a tree (formerly ``cassiopeia.solver.dissimilarity``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from treedata import TreeData


# ── Dissimilarity metric functions ────────────────────────────────────────────


def weighted_hamming_distance(
    s1: list[int],
    s2: list[int],
    missing_state_indicator=-1,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    """Computes the weighted hamming distance between samples.

    Evaluates the dissimilarity of two phylogenetic samples on the basis of
    their shared indel states and the probability of these indel states
    occurring. Specifically, for a given character, if two states are identical
    we decrement the dissimilarity by the probability of these two occurring
    independently; if the two states disagree, we increment the dissimilarity by
    the probability of these states occurring. We normalize the dissimilarity by
    the number of non-missing characters shared by the two samples.

    If weights are not given, then we increment dissimilarity by +2 if the
    states are different, +1 if one state is uncut and the other is an indel,
    and +0 if the two states are identical.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A dictionary storing the state weights for each character,
            derived from the state priors. This should be a nested
            dictionary where each key corresponds to character that then indexes
            another dictionary storing the weight of each observed state.
            (Character -> State -> Weight)

    Returns:
            A dissimilarity score.

    """
    d = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue

        num_present += 1

        if s1[i] != s2[i]:
            if s1[i] == 0 or s2[i] == 0:
                if weights:
                    if s1[i] != 0:
                        d += weights[i][s1[i]]
                    else:
                        d += weights[i][s2[i]]
                else:
                    d += 1
            else:
                if weights:
                    d += weights[i][s1[i]] + weights[i][s2[i]]
                else:
                    d += 2

    if num_present == 0:
        return 0

    return d / num_present


def hamming_similarity_without_missing(
    s1: list[int],
    s2: list[int],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    """A function to return the number of (non-missing) character/state mutations shared by two samples.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a
            mutation
     Returns:
        The number of shared mutations between two samples, weighted or
            unweighted
    """
    # TODO Optimize this using masks
    similarity = 0
    for i in range(len(s1)):
        if (
            s1[i] == missing_state_indicator
            or s2[i] == missing_state_indicator
            or s1[i] == 0
            or s2[i] == 0
        ):
            continue

        if s1[i] == s2[i]:
            if weights:
                similarity += weights[i][s1[i]]
            else:
                similarity += 1

    return similarity


def hamming_similarity_normalized_over_missing(
    s1: list[int],
    s2: list[int],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    """Number of shared mutations normalized over missing data.

    The number of (non-missing) character/state mutations
    shared by two samples, normalized over the amount of missing data.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a
            mutation

    Returns:
            The number of shared mutations between two samples normalized over the
            number of missing data events, weighted or unweighted
    """
    # TODO Optimize this using masks
    similarity = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue

        num_present += 1

        if s1[i] == 0 or s2[i] == 0:
            continue

        if s1[i] == s2[i]:
            if weights:
                similarity += weights[i][s1[i]]
            else:
                similarity += 1

    if num_present == 0:
        return 0

    return similarity / num_present


@numba.jit(nopython=True)
def hamming_distance(
    s1: np.ndarray,
    s2: np.ndarray,
    ignore_missing_state: bool = False,
    missing_state_indicator: int = -1,
) -> int:
    """Computes the vanilla hamming distance between two samples.

    Counts the number of positions that two samples disagree at. A user can
    optionally specify to ignore missing data.

    Args:
        s1: The first sample
        s2: The second sample
        ignore_missing_state: Ignore comparisons where one is the missing state
            indicator
        missing_state_indicator: Indicator for missing data.

    Returns:
            The number of positions two nodes disagree at.
    """
    dist = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if (
                s1[i] == missing_state_indicator or s2[i] == missing_state_indicator
            ) and ignore_missing_state:
                dist += 0
            else:
                dist += 1

    return dist


def weighted_hamming_similarity(
    s1: list[int],
    s2: list[int],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    """The weighted number of (non-missing) character/state mutations shared by two samples.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a
            mutation

    Returns:
            The weighted number of shared mutations between two samples
    """
    d = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue

        num_present += 1

        if s1[i] == s2[i]:
            if s1[i] != 0:
                if weights:
                    d += 2 * weights[i][s1[i]]
                else:
                    d += 2
            else:
                if not weights:
                    d += 1

    if num_present == 0:
        return 0

    return d / num_present


def exponential_negative_hamming_distance(
    s1: list[int],
    s2: list[int],
    missing_state_indicator=-1,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    """Gives a similarity function from the inverse of weighted hamming distance.

    This simply returns exp(-d(i,j)) where d is the weighted_hamming_distance
    function as above where no weights are passed in.  In other words, we
    increment d by +2 if the states are different, +1 if one state is uncut and
    the other is an indel, and +0 if the two states are identical. Then, we take
    this total d and return exp(-d). Note that since d is a metric,
    'exponential_negative_hamming_distance' is a multiplicative similarity
    score, i.e. s(i, j) = s(i, k) * s(k, j) for k on the path between i and j.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A dictionary storing the state weights for each character,
            derived from the state priors. This should be a nested
            dictionary where each key corresponds to character that then indexes
            another dictionary storing the weight of each observed state.
            (Character -> State -> Weight)

    Returns:
            A similarity score.
    """
    d = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue

        num_present += 1

        if s1[i] != s2[i]:
            if s1[i] == 0 or s2[i] == 0:
                if weights:
                    if s1[i] != 0:
                        d += weights[i][s1[i]]
                    else:
                        d += weights[i][s2[i]]
                else:
                    d += 1
            else:
                if weights:
                    d += weights[i][s1[i]] + weights[i][s2[i]]
                else:
                    d += 2

    if num_present == 0:
        return 0

    weighted_hamm_dist = d / num_present

    return np.exp(-weighted_hamm_dist)


def cluster_dissimilarity(
    dissimilarity_function: Callable[
        [list[int], list[int], int, dict[int, dict[int, float]]], float
    ],
    s1: list[int] | list[tuple[int, ...]],
    s2: list[int] | list[tuple[int, ...]],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
    linkage_function: Callable[[np.ndarray | list[float]], float] = np.mean,
    normalize: bool = True,
) -> float:
    r"""Compute the dissimilarity between (possibly) ambiguous character strings.

    An ambiguous character string is a character string in
    which each character contains an tuple of possible states, and such a
    character string is represented as a list of tuples of integers.

    A naive implementation is to first disambiguate each of the two
    ambiguous character strings by generating all possible strings, then
    computing the dissimilarity between all pairwise combinations, and finally
    applying the linkage function on the calculated dissimilarities. However,
    doing so has complexity O(\prod_{i=1}^N |a_i| x |b_i|) where N is the number
    of target sites, |a_i| is the number of ambiguous characters at target site
    i of string a, and |b_i| is the number of amiguous characters at target site
    i of string b.  As an example, if we have N=10 and all a_i=b_i=2, then we
    have to construct 1,038,576 * 2 strings and compute over 4 trillion
    dissimilarities.

    By assuming each target site is independent, simply calculating the sum of
    the linkages of each target site separately is equivalent to the naive
    implementation (can be proven by induction). This procedure is implemented
    in this function. One caveat is that we usually normalize the distance by
    the number of shared non-missing positions. We approximate this by dividing
    the absolute distance by the sum of the probability of each site not being
    a missing site for both strings.

    The idea of linkage is analogous to that in hierarchical clustering, where
    ``np.min`` can be used for single linkage, ``np.max`` for complete linkage,
    and ``np.mean`` for average linkage (the default).

    The reason the ``dissimilarity_function`` argument is defined as the
    first argument is so that this function may be used as input to
    :func:`cassiopeia.data.CassiopeiaTree.compute_dissimilarity_map`. This can
    be done by partial application of this function with the desired
    dissimilarity function.

    Note:
        If neither character string is ambiguous, then calling this function is
        equivalent to calling ``dissimilarity_function`` separately.

    Args:
        s1: The first (possibly) ambiguous sample
        s2: The second (possibly) ambiguous sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a
            mutation
        dissimilarity_function: The dissimilarity function to use to
            calculate pairwise dissimilarities.
        linkage_function: The linkage function to use to aggregate
        dissimilarities into a single number. Defaults to ``np.mean`` for
            average linkage.
        normalize: Whether to normalize to the proportion of sites present in
            both strings.

    Returns:
            The dissimilarity between the two ambiguous samples
    """
    # Make any unambiguous character strings into pseudo-ambiguous so that we
    # can easily use itertools.product
    s1 = [s if isinstance(s, tuple) else (s,) for s in s1]
    s2 = [s if isinstance(s, tuple) else (s,) for s in s2]

    result = 0
    num_present = 0
    for i, (c1, c2) in enumerate(zip(s1, s2, strict=False)):
        dissim = []
        present = []
        for _c1, _c2 in itertools.product(c1, c2):
            present.append(_c1 != missing_state_indicator and _c2 != missing_state_indicator)
            dissim.append(
                dissimilarity_function(
                    [_c1],
                    [_c2],
                    missing_state_indicator,
                    {0: weights[i]} if weights else None,
                )
            )
        result += linkage_function(dissim)
        num_present += np.mean(present)

    if num_present == 0:
        return 0

    return result / num_present if normalize else result


def cluster_dissimilarity_weighted_hamming_distance_min_linkage(
    s1: list[int] | list[tuple[int, ...]],
    s2: list[int] | list[tuple[int, ...]],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    r"""Compute the dissimilarity between (possibly) ambiguous character strings.

    An ambiguous character string is a character string in
    which each character contains an tuple of possible states, and such a
    character string is represented as a list of tuples of integers.

    A naive implementation is to first disambiguate each of the two
    ambiguous character strings by generating all possible strings, then
    computing the dissimilarity between all pairwise combinations, and finally
    applying the linkage function on the calculated dissimilarities. However,
    doing so has complexity O(\prod_{i=1}^N |a_i| x |b_i|) where N is the number
    of target sites, |a_i| is the number of ambiguous characters at target site
    i of string a, and |b_i| is the number of amiguous characters at target site
    i of string b.  As an example, if we have N=10 and all a_i=b_i=2, then we
    have to construct 1,038,576 * 2 strings and compute over 4 trillion
    dissimilarities.

    By assuming each target site is independent, simply calculating the sum of
    the linkages of each target site separately is equivalent to the naive
    implementation (can be proven by induction). This procedure is implemented
    in this function. One caveat is that we usually normalize the distance by
    the number of shared non-missing positions. We approximate this by dividing
    the absolute distance by the sum of the probability of each site not being
    a missing site for both strings.

    Note:
        If neither character string is ambiguous, then calling this function is
        equivalent to calling ``dissimilarity_function`` separately.

    Args:
        s1: The first (possibly) ambiguous sample
        s2: The second (possibly) ambiguous sample
        missing_state_indicator: The character representing missing values
        weights: A set of optional weights to weight the similarity of a
            mutation

    Returns:
            The dissimilarity between the two ambiguous samples
    """
    # Make any unambiguous character strings into pseudo-ambiguous so that we
    # can easily iterate through combinations
    s1 = [list(s) if isinstance(s, tuple) else [s] for s in s1]
    s2 = [list(s) if isinstance(s, tuple) else [s] for s in s2]

    result = 0
    num_present = 0
    for i in range(len(s1)):
        c1, c2 = s1[i], s2[i]

        dissim = []
        present = 0
        total = 0
        for _c1 in c1:
            for _c2 in c2:
                d = 0

                total += 1
                if _c1 != missing_state_indicator and _c2 != missing_state_indicator:
                    present += 1

                if (_c1 != _c2) and (
                    _c1 != missing_state_indicator and _c2 != missing_state_indicator
                ):
                    if _c1 == 0 or _c2 == 0:
                        if weights:
                            if _c1 != 0:
                                d += weights[i][_c1]
                            else:
                                d += weights[i][_c2]
                        else:
                            d += 1
                    else:
                        if weights:
                            d += weights[i][_c1] + weights[i][_c2]
                        else:
                            d += 2
                dissim.append(d)

        result += np.min(np.array(dissim))
        num_present += present / total

    if num_present == 0:
        return 0

    return result / num_present


# ── Resolution and computation ────────────────────────────────────────────────

# Built-in metrics resolvable by string name.
_DISSIMILARITY_FUNCTIONS: dict[str, Callable] = {
    "weighted_hamming_distance": weighted_hamming_distance,
    "hamming_similarity_without_missing": hamming_similarity_without_missing,
    "hamming_similarity_normalized_over_missing": hamming_similarity_normalized_over_missing,
    "hamming_distance": hamming_distance,
    "weighted_hamming_similarity": weighted_hamming_similarity,
    "exponential_negative_hamming_distance": exponential_negative_hamming_distance,
    "cluster_dissimilarity": cluster_dissimilarity,
    "cluster_dissimilarity_weighted_hamming_distance_min_linkage": (
        cluster_dissimilarity_weighted_hamming_distance_min_linkage
    ),
}


def _resolve_dissimilarity(
    dissimilarity: str | Callable | None,
) -> Callable | None:
    """Return a callable dissimilarity function given a string name or callable.

    Args:
        dissimilarity: A callable, a string name of a built-in metric in this
            module, or ``None``.

    Returns:
        The resolved callable, or ``None`` if *dissimilarity* was ``None``.

    Raises:
        ValueError: If *dissimilarity* is a string not found in the module.
        TypeError: If *dissimilarity* is not a string, callable, or ``None``.
    """
    if dissimilarity is None or callable(dissimilarity):
        return dissimilarity
    if isinstance(dissimilarity, str):
        fn = _DISSIMILARITY_FUNCTIONS.get(dissimilarity)
        if fn is None:
            available = sorted(_DISSIMILARITY_FUNCTIONS)
            raise ValueError(
                f"Unknown dissimilarity function {dissimilarity!r}. Available: {available}"
            )
        return fn
    raise TypeError(
        f"dissimilarity must be a string, callable, or None, got {type(dissimilarity).__name__!r}"
    )


def _pairwise(
    characters: pd.DataFrame,
    method: str | Callable | None,
    missing_state_indicator: int = -1,
    priors: dict | None = None,
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> pd.DataFrame:
    """Compute a symmetric n×n pairwise dissimilarity map from a character matrix.

    A pure helper that depends only on a character ``pd.DataFrame`` (no
    :class:`~cassiopeia.data.CassiopeiaTree` or :class:`~treedata.TreeData`).
    Dissimilarities are computed over *unique* states for efficiency and then
    expanded back to all samples, mirroring the standard Cassiopeia computation.

    Args:
        characters: Character matrix (samples × characters), indexed by sample
            name.
        method: Dissimilarity function: a callable or the string name of a
            built-in metric in this module.
        missing_state_indicator: Missing state indicator value.
        priors: Character priors dict, or ``None``.
        prior_transformation: Transformation applied to priors to form weights.
        threads: Threads for parallel computation.

    Returns:
        Symmetric pairwise distance ``pd.DataFrame`` indexed (and columned) by
        the rows of *characters*, in the original order.

    Raises:
        DistanceSolverError: If *method* resolves to ``None``.
    """
    import scipy.spatial.distance

    from cassiopeia.data import utilities as data_utilities
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
    condensed = data_utilities.compute_dissimilarity_map(
        dedup_character_matrix.to_numpy(),
        N,
        fn,
        weights,
        missing_state_indicator,
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
    method: str | Callable = "weighted_hamming_distance",
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
            of a built-in metric in this module (e.g.
            ``'weighted_hamming_distance'``, ``'hamming_distance'``).
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
    priors = tdata.uns.get("priors", None)
    result = _pairwise(chars, method, missing, priors, prior_transformation, threads)
    tdata.obsp[key_added] = result.to_numpy()
