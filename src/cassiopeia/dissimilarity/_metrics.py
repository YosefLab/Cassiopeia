"""Pairwise dissimilarity (distance) metrics for phylogenetic samples.

The per-pair metrics are simple numeric loops that are JIT-compiled with
``numba`` at compute time by :func:`cassiopeia.dissimilarity.compute_dissimilarity_map`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np


def weighted_hamming(
    s1: list[int],
    s2: list[int],
    missing_state_indicator=-1,
    weights: dict[int, dict[int, float]] | None = None,
    unmodified_state=0,
) -> float:
    """Weighted hamming distance between two samples (ignores missing positions).

    Positions where either sample is missing are ignored, and the dissimilarity
    is normalized by the number of non-missing shared characters.  Without
    weights, a mismatch contributes +2, a mismatch where one state is the
    unmodified (uncut) state contributes +1, and identical states contribute +0.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A nested dictionary of per-(character, state) weights derived
            from the priors (character -> state -> weight), or ``None``.
        unmodified_state: The state representing the unmodified (uncut) site.

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
            if s1[i] == unmodified_state or s2[i] == unmodified_state:
                if weights:
                    if s1[i] != unmodified_state:
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


def hamming(
    s1: list[int],
    s2: list[int],
    missing_state_indicator: int = -1,
    weights: dict[int, dict[int, float]] | None = None,
) -> int:
    """Number of positions at which two samples disagree.

    A missing state is treated as an ordinary value, so a position where exactly
    one sample is missing counts as a difference.  Use :func:`nonmissing_hamming`
    to ignore (and normalize over) missing positions.

    Args:
        s1: The first sample
        s2: The second sample
        missing_state_indicator: Unused; present for a uniform metric signature.
        weights: Unused; present for a uniform metric signature.

    Returns:
        The number of positions two samples disagree at.
    """
    dist = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            dist += 1
    return dist


def nonmissing_hamming(
    s1: list[int],
    s2: list[int],
    missing_state_indicator: int = -1,
    weights: dict[int, dict[int, float]] | None = None,
    unmodified_state=0,
) -> float:
    """Hamming-style dissimilarity over non-missing positions.

    Positions where either sample is missing are ignored.  A mismatch
    contributes +2, a mismatch where one state is the unmodified (uncut) state
    contributes +1, and identical states contribute +0; the total is normalized
    by the number of non-missing shared positions (as in
    :func:`weighted_hamming` without weights).

    Args:
        s1: The first sample
        s2: The second sample
        missing_state_indicator: The character representing missing values.
        weights: Unused; present for a uniform metric signature.
        unmodified_state: The state representing the unmodified (uncut) site.

    Returns:
        The normalized dissimilarity over non-missing positions.
    """
    d = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue
        num_present += 1
        if s1[i] != s2[i]:
            if s1[i] == unmodified_state or s2[i] == unmodified_state:
                d += 1
            else:
                d += 2

    if num_present == 0:
        return 0

    return d / num_present


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

    An ambiguous character string is a character string in which each character
    contains a tuple of possible states, represented as a list of tuples of
    integers.  Assuming target-site independence, the dissimilarity is the sum
    over sites of the linkage of pairwise state dissimilarities, normalized by
    the number of shared non-missing positions.

    The ``dissimilarity_function`` argument is first so that this function can be
    partially applied and passed to
    :func:`cassiopeia.dissimilarity.compute_dissimilarity_map`.

    Args:
        dissimilarity_function: The per-state dissimilarity function.
        s1: The first (possibly) ambiguous sample.
        s2: The second (possibly) ambiguous sample.
        missing_state_indicator: The character representing missing values.
        weights: Optional per-(character, state) weights.
        linkage_function: Linkage to aggregate dissimilarities (``np.mean`` for
            average linkage, ``np.min``/``np.max`` for single/complete).
        normalize: Whether to normalize by the proportion of shared present sites.

    Returns:
        The dissimilarity between the two ambiguous samples.
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


def cluster_weighted_hamming(
    s1: list[int] | list[tuple[int, ...]],
    s2: list[int] | list[tuple[int, ...]],
    missing_state_indicator: int,
    weights: dict[int, dict[int, float]] | None = None,
) -> float:
    r"""Weighted-hamming, min-linkage dissimilarity for (possibly) ambiguous strings.

    Equivalent to :func:`cluster_dissimilarity` with the weighted-hamming metric
    and single (min) linkage, specialized for speed.

    Args:
        s1: The first (possibly) ambiguous sample.
        s2: The second (possibly) ambiguous sample.
        missing_state_indicator: The character representing missing values.
        weights: Optional per-(character, state) weights.

    Returns:
        The dissimilarity between the two ambiguous samples.
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


# Built-in metrics resolvable by string name.
_DISSIMILARITY_FUNCTIONS: dict[str, Callable] = {
    "weighted_hamming": weighted_hamming,
    "hamming": hamming,
    "nonmissing_hamming": nonmissing_hamming,
    "cluster_dissimilarity": cluster_dissimilarity,
    "cluster_weighted_hamming": cluster_weighted_hamming,
}

# Metrics that actually consume the per-(character, state) ``weights`` derived
# from priors. Others ignore weights, so supplying priors with them is a no-op.
_WEIGHTED_DISSIMILARITY_FUNCTIONS = {weighted_hamming, cluster_weighted_hamming}


def _uses_weights(dissimilarity: str | Callable | None) -> bool:
    """Whether a dissimilarity metric consumes prior-derived weights.

    Resolves a string name to its callable and unwraps :func:`functools.partial`
    wrappers, inspecting any bound sub-metric (e.g. a ``cluster_dissimilarity``
    partial built around ``weighted_hamming``).
    """
    import functools

    fn = _resolve_dissimilarity(dissimilarity)
    candidates: list = []
    while isinstance(fn, functools.partial):
        candidates.extend(fn.args)
        candidates.extend(fn.keywords.values())
        fn = fn.func
    candidates.append(fn)
    return any(c in _WEIGHTED_DISSIMILARITY_FUNCTIONS for c in candidates)


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
