"""
A library that contains dissimilarity functions for the purpose of comparing
phylogenetic samples.
"""
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numba
import numpy as np


def weighted_hamming_distance(
    s1: List[int],
    s2: List[int],
    missing_state_indicator=-1,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
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

    Args: s1: Character states of the first sample
    s2: Character states of the second sample
    missing_state_indicator: The character representing missing values
    weights: A dictionary storing the state weights for each character,
        derived from the state priors. This should be a nested dictionary where
        each key corresponds to character that then indexes another dictionary
        storing the weight of each observed state.  (Character -> State ->
        Weight)

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
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """A function to return the number of (non-missing) character/state
    mutations shared by two samples.

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
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """
    A function to return the number of (non-missing) character/state mutations
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
    s1: np.array(int),
    s2: np.array(int),
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
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """A function to return the weighted number of (non-missing) character/state
    mutations shared by two samples.

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
    s1: List[int],
    s2: List[int],
    missing_state_indicator=-1,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:

    """
    Gives a similarity function from the inverse of weighted hamming distance.

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
        [List[int], List[int], int, Dict[int, Dict[int, float]]], float
    ],
    s1: Union[List[int], List[Tuple[int, ...]]],
    s2: Union[List[int], List[Tuple[int, ...]]],
    missing_state_indicator: int,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
    linkage_function: Callable[[Union[np.array, List[float]]], float] = np.mean,
    normalize: bool = True,
) -> float:
    """Compute the dissimilarity between (possibly) ambiguous character strings.

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
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        dissim = []
        present = []
        for _c1, _c2 in itertools.product(c1, c2):
            present.append(
                _c1 != missing_state_indicator and _c2 != missing_state_indicator
            )
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
