"""
A library that contains dissimilarity functions for the purpose of comparing
phylogenetic samples.
"""
import numba
import numpy as np
from typing import Dict, List, Optional


def weighted_hamming_distance(
    s1: List[int],
    s2: List[int],
    missing_state=-1,
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """Computes the weighted hamming distance between samples.

    Evaluates the dissimilarity of two phylogenetic samples on the basis of
    their shared indel states and the probability of these indel states
    occurring. Specifically, for a given character, if two states are identical
    we decrement the dissimilarity by the probability of these two occurring
    independently; if the two states disagree, we increment the dissimilarity by
    the probability of these states occurring. We normalize the dissimilarity
    by the number of non-missing characters shared by the two samples.

    If weights are not given, then we increment dissimilarity by +2 if the states
    are different, +1 if one state is uncut and the other is an indel, and +0 if
    the two states are identical.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_char: The character representing missing values
        w: A dictionary storing the state weights for each character, derived
            from the state priors. This should be a nested dictionary where each
            key corresponds to character that then indexes another dictionary
            storing the weight of each observed state.
            (Character -> State -> Weight)

    Returns:
        A dissimilarity score.

    """
    d = 0
    num_present = 0
    for i in range(len(s1)):

        if s1[i] == missing_state or s2[i] == missing_state:
            continue

        num_present += 1

        if s1[i] != s2[i]:
            if s1[i] == 0 or s2[i] == 0:
                if w:
                    if s1[i] != 0:
                        d += w[i][s1[i]]
                    else:
                        d += w[i][s2[i]]
                else:
                    d += 1
            else:
                if w:
                    d += w[i][s1[i]] + w[i][s2[i]]
                else:
                    d += 2
        else:
            if s1[i] != 0:
                if w:
                    d -= 2 * w[i][s1[i]]

    if num_present == 0:
        return 0

    return d / num_present


def hamming_similarity_without_missing(
    s1: List[int],
    s2: List[int],
    missing_char: int,
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """A function to return the number of (non-missing) character/state
    mutations shared by two samples.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_char: The character representing missing values
        w: A set of optional weights to weight the similarity of a mutation
    Returns:
        The number of shared mutations between two samples, weighted or unweighted
    """

    # TODO Optimize this using masks
    num_chars = len(s1)
    assert len(s2) == num_chars
    if w is None:
        return sum(
            [
                1
                for i in range(num_chars)
                if s1[i] == s2[i] and (s1[i] != 0 and s1[i] != missing_char)
            ]
        )
    else:
        return sum(
            [
                w[i][s1[i]]
                for i in range(num_chars)
                if s1[i] == s2[i] and (s1[i] != 0 and s1[i] != missing_char)
            ]
        )


@numba.jit(nopython=True)
def hamming_distance(s1: np.array(int), s2: np.array(int)) -> int:
    """Computes the vanilla hamming distance between two samples.

    Counts the number of positions that two samples disagree at.

    Args:
        s1: The first sample
        s2: The second sample

    Returns:
        The number of positions two nodes disagree at.
    """

    dist = 0
    for i in range(len(s1)):

        if s1[i] != s2[i]:
            dist += 1

    return dist


def negative_weighted_hamming_distance(
    s1: List[int],
    s2: List[int],
    missing_char: int,
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """Computes the negative weighted hamming distance between samples.

    Returns the negative value of the weighted_hamming_distance function above
    as a similarity.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_char: The character representing missing values
        w: A dictionary storing the state weights for each character, derived
            from the state priors. This should be a nested dictionary where each
            key corresponds to character that then indexes another dictionary
            storing the weight of each observed state.
            (Character -> State -> Weight)

    Returns:
        A similarity score.
    """

    return -1 * weighted_hamming_distance(s1, s2, missing_char, w)


def weighted_hamming_similarity(
    s1: List[int],
    s2: List[int],
    missing_char: int,
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """A function to return the weighted number of (non-missing) character/state
    mutations shared by two samples.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_char: The character representing missing values
        w: A set of optional weights to weight the similarity of a mutation

    Returns:
        The weighted number of shared mutations between two samples
    """

    d = 0
    num_present = 0
    for i in range(len(s1)):

        if s1[i] == missing_char or s2[i] == missing_char:
            continue

        num_present += 1

        if s1[i] == s2[i]:
            if s1[i] != 0:
                if w:
                    d += 2 * w[i][s1[i]]
                else:
                    d += 2
            else:
                if not w:
                    d += 1
        else:
            if s1[i] != 0 and s2[i] != 0:
                if w:
                    d -= w[i][s1[i]] - w[i][s2[i]]

    if num_present == 0:
        return 0

    return d / num_present
