"""
A library that contains dissimilarity functions for the purpose of comparing
phylogenetic samples.
"""
from typing import Dict, List, Optional

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
    the probability of these states occurring. We normalize the dissimilarity
    by the number of non-missing characters shared by the two samples.

    If weights are not given, then we increment dissimilarity by +2 if the states
    are different, +1 if one state is uncut and the other is an indel, and +0 if
    the two states are identical.

    Args:
        s1: Character states of the first sample
        s2: Character states of the second sample
        missing_state_indicator: The character representing missing values
        weights: A dictionary storing the state weights for each character, derived
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
        weights: A set of optional weights to weight the similarity of a mutation
    Returns:
        The number of shared mutations between two samples, weighted or unweighted
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


@numba.jit(nopython=True)
def hamming_distance(
    s1: np.array(int),
    s2: np.array(int),
    ignore_missing_state: bool = False,
    missing_state_indicator: int = -1,
) -> int:
    """Computes the vanilla hamming distance between two samples.

    Counts the number of positions that two samples disagree at.

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
                s1[i] == missing_state_indicator
                or s2[i] == missing_state_indicator
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
        weights: A set of optional weights to weight the similarity of a mutation

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
