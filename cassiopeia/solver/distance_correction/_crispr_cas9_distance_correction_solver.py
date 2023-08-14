"""
This file stores the CRISPRCas9DistanceCorrectionSolver class, which enables the
estimation of tree topologies from _corrected_ distances, which are typically
superior to raw distances such as the Hamming distance.
"""
from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree

from ..CassiopeiaSolver import CassiopeiaSolver
from ..DistanceSolver import DistanceSolver
from ..NeighborJoiningSolver import NeighborJoiningSolver

CharacterMatrixType = pd.DataFrame
CharacterStatesType = List[int]
MissingDataIndicatorType = int
MutationProportionType = float
CollisionProbabilityType = float
MutationProportionEstimatorType = Callable[
    [CharacterMatrixType], MutationProportionType
]
CollisionProbabilityEstimatorType = Callable[
    [CharacterMatrixType], CollisionProbabilityType
]
CRISPRCas9DistanceCorrectorType = Callable[
    [
        MutationProportionType,
        CollisionProbabilityType,
        CharacterStatesType,
        CharacterStatesType,
        MissingDataIndicatorType,
    ],
    float,
]


def inverse(
    f: Callable[[float], float],
    y: float,
    lower: float,
    upper: float,
) -> float:
    """
    Invert an increasing function.

    Finds x s.t. f(x) = y. Search for x is limited to the range [lower, upper].
    Uses 30 fixed iterations of binary search.

    Args:
        f: Function to invert.
        y: Value to invert.
        lower: Lower bound on the inverse.
        upper: Upper bound on the inverse.

    Returns:
        f^{-1}(y) restricted to the range [lower, upper].
    """
    for iteration in range(30):
        mid = (upper + lower) / 2.0
        if f(mid) < y:
            lower, upper = mid, upper
        else:
            lower, upper = lower, mid
    return (upper + lower) / 2.0


def hamming_distance(
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int = -1,
) -> float:
    """
    (Scaled) Hamming distance, ignoring missing data.

    Here, by `scaled` we mean that the Hamming distance is divided by the total
    number of characters considered, thus bringing it into the range [0, 1].

    Args:
        s1: Character states of first leaf.
        s2: Character states of second leaf.
        missing_state_indicator: Missing data indicator.

    Returns:
        (Scaled) Hamming distance between s1 and s2, ignoring missing data.
    """
    dist = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue
        num_present += 1
        if s1[i] != s2[i]:
            dist += 1
        else:
            dist += 0
    if num_present == 0:
        return 0
    return dist / num_present


def crispr_cas9_expected_hamming_distance(
    height: float,
    mutation_proportion: float,
    collision_probability: float,
) -> float:
    """
    Expected Hamming distance under the CRISPR-Cas9 model.

    Expected Hamming distance function for two leaves with LCA at a height
    of `height` in an ultrametric tree of depth 1.

    Args:
        height: The height of the LCA of the two leaves.
        mutation_proportion: Mutation proportion parameter of the model.
        collision_probability: Collision probability parameter of the model.

    Returns:
        The expected Hamming distance.
    """
    h = height
    exp_minus_r = 1.0 - mutation_proportion
    q = collision_probability
    if exp_minus_r < 1e-8:
        # To avoid doing log(0) later on.
        return 0.0
    else:
        r = -np.log(exp_minus_r)  # The estimate of the mutation rate.
        return np.exp(-r) * (
            ((1 - q) * np.exp(r * h)) + (2 * q) - ((1 + q) * np.exp(-r * h))
        )


def crispr_cas9_corrected_hamming_distance(
    mutation_proportion: float,
    collision_probability: float,
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int,
) -> float:
    """
    Corrected Hamming distance.

    The "corrected" distance is an estimate of true tree distance.

    It is assumed that the tree is ultrametric of depth 1.

    Args:
        mutation_proportion: Mutation proportion parameter of the model.
        collision_probability: Collision probability parameter of the model.
        s1: Character states of the first leaf.
        s2: Character states of the second leaf.
        missing_state_indicator: Missing state indicator.

    Returns:
        The corrected distance.
    """
    expected_hamming_distance_given_height = partial(
        crispr_cas9_expected_hamming_distance,
        mutation_proportion=mutation_proportion,
        collision_probability=collision_probability,
    )
    observed_hamming_distance = hamming_distance(
        s1=s1, s2=s2, missing_state_indicator=missing_state_indicator
    )
    height = inverse(
        f=expected_hamming_distance_given_height,
        y=observed_hamming_distance,
        lower=0,
        upper=1,
    )
    estimated_tree_distance = 2 * height
    return estimated_tree_distance


def ternary_hamming_distance(
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int = -1,
) -> float:
    """
    (Scaled) ternary Hamming distance, ignoring missing data.

    Here, by `scaled` we mean that the Hamming distance is divided by the total
    number of characters considered, thus bringing it into the range [0, 2].

    Here, `ternary` means that we score two mutated states that are different
    as 2 rather then 1.

    Args:
        s1: Character states of first leaf
        s2: Character states of second leaf.
        missing_state_indicator: Missing data indicator.

    Returns:
        (Scaled) ternary Hamming distance between s1 and s2, ignoring missing
        data.
    """
    dist = 0
    num_present = 0
    for i in range(len(s1)):
        if s1[i] == missing_state_indicator or s2[i] == missing_state_indicator:
            continue
        num_present += 1
        if s1[i] != s2[i]:
            if s1[i] > 0 and s2[i] > 0:
                dist += 2
            else:
                dist += 1
        else:
            dist += 0
    if num_present == 0:
        return 0
    return dist / num_present


def crispr_cas9_expected_ternary_hamming_distance(
    height: float,
    mutation_proportion: float,
    collision_probability: float,
) -> float:
    """
    Expected ternary Hamming distance under the CRISPR-Cas9 model.

    Expected ternary Hamming distance function for two leaves with LCA at a
    height of `height` in an ultrametric tree of depth 1.

    Args:
        height: The height of the LCA of the two leaves.
        mutation_proportion: Mutation proportion parameter of the model.
        collision_probability: Collision probability parameter of the model.

    Returns:
        The expected ternary Hamming distance.
    """
    h = height
    exp_minus_r = 1.0 - mutation_proportion
    q = collision_probability
    if exp_minus_r < 1e-8:
        # To avoid doing log(0) later on.
        return 0.0
    else:
        r = -np.log(exp_minus_r)  # The estimate of the mutation rate.
        return (
            2 * (1 - np.exp(-h * r)) ** 2 * (1 - q)
            + 2 * (1 - np.exp(-h * r)) * (np.exp(-h * r))
        ) * (np.exp(r * (h - 1)))


def crispr_cas9_corrected_ternary_hamming_distance(
    mutation_proportion: float,
    collision_probability: float,
    s1: List[int],
    s2: List[int],
    missing_state_indicator: int,
) -> float:
    """
    Corrected ternary Hamming distance.

    The "corrected" distance is an estimate of true tree distance.

    It is assumed that the tree is ultrametric of depth 1.

    Args:
        mutation_proportion: Mutation proportion parameter of the model.
        collision_probability: Collision probability parameter of the model.
        s1: Character states of the first leaf.
        s2: Character states of the second leaf.
        missing_state_indicator: Missing state indicator.

    Returns:
        The corrected distance.
    """
    expected_ternary_hamming_distance_given_height = partial(
        crispr_cas9_expected_ternary_hamming_distance,
        mutation_proportion=mutation_proportion,
        collision_probability=collision_probability,
    )
    observed_ternary_hamming_distance = ternary_hamming_distance(
        s1=s1, s2=s2, missing_state_indicator=missing_state_indicator
    )
    height = inverse(
        f=expected_ternary_hamming_distance_given_height,
        y=observed_ternary_hamming_distance,
        lower=0,
        upper=1,
    )
    estimated_tree_distance = 2 * height
    return estimated_tree_distance


def crispr_cas9_default_mutation_proportion_estimator(
    character_matrix: pd.DataFrame,
) -> float:
    """
    Estimate mutation proportion as #alleles>0/#alleles>=0.

    Assumes that the missing data indicator is negative, and that all mutated
    states are positive, for simplicity.

    Args:
        character_matrix: The character matrix.
    Returns:
        The estimated mutation proportion.
    """
    num_positive = (character_matrix > 0).sum().sum()
    num_non_negative = (character_matrix >= 0).sum().sum()
    if num_non_negative == 0:
        # Everything is missing, avoid division by zero.
        mutation_proportion = 0.0
    else:
        mutation_proportion = num_positive / num_non_negative
    return mutation_proportion


def crispr_cas9_hardcoded_mutation_proportion_estimator(
    character_matrix: pd.DataFrame,
    mutation_proportion: float,
) -> float:
    """
    Hardcode the mutation proportion.

    Args:
        character_matrix: The character matrix (which will be ignored).
        mutation_proportion: The mutation proportion to hardcode.
    Returns:
        The hardcoded `mutation_proportion`.
    """
    return mutation_proportion


def crispr_cas9_default_collision_probability_estimator(
    character_matrix: pd.DataFrame,
) -> float:
    """
    Estimate collision probability by estimating individual state probabilities.

    The estimated collision probability is:
    q = sum q_i^2
    where q_i is the estimated collision probability for state i using
    #alleles=i / #alleles>0 (i.e. the number of times state i appears
    in the character matrix, divided by the total number of mutations).

    Assumes that the missing data indicator is negative, and that all mutated
    states are positive, for simplicity.

    Args:
        character_matrix: The character matrix.
    Returns:
        Estimate of the collision probability.
    """
    # First we compute the number of times that each state appears, e.g.
    # {'state1': 10, 'state2': 123, ...}
    state_count_dict = dict(
        character_matrix[character_matrix > 0]
        .stack()
        .reset_index(drop=True)
        .value_counts()
    )
    # Now we just extract the counts, e.g. [10, 123, ...]
    state_counts = np.array(list(state_count_dict.values()))
    if state_counts.sum() == 0:
        # To avoid division by 0, we estimate a collision probability of 0.
        collision_probability = 0.0
    else:
        collision_probability = sum((state_counts / state_counts.sum()) ** 2)
    return collision_probability


def crispr_cas9_hardcoded_collision_probability_estimator(
    character_matrix: pd.DataFrame,
    collision_probability: float,
) -> float:
    """
    Hardcode the collision probability.

    Args:
        character_matrix: The character matrix (which will be ignored).
        collision_probability: The collision probability to hardcode.
    Returns:
        The hardcoded `collision_probability`.
    """
    return collision_probability


class Crispr_cas9_corrected_hamming_distance_wrapper:
    """
    Dissimilarity function to inject into the distance solver.

    This is just a wrapper around `crispr_cas9_corrected_hamming_distance`
    that makes it conform to the DistanceSolver API.

    E.g. the `weights` parameter is required by the DistanceSolver API,
    which is why it is part of the argument list, but it is not used at all.
    """

    def __init__(
        self,
        mutation_proportion: float,
        collision_probability: float,
    ):
        self._mutation_proportion = mutation_proportion
        self._collision_probability = collision_probability

    def __call__(
        self,
        s1: List[int],
        s2: List[int],
        missing_state_indicator: int,
        weights: Optional[Dict[int, Dict[int, float]]],
    ) -> float:
        return crispr_cas9_corrected_hamming_distance(
            mutation_proportion=self._mutation_proportion,
            collision_probability=self._collision_probability,
            s1=s1,
            s2=s2,
            missing_state_indicator=missing_state_indicator,
        )


class Crispr_cas9_corrected_ternary_hamming_distance_wrapper:
    """
    Dissimilarity function to inject into the distance solver.

    This is just a wrapper around
    `_crispr_cas9_corrected_ternary_hamming_distance_wrapper` that makes it
    conform to the DistanceSolver API.

    E.g. the `weights` parameter is required by the DistanceSolver API,
    which is why it is part of the argument list, but it is not used at all.
    """

    def __init__(
        self,
        mutation_proportion: float,
        collision_probability: float,
    ):
        self._mutation_proportion = mutation_proportion
        self._collision_probability = collision_probability

    def __call__(
        self,
        s1: List[int],
        s2: List[int],
        missing_state_indicator: int,
        weights: Optional[Dict[int, Dict[int, float]]],
    ) -> float:
        return crispr_cas9_corrected_ternary_hamming_distance(
            mutation_proportion=self._mutation_proportion,
            collision_probability=self._collision_probability,
            s1=s1,
            s2=s2,
            missing_state_indicator=missing_state_indicator,
        )


class CRISPRCas9DistanceCorrectionSolver(CassiopeiaSolver):
    """
    Apply a distance solver to corrected CRISPR-Cas9 distances.

    Here, "corrected distances" are estimates of true pairwise tree distance,
    as opposed to, say, Hamming distances.

    In the distance correction scheme, we first estimate the two parameters of
    the CRISPR-Cas9 model (the mutation rate and the collision probability),
    and then use these values to compute corrected distances from the character
    states of each leaf.

    Assumes that the missing data indicator is negative, and that all mutated
    states are positive, for simplicity.

    Technical note: The reason why we use a "distance_corrector_name" instead
    of just injecting the distance_corrector_function directly is because the
    latter approach fails due to Numba compilation issues.

    Args:
        mutation_proportion_estimator: The mutation proportion estimator.
            Default is `crispr_cas9_default_mutation_proportion_estimator`,
            which uses the simple estimate #alleles>0 / #alleles>=0.
            The mutation proportion can be set up front (i.e. hardcoded)
            with the `crispr_cas9_hardcoded_mutation_proportion_estimator`.
        collision_probability_estimator: The collision probability estimator.
            Default is `crispr_cas9_default_collision_probability_estimator`,
            which uses the simple estimate sum q_i^2, where each q_i is
            estimated as q_i = #alleles=i / #alleles>0. The collision
            probability can be set up front (i.e. hardcoded) with the
            `crispr_cas9_hardcoded_collision_probability_estimator`.
        distance_corrector_name: The name of the function that computes
            corrected distances given the mutation rate, collision probability,
            and character states of the two leaves. Allowed options:
            `crispr_cas9_corrected_ternary_hamming_distance` (default) and
            `crispr_cas9_corrected_hamming_distance`. Generally, the ternary
            Hamming distance tends to perform better.
        distance_solver: The distance solver used to reconstruct the tree
            topology using the corrected distances. By default, NJ with rooting
            is used, but other distance solvers such as UPGMA can be used too.
    """

    def __init__(
        self,
        mutation_proportion_estimator: MutationProportionEstimatorType = crispr_cas9_default_mutation_proportion_estimator,  # noqa
        collision_probability_estimator: CollisionProbabilityEstimatorType = crispr_cas9_default_collision_probability_estimator,  # noqa
        distance_corrector_name: str = "crispr_cas9_corrected_ternary_hamming_distance",  # noqa
        distance_solver: DistanceSolver = NeighborJoiningSolver(add_root=True),
    ):
        self._mutation_proportion_estimator = mutation_proportion_estimator
        self._collision_probability_estimator = collision_probability_estimator
        self._distance_corrector_name = distance_corrector_name
        self._distance_solver = distance_solver

    def solve(self, tree: CassiopeiaTree):
        if tree.missing_state_indicator >= 0:
            raise ValueError(
                "For simplicity, it is required that the missing state "
                "indicator is negative. For your data, it is: "
                f"{tree.character_matrix.missing_state_indicator}"
            )
        mutation_proportion = self._mutation_proportion_estimator(
            character_matrix=tree.character_matrix,
        )
        collision_probability = self._collision_probability_estimator(
            character_matrix=tree.character_matrix,
        )

        if (
            self._distance_corrector_name
            == "crispr_cas9_corrected_ternary_hamming_distance"
        ):
            corrected_distance_function = (
                Crispr_cas9_corrected_ternary_hamming_distance_wrapper(
                    mutation_proportion=mutation_proportion,
                    collision_probability=collision_probability,
                )
            )
        elif (
            self._distance_corrector_name
            == "crispr_cas9_corrected_hamming_distance"
        ):
            corrected_distance_function = (
                Crispr_cas9_corrected_hamming_distance_wrapper(
                    mutation_proportion=mutation_proportion,
                    collision_probability=collision_probability,
                )
            )
        else:
            raise ValueError(
                "Unknown distance_corrector_name: "
                f"'{self._distance_corrector_name}'"
            )

        distance_solver = self._distance_solver
        distance_solver.dissimilarity_function = corrected_distance_function
        distance_solver.solve(tree)
