from typing import Callable, List, Optional

import numpy as np
from cassiopeia import solver
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver.DistanceSolver import DistanceSolver


def _inverse(
    f: Callable[[float], float],
    y: float,
    lower: float,
    upper: float,
) -> float:
    """
    Invert a function.

    Finds x s.t. f(x) = y. Search for x is limited to the range [lower, upper].
    Uses 30 fixed iterations of binary search.
    """
    for iteration in range(30):
        x = (upper + lower) / 2.0
        if f(x) < y:
            lower, upper = x, upper
        else:
            lower, upper = lower, x
    return (upper + lower) / 2.0


class _CorrectedWHD:
    """
    Dissimilarity function that computes corrected distances from the WHD.

    Requires knowledge of the collision_probability and the mut_prop.
    """

    def __init__(
        self,
        collision_probability: float,
        mut_prop: float,
    ):
        self.q = collision_probability
        self.mut_rate = -np.log(1 - mut_prop)

    @staticmethod
    def _ewhd_given_h(mut_rate: float, q: float, height: float) -> float:
        r = mut_rate
        h = height
        return (
            2 * (1 - np.exp(-h * r)) ** 2 * (1 - q)
            + 2 * (1 - np.exp(-h * r)) * (np.exp(-h * r))
        ) * (np.exp(r * (h - 1)))

    @staticmethod
    def _corrected_whd(mut_rate, q, whd) -> float:
        f = lambda h: _CorrectedWHD._ewhd_given_h(mut_rate, q, h)
        height = _inverse(f, whd, 0, 1)
        return 2 * height

    def __call__(
        self,
        s1: List[int],
        s2: List[int],
        missing_state_indicator: int = -1,
    ) -> float:
        whd = solver.dissimilarity.weighted_hamming_distance(
            s1, s2, missing_state_indicator
        )
        return self._corrected_whd(
            mut_rate=self.mut_rate,
            q=self.q,
            whd=whd,
        )


def hamming_distance(
    s1: List[int], s2: List[int], missing_state_indicator=-1, weights=None
) -> float:
    """
    Scaled hamming distance, ignoring missing data.
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


class _CorrectedHD:
    """
    Dissimilarity function that computes corrected distances from the HD.

    Requires knowledge of the collision_probability and the mut_prop.
    """

    def __init__(
        self,
        collision_probability: float,
        mut_prop: float,
    ):
        self.q = collision_probability
        self.mut_rate = -np.log(1 - mut_prop)

    @staticmethod
    def _ehd_given_h(mut_rate: float, q: float, height: float) -> float:
        r = mut_rate
        h = height
        return np.exp(-r) * (
            ((1 - q) * np.exp(r * h)) + (2 * q) - ((1 + q) * np.exp(-r * h))
        )

    @staticmethod
    def _corrected_hd(mut_rate, q, hd):
        f = lambda h: _CorrectedHD._ehd_given_h(mut_rate, q, h)
        height = _inverse(f, hd, 0, 1)
        return 2 * height

    def __call__(
        self,
        s1: List[int],
        s2: List[int],
        missing_state_indicator: int = -1,
    ) -> float:
        hd = hamming_distance(
            s1, s2, missing_state_indicator=missing_state_indicator
        )
        return self._corrected_hd(
            mut_rate=self.mut_rate,
            q=self.q,
            hd=hd,
        )


class CorrectedHammingDistanceSolver:
    """
    Use distance correction when applying a solver.

    In the distance correction scheme, the (weighted) Hamming distances are
    "corrected" by inferring the true tree distance associated to said
    Hamming distance.

    The distance correction scheme requires knowledge of the collision
    probability q, defines as q := sum_j q_j^2 (where q_j is the probability
    if state j), and of the expected mutation proportion (or "mut_prop" for
    short), which is 1 - e^{-r} where "r" is the mutation rate of CRISPR-Cas9.

    If the collision_probability is not provided, then it will be estimated by
    first estimating the q_j using the simple estimator which takes the fraction
    of mutations equal to j divided by the total number of mutations.

    If the mut_prop is not provided, it will be estimated using the simple
    estimator.

    Args:
        solver: Underlying distance solver to use.
        mut_prop: Expected mutation proportion.
        collision_probability: Collision probability.
        weighting: What kind of weighting to use, if any. Only weightings
            currently provided are None (for the vanilla Hamming distance) and
            "2", which unlike the the vanilla Hamming distance gives a
            larger dissimilarity of 2 for two different mutations.
    """

    def __init__(
        self,
        solver: DistanceSolver,
        mut_prop: Optional[float] = None,
        collision_probability: Optional[float] = None,
        weighting: Optional[str] = None,
    ):
        self.solver = solver
        self.mut_prop = mut_prop
        self.collision_probability = collision_probability
        self.weighting = weighting

    def solve(self, tree: CassiopeiaTree):
        cm = tree.character_matrix

        # Estimate parameters of the model if not provided
        num_positive = (cm > 0).sum().sum()
        nun_non_negative = (cm >= 0).sum().sum()
        if self.mut_prop is None:
            self.mut_prop = num_positive / nun_non_negative
        if self.collision_probability is None:
            state_count_dict = dict(
                cm[cm > 0].stack().reset_index(drop=True).value_counts()
            )
            state_counts = np.array(list(state_count_dict.values()))
            if state_counts.sum() == 0:
                self.collision_probability = 0.0
            else:
                self.collision_probability = sum(
                    (state_counts / state_counts.sum()) ** 2
                )

        if self.weighting is None:
            distance_function = _CorrectedHD(
                collision_probability=self.collision_probability,
                mut_prop=self.mut_prop,
            )
        elif self.weighting == "2":
            distance_function = _CorrectedWHD(
                collision_probability=self.collision_probability,
                mut_prop=self.mut_prop,
            )
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
        self.solver.dissimilarity_function = distance_function
        self.solver.solve(tree)
