import abc
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from cassiopeia.data import CassiopeiaTree

# Enable lazy trees somehow? E.g. with Proxy Value pattern? How about CassiopeiaTreeProxy subclassing CassiopeiaTree? __init__ on the superclass is only called on attempt to __getattr__. Also: conditioning beyond the leaves is allowed!
Tree = CassiopeiaTree
Trees = TypeVar("Trees", bound=List[Tree])


class QuantizationScheme(abc.ABC):
    @abc.abstractmethod
    def construct_grid(self, trees: Trees):
        """
        Allows learning the grid from the trees.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantize(self, t: float) -> float:
        """
        Maps t to the closest grid point according to this scheme.
        """
        raise NotImplementedError


class NoQuantizationScheme(QuantizationScheme):
    def construct_grid(self, trees: Trees):
        pass

    def quantize(self, t: float) -> float:
        return t


class GeometricQuantizationScheme(QuantizationScheme):
    def __init__(
        self,
        center: float,
        step_size: float,
        n_steps: int,
    ):
        self._center = center
        self._step_size = step_size
        self._n_steps = n_steps
        self._grid = np.array(
            [
                center * (1.0 + step_size) ** i
                for i in range(-n_steps, n_steps + 1, 1)
            ]
        )

    def construct_grid(self):
        pass

    def quantize(self, t: float) -> float:
        """
        Closest grid point in **log**-space.
        """
        if t <= self._grid[0]:
            return self._grid[0]
        id = np.argmin(np.abs(np.log(self._grid) - np.log(t)))
        return self._grid[id]

    def get_grid(self) -> np.array:
        return self._grid.copy()


class TransitionModel(abc.ABC):
    @abc.abstractmethod
    def transition_probability_matrix(self, t: float) -> np.array:
        raise NotImplementedError

    @abc.abstractmethod
    def root_prior(self) -> np.array:
        raise NotImplementedError


class MarkovModel(TransitionModel):
    def __init__(
        self,
        rate_matrix: np.array,
        root_prior: np.array,
    ):
        self._rate_matrix = rate_matrix
        stationary_distribution = self._solve_stationary_distribution(
            rate_matrix
        )
        self._stationary_distribution = stationary_distribution
        self._S = self._symmetrize_rate_matrix(
            rate_matrix, stationary_distribution
        )
        self._P1, self._P2 = self._diagonal_stationary_matrix(
            stationary_distribution
        )
        self._D, self._U = np.linalg.eigh(self._S)
        self._root_prior = root_prior

    def transition_probability_matrix(self, t: float) -> np.array:
        exp_D = np.diag(np.exp(t * self._D))
        exp_S = np.dot(np.dot(self._U, exp_D), self._U.transpose())
        exp_R = np.dot(np.dot(self._P2, exp_S), self._P1)
        return exp_R

    def root_prior(self) -> np.array:
        return self._root_prior

    @staticmethod
    def _symmetrize_rate_matrix(rate_matrix, stationary_distribution):
        n = rate_matrix.shape[0]
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (
                    rate_matrix[i, j]
                    * np.sqrt(stationary_distribution[i])
                    / np.sqrt(stationary_distribution[j])
                )
        return S

    @staticmethod
    def _diagonal_stationary_matrix(stationary_distribution):
        P1 = np.diag(np.sqrt(stationary_distribution))
        P2 = np.diag(np.sqrt(1.0 / stationary_distribution))
        return P1, P2

    @staticmethod
    def _solve_stationary_distribution(rate_matrix):
        eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        eigvals = np.abs(eigvals)
        index = np.argmin(eigvals)
        stationary_distribution = eigvecs[:, index]
        stationary_distribution = stationary_distribution / sum(
            stationary_distribution
        )
        return stationary_distribution


class QuantizedTransitionModel(TransitionModel):
    def __init__(
        self,
        wrapped_transition_model: TransitionModel,
        quantization_scheme: QuantizationScheme,
    ):
        self._wrapped_transition_model = wrapped_transition_model
        self._quantization_scheme = quantization_scheme
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def transition_probability_matrix(self, t: float) -> np.array:
        return self._cached_transition_probability_matrix(
            self._quantization_scheme.quantize(t)
        )

    def root_prior(self) -> np.array:
        return self._wrapped_transition_model.root_prior()

    def _cached_transition_probability_matrix(self, t: float) -> np.array:
        if t in self._cache:
            self._cache_hits += 1
            return self._cache[t]
        else:
            self._cache_misses += 1
            res = self._wrapped_transition_model.transition_probability_matrix(
                t
            )
            self._cache[t] = res
            return res

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses


TransitionModels = TypeVar("TransitionModels", bound=List[TransitionModel])


class TreeStatistic:
    def __init__(
        self,
        root_frequencies: np.array,
        transition_frequencies: List[Tuple[float, np.array]],
    ):
        self.root_frequencies = root_frequencies.copy()
        self.transition_frequencies = deepcopy(transition_frequencies)

    def __eq__(self, other):
        if np.any(~np.isclose(self.root_frequencies, other.root_frequencies)):
            return False
        if len(self.transition_frequencies) != len(
            other.transition_frequencies
        ):
            return False
        num_branch_lengths = len(self.transition_frequencies)
        for i in range(num_branch_lengths):
            if (
                self.transition_frequencies[i][0]
                != other.transition_frequencies[i][0]
            ):
                return False
            if np.any(
                ~np.isclose(
                    self.transition_frequencies[i][1],
                    other.transition_frequencies[i][1],
                )
            ):
                return False
        return True


class Statistics:
    def __init__(
        self,
        per_tree_statistics: bool,
        quantization_scheme: QuantizationScheme,
        num_states: int,
    ):
        self._per_tree_statistics = per_tree_statistics
        self._quantization_scheme = quantization_scheme
        self._num_states = num_states

        self._root_frequencies = defaultdict(
            lambda: np.zeros(shape=num_states, dtype=np.float)
        )
        self._transition_frequencies = defaultdict(
            lambda: defaultdict(
                lambda: np.zeros(shape=(num_states, num_states), dtype=np.float)
            )
        )

    def add_tree_statistic(self, tree_id: int, tree_statistic: TreeStatistic):
        self._root_frequencies[
            self._tree_key(tree_id)
        ] += tree_statistic.root_frequencies
        for (t, frequency_matrix) in tree_statistic.transition_frequencies:
            self._transition_frequencies[self._tree_key(tree_id)][
                self._branch_length_key(t)
            ] += frequency_matrix
        return self  # For chaining

    def __iadd__(self, other):
        if other._per_tree_statistics != self._per_tree_statistics:
            raise ValueError(
                "Cannot add a tree statistics that is per-tree with a tree "
                "statistic that is not per-tree."
            )
        for tree_key, value in other._root_frequencies.items():
            self._root_frequencies[tree_key] += value
        for (
            tree_key,
            frequency_matrices,
        ) in other._transition_frequencies.items():
            for (
                branch_length_key,
                frequency_matrix,
            ) in frequency_matrices.items():
                self._transition_frequencies[tree_key][
                    branch_length_key
                ] += frequency_matrix
        return self

    def __add__(self, other):
        res = deepcopy(self)
        res += other
        return res

    def _tree_key(self, tree_id: int) -> int:
        if self._per_tree_statistics:
            return tree_id
        else:
            return 0

    def _branch_length_key(self, t: float) -> float:
        return self._quantization_scheme.quantize(t)

    def get_statistics_for_tree(self, tree_id: int) -> TreeStatistic:
        tree_key = self._tree_key(tree_id)
        root_frequencies = self._root_frequencies[tree_key].copy()
        transition_statistics = []
        for branch_length_key, frequency_matrix in self._transition_frequencies[
            tree_key
        ].items():
            transition_statistics.append(
                (
                    branch_length_key,
                    frequency_matrix,
                )
            )
        return TreeStatistic(
            root_frequencies=root_frequencies,
            transition_frequencies=sorted(transition_statistics),
        )

    def get_statistics(self) -> List[Tuple[int, TreeStatistic]]:
        tree_ids = self._root_frequencies.keys()
        return [
            (
                tree_id,
                self.get_statistics_for_tree(tree_id),
            )
            for tree_id in sorted(tree_ids)
        ]


class EStep(abc.ABC):
    @abc.abstractmethod
    def perform_e_step(
        tree: CassiopeiaTree,
        transition_model: Optional[TransitionModel],
    ) -> TreeStatistic:
        raise NotImplementedError


class MStep(abc.ABC):
    @abc.abstractmethod
    def perform_m_step(
        stats: Statistics,
        params: Optional[TransitionModels],  # For XRATE
    ) -> TransitionModels:
        raise NotImplementedError

    @abc.abstractmethod
    def initialization(self, trees: Trees) -> TransitionModels:
        raise NotImplementedError

    @abc.abstractmethod
    def requires_per_tree_statistics(
        self,
    ) -> bool:  # To indicate what the sufficient statistics are, and crunch the data down as much as possible.
        raise NotImplementedError


class EM:
    def __init__(
        self,
        e_step: EStep,
        m_step: MStep,
        initialization: Optional[TransitionModels],
        num_em_steps: int,
        quantization_scheme: QuantizationScheme,
    ):
        self._e_step = e_step
        self._m_step = m_step
        self._initialization = initialization
        self._num_em_steps = num_em_steps
        self._quantization_scheme = quantization_scheme

    def perform_em(self, trees: Trees):
        e_step = self.e_step
        m_step = self.m_step
        initialization = self._initialization
        num_em_steps = self._num_em_steps
        quantization_scheme = self._quantization_scheme

        quantization_scheme.construct_grid(trees)

        transition_models = (
            initialization if initialization else m_step.initialization(trees)
        )

        for step in range(num_em_steps):
            # Quantization and caching scaffold
            transition_models = self._quantize_transition_models(
                transition_models, quantization_scheme
            )
            # Warm-start the expm caches; only if it's just one global model.
            if len(transition_models) == 1:
                transition_models[0].precompute()

            # E-Step
            stats = sum
            [
                Statistics(
                    m_step.requires_per_tree_statistics,
                    quantization_scheme,
                    tree_id,
                    e_step.perform_e_step(tree, transition_model),
                )
                for tree_id, (tree, transition_model) in enumerate(
                    zip(trees, cycle(transition_models))
                )
            ]  # Parallelizable! MapReduce! Carefull though: we want this is O(n) time, not O(n^2). It might require using __iadd__ rather than __add__ in the sum reduction (how can we do that in a pythonic way?)

            # M-Step
            transition_models = m_step.perform_m_step(
                stats
            )  # Stats must contain the info about each tree.

    @staticmethod
    def _quantize_transition_models(transition_models, quantization_scheme):
        return [
            QuantizedTransitionModel(transition_model, quantization_scheme)
            for transition_model in transition_models
        ]
