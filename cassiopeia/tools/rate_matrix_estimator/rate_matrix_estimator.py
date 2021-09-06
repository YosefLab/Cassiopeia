import os
import abc
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from typing import List, Optional, Tuple, TypeVar
import hashlib
import pickle

import numpy as np

from cassiopeia.data import CassiopeiaTree

# Enable lazy trees somehow? E.g. with Proxy Value pattern? How about CassiopeiaTreeProxy subclassing CassiopeiaTree? __init__ on the superclass is only called on attempt to __getattr__. Also: conditioning beyond the leaves is allowed!
Tree = CassiopeiaTree
Trees = TypeVar("Trees", bound=List[Tree])
# Trees = List[Tree]

f(t: Trees) -> Trees:


class LongHashable(abc.ABC):
    """
    An object composed of basic objects, or long hashables,
    allowing for recursive long hashing (the hash has length
    512)
    """
    def long_hash(self):
        keys = sorted([k for k in vars(self) if k.startswith('_') and not k.endswith('_')])
        sub_hashes = [self._get_sub_hash(key) for key in keys]
        return hashlib.sha512(''.join(sub_hashes).encode("utf-8")).hexdigest()

    def _get_sub_hash(self, key):
        val = vars(self)[key]
        if isinstance(val, LongHashable):
            return val.long_hash()
        else:
            if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or (val is None):
                return hashlib.sha512((str(type(val).__name__) + str(val)).encode("utf-8")).hexdigest()
            else:
                raise ValueError(f"Do not know how to long hash key {key} with value {val}.")


class QuantizationScheme(abc.ABC, LongHashable):
    @abc.abstractmethod
    def quantize(self, t: float) -> float:
        """
        Maps t to the closest grid point according to this scheme.
        """
        raise NotImplementedError


class NoQuantizationScheme(QuantizationScheme):
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

        self._grid_ = np.array(
            [
                center * (1.0 + step_size) ** i
                for i in range(-n_steps, n_steps + 1, 1)
            ]
        )

    def quantize(self, t: float) -> float:
        """
        Closest grid point in **log**-space.
        """
        if t <= self._grid_[0]:
            return self._grid_[0]
        id = np.argmin(np.abs(np.log(self._grid_) - np.log(t)))
        return self._grid_[id]

    def get_grid(self) -> np.array:
        return self._grid_.copy()


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
        self._S_ = self._symmetrize_rate_matrix(
            rate_matrix, stationary_distribution
        )
        self._P1_, self._P2_ = self._diagonal_stationary_matrix(
            stationary_distribution
        )
        self._D_, self._U_ = np.linalg.eigh(self._S_)
        self._root_prior = root_prior

    def transition_probability_matrix(self, t: float) -> np.array:
        exp_D = np.diag(np.exp(t * self._D_))
        exp_S = np.dot(np.dot(self._U_, exp_D), self._U_.transpose())
        exp_R = np.dot(np.dot(self._P2_, exp_S), self._P1_)
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
        self._cache_ = {}
        self._cache_hits_ = 0
        self._cache_misses_ = 0

    def transition_probability_matrix(self, t: float) -> np.array:
        return self._cached_transition_probability_matrix(
            self._quantization_scheme.quantize(t)
        )

    def root_prior(self) -> np.array:
        return self._wrapped_transition_model.root_prior()

    def _cached_transition_probability_matrix(self, t: float) -> np.array:
        if t in self._cache_:
            self._cache_hits_ += 1
            return self._cache_[t]
        else:
            self._cache_misses_ += 1
            res = self._wrapped_transition_model.transition_probability_matrix(
                t
            )
            self._cache_[t] = res
            return res

    @property
    def cache_hits(self) -> int:
        return self._cache_hits_

    @property
    def cache_misses(self) -> int:
        return self._cache_misses_


TransitionModels = TypeVar("TransitionModels", bound=List[TransitionModel])


class TreeStatistic:
    def __init__(
        self,
        root_frequencies: np.array,
        transition_frequencies: List[Tuple[float, np.array]],
    ):
        self._root_frequencies = root_frequencies.copy()
        self._transition_frequencies = deepcopy(transition_frequencies)

    def __eq__(self, other):
        if np.any(~np.isclose(self._root_frequencies, other._root_frequencies)):
            return False
        if len(self._transition_frequencies) != len(
            other._transition_frequencies
        ):
            return False
        num_branch_lengths = len(self._transition_frequencies)
        for i in range(num_branch_lengths):
            if (
                self._transition_frequencies[i][0]
                != other._transition_frequencies[i][0]
            ):
                return False
            if np.any(
                ~np.isclose(
                    self._transition_frequencies[i][1],
                    other._transition_frequencies[i][1],
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


class EStep(abc.ABC, LongHashable):
    @abc.abstractmethod
    def perform_e_step(
        tree: CassiopeiaTree,
        transition_model: Optional[TransitionModel],
    ) -> TreeStatistic:
        raise NotImplementedError


class MStep(abc.ABC, LongHashable):
    @abc.abstractmethod
    def perform_m_step(
        stats: Statistics,
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


class Lazy(LongHashable):
    def __init__(self, wrapped_long_hashable: LongHashable):
        long_hash = wrapped_long_hashable.long_hash()
        filename = os.path.join(CACHE_DIR, long_hash + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(wrapped_long_hashable, f)
        self._filename = filename

    def load(self):
        with open(self._filename, 'rb') as f:
            return pickle.load(f)


class EMWithCaching(LongHashable):
    """
    In charge of loading lazy objects when needed, or otherwise returning
    cached results.
    """
    def __init__(
        self,
        e_step: EStep,
        m_step: MStep,
        maybe_lazy_initialization: Optional[Lazy[List[Lazy[TransitionModel]]]],  # E.g. None the first time to get JTT-IPW, then JTT-IPW
        num_em_steps: int,
        quantization_scheme: QuantizationScheme,
    ):
        self._e_step = e_step
        self._m_step = m_step
        self._maybe_lazy_initialization = maybe_lazy_initialization
        self._num_em_steps = num_em_steps
        self._quantization_scheme = quantization_scheme

    # @cached
    def perform_em(
        self,
        lazy_list_of_lazy_trees: Lazy[List[Lazy[Tree]]]
    ):
        lazy_transition_models = self.get_initialization(lazy_list_of_lazy_trees)

        for _ in range(self._num_em_steps):
            # Quantization and in-memory caching scaffold
            transition_models = self._quantize_transition_models(
                transition_models, self._quantization_scheme
            )
            # Warm-start the expm caches; only if it's just one global model.
            if len(transition_models) == 1:
                transition_models[0].precompute()

            # E-Step
            lazy_stats = self.perform_e_step(
                lazy_list_of_lazy_trees,
                lazy_list_of_lazy_transition_models
            )

            # M-Step
            transition_models = self.perform_m_step(
                lazy_stats,
            )

    # @cached
    def get_initialization(
        self,
        lazy_list_of_lazy_trees: Lazy[List[Lazy[Tree]]],
    ):
        return self._maybe_lazy_initialization if self._maybe_lazy_initialization else self._m_step.initialization(lazy_list_of_lazy_trees)

    # @cached
    # @lazy_output
    def perform_e_step(
        self,
        lazy_list_of_lazy_trees: Lazy[List[Lazy[Tree]]],
        lazy_list_of_lazy_transition_models: Lazy[List[Lazy[TransitionModel]]],
    ) -> Statistics:
        stats = sum(
            [
                self.perform_e_step_individual(
                    tree_id,
                    lazy_tree,
                    lazy_transition_model,
                )
                for tree_id, (lazy_tree, lazy_transition_model) in enumerate(
                    zip(
                        lazy_list_of_lazy_trees.materialize(),
                        cycle(lazy_list_of_lazy_transition_models.materialize())
                    )
                )
            ]  # Parallelizable! MapReduce! Carefull though: we want this is O(n) time, not O(n^2). It might require using __iadd__ rather than __add__ in the sum reduction (how can we do that in a pythonic way?)
        )
        return stats

    # @cached
    # Note that functions called from within @cached functions will always be materialized, so no need to make their outputs lazy.
    def perform_e_step_individual(
        self,
        tree_id: int,
        lazy_tree: Lazy[Tree],
        lazy_transition_model: Lazy[TransitionModel],
    ) -> Statistics:
        stats = Statistics(
            self._m_step.requires_per_tree_statistics,
            self._quantization_scheme,
            tree_id,
            self._e_step.perform_e_step(lazy_tree.materialize(), lazy_transition_model.materialize()),
        )
        return stats

    # @cached
    # @lazy_output
    def perform_m_step(
        self,
        stats: Lazy[Statistics],
    ) -> TransitionModels:
        transition_models = self._m_step.perform_m_step(
            stats.materialize()
        )  # Stats must contain the info about each tree.
        return transition_models

    @staticmethod
    def _quantize_transition_models(transition_models, quantization_scheme):
        return [
            QuantizedTransitionModel(transition_model, quantization_scheme)
            for transition_model in transition_models
        ]
