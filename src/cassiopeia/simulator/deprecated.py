"""Deprecated OO tree simulator classes.

Thin wrappers around the functional implementations in
:mod:`cassiopeia.simulator.topology` emit a :class:`DeprecationWarning` and
delegate all work to the underlying functions.

Removed classes raise :class:`NotImplementedError` and direct users to install
cassiopeia v2 for the legacy implementation.
"""

import warnings
from collections.abc import Callable

import networkx as nx
import numpy as np

from cassiopeia.simulator.topology import birth_death_process, complete_binary


class CompleteBinarySimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.complete_binary` instead."""

    def __init__(self, num_cells: int | None = None, depth: int | None = None):
        warnings.warn(
            "CompleteBinarySimulator is deprecated and will be removed in a future release. "
            "Use complete_binary() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._num_cells = num_cells
        self._depth = depth

    def simulate_tree(self, tree_key: str = "tree"):
        return complete_binary(num_cells=self._num_cells, depth=self._depth, tree_key=tree_key)


class BirthDeathFitnessSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.birth_death_process` instead."""

    def __init__(
        self,
        birth_waiting_distribution: Callable[[float], float],
        initial_birth_scale: float,
        death_waiting_distribution: Callable[[], float] = lambda: np.inf,
        mutation_distribution: Callable[[], int] | None = None,
        fitness_distribution: Callable[[], float] | None = None,
        fitness_base: float = np.e,
        num_extant: int | None = None,
        experiment_time: float | None = None,
        collapse_unifurcations: bool = True,
        random_seed: int | None = None,
        initial_tree: nx.DiGraph | None = None,
    ):
        warnings.warn(
            "BirthDeathFitnessSimulator is deprecated and will be removed in a future release. "
            "Use birth_death_process() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._birth_waiting_distribution = birth_waiting_distribution
        self._initial_birth_scale = initial_birth_scale
        self._death_waiting_distribution = death_waiting_distribution
        self._mutation_distribution = mutation_distribution
        self._fitness_distribution = fitness_distribution
        self._fitness_base = fitness_base
        self._num_extant = num_extant
        self._experiment_time = experiment_time
        self._collapse_unifurcations = collapse_unifurcations
        self._random_seed = random_seed
        self._initial_tree = initial_tree

    def simulate_tree(self, tree_key: str = "tree"):
        return birth_death_process(
            birth_waiting_distribution=self._birth_waiting_distribution,
            initial_birth_scale=self._initial_birth_scale,
            death_waiting_distribution=self._death_waiting_distribution,
            mutation_distribution=self._mutation_distribution,
            fitness_distribution=self._fitness_distribution,
            fitness_base=self._fitness_base,
            num_extant=self._num_extant,
            experiment_time=self._experiment_time,
            collapse_unifurcations=self._collapse_unifurcations,
            random_seed=self._random_seed,
            initial_tree=self._initial_tree,
            tree_key=tree_key,
        )


class ecDNABirthDeathSimulator:
    """Removed simulator for extrachromosomal DNA birth-death processes."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ecDNABirthDeathSimulator was removed in v3.0.0. "
            "Install cassiopeia v2 to use this simulator: "
            "pip install 'cassiopeia-lineage<3.0.0'"
        )


class Cas9LineageTracingDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.stochastic_tracing` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Cas9LineageTracingDataSimulator is deprecated and will be removed in a future release. "
            "Use stochastic_tracing() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        raise NotImplementedError(
            "Cas9LineageTracingDataSimulator.overlay_data() requires CassiopeiaTree which is "
            "deprecated. Use cassiopeia.simulator.stochastic_tracing() with TreeData instead."
        )


class SequentialLineageTracingDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.stochastic_tracing` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SequentialLineageTracingDataSimulator is deprecated and will be removed in a future "
            "release. Use stochastic_tracing() with initiation_rate set instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        raise NotImplementedError(
            "SequentialLineageTracingDataSimulator.overlay_data() requires CassiopeiaTree which is "
            "deprecated. Use cassiopeia.simulator.stochastic_tracing() with TreeData instead."
        )


class LeafSubsampler:
    """Deprecated abstract base. Use :func:`cassiopeia.simulator.sample_uniform`,
    :func:`cassiopeia.simulator.sample_spatial`, or
    :func:`cassiopeia.simulator.sample_supercellular` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LeafSubsampler is deprecated and will be removed in a future release. "
            "Use sample_uniform(), sample_spatial(), or sample_supercellular() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def subsample_leaves(self, tree, **kwargs):
        raise NotImplementedError(
            "LeafSubsampler.subsample_leaves() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.sample_uniform(), sample_spatial(), or "
            "sample_supercellular() with TreeData instead."
        )


class UniformLeafSubsampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_uniform` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "UniformLeafSubsampler is deprecated and will be removed in a future release. "
            "Use sample_uniform() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def subsample_leaves(self, tree, **kwargs):
        raise NotImplementedError(
            "UniformLeafSubsampler.subsample_leaves() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.sample_uniform() with TreeData instead."
        )


class SpatialLeafSubsampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_spatial` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SpatialLeafSubsampler is deprecated and will be removed in a future release. "
            "Use sample_spatial() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def subsample_leaves(self, tree, **kwargs):
        raise NotImplementedError(
            "SpatialLeafSubsampler.subsample_leaves() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.sample_spatial() with TreeData instead."
        )


class SupercellularSampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_supercellular` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SupercellularSampler is deprecated and will be removed in a future release. "
            "Use sample_supercellular() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def subsample_leaves(self, tree, **kwargs):
        raise NotImplementedError(
            "SupercellularSampler.subsample_leaves() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.sample_supercellular() with TreeData instead."
        )


class BrownianSpatialDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.brownian_spatial` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BrownianSpatialDataSimulator is deprecated and will be removed in a future release. "
            "Use brownian_spatial() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        raise NotImplementedError(
            "BrownianSpatialDataSimulator.overlay_data() requires CassiopeiaTree which is "
            "deprecated. Use cassiopeia.simulator.brownian_spatial() with TreeData instead."
        )


class ClonalSpatialDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.clonal_spatial` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ClonalSpatialDataSimulator is deprecated and will be removed in a future release. "
            "Use clonal_spatial() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        raise NotImplementedError(
            "ClonalSpatialDataSimulator.overlay_data() requires CassiopeiaTree which is "
            "deprecated. Use cassiopeia.simulator.clonal_spatial() with TreeData instead."
        )


class TreeSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.birth_death_process`,
    :func:`cassiopeia.simulator.complete_binary`, or
    :func:`cassiopeia.simulator.simple_fit_subclone` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TreeSimulator is deprecated and will be removed in a future release. "
            "Use birth_death_process(), complete_binary(), or simple_fit_subclone() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def simulate_tree(self):
        raise NotImplementedError(
            "TreeSimulator.simulate_tree() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.birth_death_process(), complete_binary(), or "
            "simple_fit_subclone() with TreeData instead."
        )


class LineageTracingDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.stochastic_tracing` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LineageTracingDataSimulator is deprecated and will be removed in a future release. "
            "Use stochastic_tracing() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        raise NotImplementedError(
            "LineageTracingDataSimulator.overlay_data() requires CassiopeiaTree which is deprecated. "
            "Use cassiopeia.simulator.stochastic_tracing() with TreeData instead."
        )


class SimpleFitSubcloneSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.simple_fit_subclone` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SimpleFitSubcloneSimulator is deprecated and will be removed in a future release. "
            "Use simple_fit_subclone() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def simulate_tree(self):
        raise NotImplementedError(
            "SimpleFitSubcloneSimulator.simulate_tree() requires CassiopeiaTree which is "
            "deprecated. Use cassiopeia.simulator.simple_fit_subclone() with TreeData instead."
        )
