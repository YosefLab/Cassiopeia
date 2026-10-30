"""Deprecated 2.x simulator classes.

The classes warn with a :class:`DeprecationWarning` and delegate to the
functional implementations, operating on :class:`treedata.TreeData` instead of
``CassiopeiaTree``. Abstract base classes and ``ecDNABirthDeathSimulator`` were
removed and raise :class:`NotImplementedError`.
"""

import warnings
from collections.abc import Callable

import networkx as nx
import numpy as np
import treedata as td

from cassiopeia.simulator.expression import trajectory_expression
from cassiopeia.simulator.sampling import sample_spatial, sample_supercellular, sample_uniform
from cassiopeia.simulator.spatial import brownian_spatial, clonal_spatial
from cassiopeia.simulator.topology import birth_death_process, complete_binary, simple_fit_subclone
from cassiopeia.simulator.tracing import missing_data, stochastic_tracing


def _warn(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release. Use {replacement} "
        "instead; this class now delegates to it and works on TreeData.",
        DeprecationWarning,
        stacklevel=3,
    )


def _require_treedata(tree, method: str, replacement: str) -> td.TreeData:
    """Return *tree* if it is a TreeData, otherwise raise a pointed error."""
    if isinstance(tree, td.TreeData):
        return tree
    raise TypeError(
        f"{method} takes a TreeData in Cassiopeia 3.0, got {type(tree).__name__}. "
        f"Convert a CassiopeiaTree with tree.to_treedata(), or use {replacement} directly."
    )


def fate_tree_expression(*args, **kwargs):
    """Deprecated. Use :func:`cassiopeia.simulator.trajectory_expression` instead."""
    warnings.warn(
        "fate_tree_expression is deprecated and will be removed in a future release. "
        "Use trajectory_expression() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map renamed arguments to the new signature.
    if "lineage_tree" in kwargs:
        kwargs["tdata"] = kwargs.pop("lineage_tree")
    if "fate_tree" in kwargs:
        kwargs["trajectory"] = kwargs.pop("fate_tree")
    if "factor_key" in kwargs:
        kwargs["latent_key"] = kwargs.pop("factor_key")
    if "random_state" in kwargs:
        kwargs["random_seed"] = kwargs.pop("random_state")
    if "noise" in kwargs:
        kwargs["latent_noise"] = kwargs.pop("noise")
    # fate_tree_expression returned a new TreeData.
    kwargs.setdefault("copy", True)
    return trajectory_expression(*args, **kwargs)


class CompleteBinarySimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.complete_binary` instead."""

    def __init__(self, num_cells: int | None = None, depth: int | None = None):
        _warn("CompleteBinarySimulator", "complete_binary()")
        self._num_cells = num_cells
        self._depth = depth

    def simulate_tree(self, tree_key: str = "tree") -> td.TreeData:
        """Simulate a complete binary tree, stored in ``obst[tree_key]``."""
        return complete_binary(num_cells=self._num_cells, depth=self._depth, key_added=tree_key)


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
        _warn("BirthDeathFitnessSimulator", "birth_death_process()")
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

    def simulate_tree(self, tree_key: str = "tree") -> td.TreeData:
        """Simulate a birth-death tree, stored in ``obst[tree_key]``."""
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
            key_added=tree_key,
        )


class ecDNABirthDeathSimulator:
    """Removed. Use :func:`cassiopeia.simulator.birth_death_process` with an ``on_division`` hook."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ecDNABirthDeathSimulator is deprecated and was removed in Cassiopeia 3.0.0. "
            "Simulate ecDNA inheritance with cassiopeia.sim.birth_death_process and an "
            "on_division hook that splits ecDNA copies between daughters; see the "
            "simulate_ecDNA tutorial: "
            "https://cassiopeia-lineage.readthedocs.io/en/latest/notebooks/simulate_ecDNA.html"
        )


class Cas9LineageTracingDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.stochastic_tracing` instead.

    :meth:`overlay_data` runs :func:`~cassiopeia.simulator.stochastic_tracing` and
    then :func:`~cassiopeia.simulator.missing_data` on a TreeData. States follow
    the 3.0 conventions (``"*"`` unedited, ``"-"`` missing), so
    ``heritable_missing_data_state`` and ``stochastic_missing_data_state`` are
    ignored.
    """

    def __init__(
        self,
        number_of_cassettes: int = 10,
        size_of_cassette: int = 3,
        mutation_rate: float | list[float] = 0.01,
        state_generating_distribution: Callable[[], float] = lambda: np.random.exponential(1e-5),
        number_of_states: int = 100,
        state_priors: dict | None = None,
        heritable_silencing_rate: float = 1e-4,
        stochastic_silencing_rate: float = 1e-2,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -1,
        random_seed: int | None = None,
        collapse_sites_on_cassette: bool = True,
    ):
        _warn("Cas9LineageTracingDataSimulator", "stochastic_tracing() and missing_data()")
        self._tracing = {
            "number_of_cassettes": number_of_cassettes,
            "size_of_cassette": size_of_cassette,
            "mutation_rate": mutation_rate,
            "state_generating_distribution": state_generating_distribution,
            "number_of_states": number_of_states,
            "state_priors": state_priors,
            "random_seed": random_seed,
        }
        self._missing = {
            "heritable_rate": heritable_silencing_rate,
            "stochastic_rate": stochastic_silencing_rate,
            "collapse_sites_on_cassette": collapse_sites_on_cassette,
        }

    def overlay_data(self, tree: td.TreeData) -> None:
        """Overlay Cas9 lineage tracing data onto a TreeData, in place."""
        tdata = _require_treedata(
            tree, "Cas9LineageTracingDataSimulator.overlay_data()", "stochastic_tracing()"
        )
        stochastic_tracing(tdata, **self._tracing)
        _apply_missing_data(tdata, **self._missing)


class SequentialLineageTracingDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.stochastic_tracing` with ``initiation_rate``.

    :meth:`overlay_data` runs :func:`~cassiopeia.simulator.stochastic_tracing` in
    sequential mode and then :func:`~cassiopeia.simulator.missing_data` on a
    TreeData. States follow the 3.0 conventions (``"*"`` unedited, ``"-"``
    missing), so ``heritable_missing_data_state`` and
    ``stochastic_missing_data_state`` are ignored.
    """

    def __init__(
        self,
        number_of_cassettes: int = 10,
        size_of_cassette: int = 3,
        initiation_rate: float = 0.1,
        continuation_rate: float = 0.1,
        state_priors: dict | None = None,
        heritable_silencing_rate: float = 0,
        stochastic_silencing_rate: float = 0,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -1,
        random_seed: int | None = None,
    ):
        _warn(
            "SequentialLineageTracingDataSimulator",
            "stochastic_tracing() with initiation_rate set",
        )
        self._tracing = {
            "number_of_cassettes": number_of_cassettes,
            "size_of_cassette": size_of_cassette,
            "initiation_rate": initiation_rate,
            "continuation_rate": continuation_rate,
            "state_priors": state_priors,
            "random_seed": random_seed,
        }
        self._missing = {
            "heritable_rate": heritable_silencing_rate,
            "stochastic_rate": stochastic_silencing_rate,
        }

    def overlay_data(self, tree: td.TreeData) -> None:
        """Overlay sequential lineage tracing data onto a TreeData, in place."""
        tdata = _require_treedata(
            tree, "SequentialLineageTracingDataSimulator.overlay_data()", "stochastic_tracing()"
        )
        stochastic_tracing(tdata, **self._tracing)
        _apply_missing_data(tdata, **self._missing)


def _apply_missing_data(tdata: td.TreeData, **kwargs) -> None:
    """Run :func:`missing_data` unless it would not change anything."""
    if any(kwargs.values()):
        missing_data(tdata, **kwargs)


class LeafSubsampler:
    """Removed abstract base class."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LeafSubsampler is deprecated and will be removed in a future release. "
            "Use sample_uniform(), sample_spatial(), or sample_supercellular() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def subsample_leaves(self, tree, **kwargs):
        """Subsample leaves from the tree."""
        raise NotImplementedError(
            "LeafSubsampler is an abstract base class that was removed in Cassiopeia 3.0.0. "
            "Use cassiopeia.simulator.sample_uniform(), sample_spatial(), or "
            "sample_supercellular() instead."
        )


class UniformLeafSubsampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_uniform` instead."""

    def __init__(self, ratio: float | None = None, number_of_leaves: int | None = None):
        _warn("UniformLeafSubsampler", "sample_uniform()")
        self._ratio = ratio
        self._number_of_leaves = number_of_leaves

    def subsample_leaves(
        self, tree: td.TreeData, keep_singular_root_edge: bool = True
    ) -> td.TreeData:
        """Return a new TreeData with a uniform sample of the leaves of *tree*."""
        tdata = _require_treedata(
            tree, "UniformLeafSubsampler.subsample_leaves()", "sample_uniform()"
        )
        return sample_uniform(
            tdata,
            ratio=self._ratio,
            number_of_leaves=self._number_of_leaves,
            keep_root_edge=keep_singular_root_edge,
        )


class SpatialLeafSubsampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_spatial` instead.

    :meth:`subsample_leaves` runs :func:`~cassiopeia.simulator.sample_spatial`,
    then :func:`~cassiopeia.simulator.sample_uniform` (with ``ratio`` or
    ``number_of_leaves``) and :func:`~cassiopeia.simulator.sample_supercellular`
    (with ``merge_cells=True``). Coordinates are read from
    ``tdata.obsm[attribute_key]``.
    """

    def __init__(
        self,
        bounding_box: list[tuple] | None = None,
        space: np.ndarray | None = None,
        ratio: float | None = None,
        number_of_leaves: int | None = None,
        attribute_key: str = "spatial",
        merge_cells: bool = False,
    ):
        _warn("SpatialLeafSubsampler", "sample_spatial()")
        self._bounding_box = bounding_box
        self._space = space
        self._ratio = ratio
        self._number_of_leaves = number_of_leaves
        self._attribute_key = attribute_key
        self._merge_cells = merge_cells

    def subsample_leaves(
        self,
        tree: td.TreeData,
        keep_singular_root_edge: bool = True,
        collapse_duplicates: bool = True,
    ) -> td.TreeData:
        """Return a new TreeData with the leaves of *tree* in the region of interest."""
        tdata = _require_treedata(
            tree, "SpatialLeafSubsampler.subsample_leaves()", "sample_spatial()"
        )
        sampled = sample_spatial(
            tdata,
            bounding_box=self._bounding_box,
            space=self._space,
            spatial_key=self._attribute_key,
            keep_root_edge=keep_singular_root_edge,
        )
        if self._ratio is not None or self._number_of_leaves is not None:
            sampled = sample_uniform(
                sampled,
                ratio=self._ratio,
                number_of_leaves=self._number_of_leaves,
                keep_root_edge=keep_singular_root_edge,
            )
        if self._merge_cells:
            sampled = sample_supercellular(
                sampled,
                spatial_key=self._attribute_key,
                keep_root_edge=keep_singular_root_edge,
                collapse_duplicates=collapse_duplicates,
            )
        return sampled


class SupercellularSampler:
    """Deprecated. Use :func:`cassiopeia.simulator.sample_supercellular` instead."""

    def __init__(self, ratio: float | None = None, number_of_merges: int | None = None):
        _warn("SupercellularSampler", "sample_supercellular()")
        self._ratio = ratio
        self._number_of_merges = number_of_merges

    def subsample_leaves(
        self,
        tree: td.TreeData,
        collapse_source: str | None = None,
        collapse_duplicates: bool = True,
    ) -> td.TreeData:
        """Return a new TreeData with pairs of leaves of *tree* merged.

        ``collapse_source`` is ignored.
        """
        tdata = _require_treedata(
            tree, "SupercellularSampler.subsample_leaves()", "sample_supercellular()"
        )
        return sample_supercellular(
            tdata,
            ratio=self._ratio,
            number_of_merges=self._number_of_merges,
            collapse_duplicates=collapse_duplicates,
        )


class BrownianSpatialDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.brownian_spatial` instead."""

    def __init__(
        self,
        dim: int,
        diffusion_coefficient: float,
        scale_unit_area: bool = True,
        random_seed: int | None = None,
    ):
        _warn("BrownianSpatialDataSimulator", "brownian_spatial()")
        self._kwargs = {
            "dim": dim,
            "diffusion_coefficient": diffusion_coefficient,
            "scale_unit_area": scale_unit_area,
            "random_seed": random_seed,
        }

    def overlay_data(self, tree: td.TreeData, attribute_key: str = "spatial") -> None:
        """Overlay Brownian spatial coordinates onto a TreeData, in place."""
        tdata = _require_treedata(
            tree, "BrownianSpatialDataSimulator.overlay_data()", "brownian_spatial()"
        )
        brownian_spatial(tdata, key_added=attribute_key, **self._kwargs)


class ClonalSpatialDataSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.clonal_spatial` instead."""

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        space: np.ndarray | None = None,
        random_seed: int | None = None,
    ):
        _warn("ClonalSpatialDataSimulator", "clonal_spatial()")
        self._kwargs = {"shape": shape, "space": space, "random_seed": random_seed}

    def overlay_data(self, tree: td.TreeData, attribute_key: str = "spatial") -> None:
        """Overlay clonal spatial coordinates onto a TreeData, in place."""
        tdata = _require_treedata(
            tree, "ClonalSpatialDataSimulator.overlay_data()", "clonal_spatial()"
        )
        clonal_spatial(tdata, key_added=attribute_key, **self._kwargs)


class TreeSimulator:
    """Removed abstract base class."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TreeSimulator is deprecated and will be removed in a future release. "
            "Use birth_death_process(), complete_binary(), or simple_fit_subclone() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def simulate_tree(self):
        """Simulate a tree."""
        raise NotImplementedError(
            "TreeSimulator is an abstract base class that was removed in Cassiopeia 3.0.0. "
            "Use cassiopeia.simulator.birth_death_process(), complete_binary(), or "
            "simple_fit_subclone() instead."
        )


class LineageTracingDataSimulator:
    """Removed abstract base class."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LineageTracingDataSimulator is deprecated and will be removed in a future release. "
            "Use stochastic_tracing() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overlay_data(self, tree):
        """Overlay lineage tracing data onto the tree."""
        raise NotImplementedError(
            "LineageTracingDataSimulator is an abstract base class that was removed in "
            "Cassiopeia 3.0.0. Use cassiopeia.simulator.stochastic_tracing() instead."
        )


class SimpleFitSubcloneSimulator:
    """Deprecated. Use :func:`cassiopeia.simulator.simple_fit_subclone` instead."""

    def __init__(
        self,
        branch_length_neutral: float | Callable[[], float],
        branch_length_fit: float | Callable[[], float],
        experiment_duration: float,
        generations_until_fit_subclone: int,
    ):
        _warn("SimpleFitSubcloneSimulator", "simple_fit_subclone()")
        self._kwargs = {
            "branch_length_neutral": branch_length_neutral,
            "branch_length_fit": branch_length_fit,
            "experiment_duration": experiment_duration,
            "generations_until_fit_subclone": generations_until_fit_subclone,
        }

    def simulate_tree(self, tree_key: str = "tree") -> td.TreeData:
        """Simulate a tree with a fit subclone, stored in ``obst[tree_key]``."""
        return simple_fit_subclone(key_added=tree_key, **self._kwargs)
