"""Dissimilarity computation utilities for tree solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cassiopeia.solver import dissimilarity_functions as _dissimilarity_functions

if TYPE_CHECKING:
    from cassiopeia.data import CassiopeiaTree
    from treedata import TreeData


def _resolve_dissimilarity(
    dissimilarity: str | Callable | None,
) -> Callable | None:
    """Return a callable dissimilarity function given a string name or callable.

    Args:
        dissimilarity: A callable, a string name of a function in
            :mod:`cassiopeia.solver.dissimilarity_functions`, or ``None``.

    Returns:
        The resolved callable, or ``None`` if *dissimilarity* was ``None``.

    Raises:
        ValueError: If *dissimilarity* is a string not found in the module.
        TypeError: If *dissimilarity* is not a string, callable, or ``None``.
    """
    if dissimilarity is None or callable(dissimilarity):
        return dissimilarity
    if isinstance(dissimilarity, str):
        fn = getattr(_dissimilarity_functions, dissimilarity, None)
        if fn is None:
            available = sorted(
                name for name in dir(_dissimilarity_functions)
                if callable(getattr(_dissimilarity_functions, name)) and not name.startswith("_")
            )
            raise ValueError(
                f"Unknown dissimilarity function {dissimilarity!r}. "
                f"Available: {available}"
            )
        return fn
    raise TypeError(
        f"dissimilarity must be a string, callable, or None, "
        f"got {type(dissimilarity).__name__!r}"
    )


def _compute_from_chars(
    characters: pd.DataFrame,
    missing: int,
    priors,
    fn: Callable,
    prior_transformation: str,
    threads: int,
) -> pd.DataFrame:
    """Compute a pairwise dissimilarity map from a character DataFrame.

    Creates a temporary :class:`~cassiopeia.data.CassiopeiaTree` to route
    through the proper Numba typed-dict infrastructure required by dissimilarity
    functions.

    Args:
        characters: Character matrix (samples × characters).
        missing: Missing state indicator value.
        priors: Character priors dict, or ``None``.
        fn: Dissimilarity function.
        prior_transformation: Transformation applied to priors.
        threads: Threads for parallel computation.

    Returns:
        Symmetric pairwise distance ``pd.DataFrame``.
    """
    from cassiopeia.data import CassiopeiaTree
    temp = CassiopeiaTree(
        character_matrix=characters,
        missing_state_indicator=missing,
        priors=priors,
    )
    temp.compute_dissimilarity_map(fn, prior_transformation, threads=threads)
    return temp.get_dissimilarity_map()


def _get_distances(
    tdata: CassiopeiaTree | TreeData,
    dist_key: str | None,
    dissimilarity_fn: Callable | None,
    characters: pd.DataFrame | None,
    prior_transformation: str,
    threads: int,
) -> pd.DataFrame:
    """Return a symmetric n×n distance ``pd.DataFrame`` for all observations.

    When *characters* is provided, distances are always computed from those
    characters (used for augmented matrices, e.g. with a synthetic outgroup
    leaf, or for TreeData without precomputed distances).

    When *characters* is ``None``:
    - **TreeData**: uses ``tdata.obsp[dist_key]``.
    - **CassiopeiaTree**: returns the cached dissimilarity map if present,
      otherwise computes from ``tdata.character_matrix`` and caches the result.

    Args:
        tdata: CassiopeiaTree or TreeData.
        dist_key: ``obsp`` key for precomputed distances (TreeData only, used
            when *characters* is ``None``).
        dissimilarity_fn: Dissimilarity function; required when computing.
        characters: Character matrix to compute distances from.  ``None``
            defers to *dist_key* or the cached map.
        prior_transformation: Prior weight transformation name.
        threads: Threads for parallel computation.
    """
    from treedata import TreeData

    if isinstance(tdata, TreeData):
        if characters is None:
            if dist_key is None:
                raise ValueError(
                    "TreeData: provide dist_key or characters."
                )
            leaf_names = list(tdata.obs_names)
            return pd.DataFrame(
                np.array(tdata.obsp[dist_key], dtype=np.float64),
                index=leaf_names,
                columns=leaf_names,
            )
        # Compute from provided characters
        if dissimilarity_fn is None:
            from cassiopeia.mixins import DistanceSolverError
            raise DistanceSolverError(
                "Please provide a dissimilarity_function or a precomputed "
                "dissimilarity map."
            )
        missing = tdata.uns.get("missing_state_indicator", -1)
        priors = tdata.uns.get("priors", None)
        return _compute_from_chars(characters, missing, priors, dissimilarity_fn, prior_transformation, threads)

    else:  # CassiopeiaTree
        if characters is not None:
            # Augmented or layer-specific characters — compute fresh (no cache update)
            if dissimilarity_fn is None:
                from cassiopeia.mixins import DistanceSolverError
                raise DistanceSolverError(
                    "Please provide a dissimilarity_function or a precomputed "
                    "dissimilarity map."
                )
            return _compute_from_chars(
                characters,
                tdata.missing_state_indicator,
                tdata.priors,
                dissimilarity_fn,
                prior_transformation,
                threads,
            )
        # Standard path: use cached map or compute and cache on tdata
        if tdata.get_dissimilarity_map() is not None:
            return tdata.get_dissimilarity_map()
        if dissimilarity_fn is None:
            from cassiopeia.mixins import DistanceSolverError
            raise DistanceSolverError(
                "Please provide a dissimilarity_function or a precomputed "
                "dissimilarity map."
            )
        tdata.compute_dissimilarity_map(dissimilarity_fn, prior_transformation, threads=threads)
        return tdata.get_dissimilarity_map()


def dissimilarity(
    tdata: CassiopeiaTree | TreeData,
    method: str | Callable = "weighted_hamming_distance",
    characters_key: str | None = None,
    key_added: str = "distances",
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> None:
    """Compute pairwise dissimilarities and store the result in *tdata*.

    For :class:`~cassiopeia.data.CassiopeiaTree`: calls
    ``tdata.compute_dissimilarity_map()`` which populates the internal
    dissimilarity map (accessible via ``tdata.get_dissimilarity_map()``).

    For :class:`~treedata.TreeData`: computes the map from
    ``tdata.obsm[characters_key or 'characters']`` and stores the result as a
    dense ``numpy`` array in ``tdata.obsp[key_added]``.

    Args:
        tdata: CassiopeiaTree or TreeData to modify in-place.
        method: Dissimilarity function.  Accepts a callable or a string name
            of a function in :mod:`cassiopeia.solver.dissimilarity_functions`
            (e.g. ``'weighted_hamming_distance'``, ``'hamming_distance'``).
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm``
            key (TreeData, default ``'characters'``).
        key_added: ``obsp`` key under which the result is stored (TreeData
            only).  Ignored for CassiopeiaTree.
        prior_transformation: Transformation applied to priors when computing
            dissimilarity weights.
        threads: Threads for parallel computation.
    """
    from treedata import TreeData

    fn = _resolve_dissimilarity(method)

    if isinstance(tdata, TreeData):
        from cassiopeia.solver import solver_utilities
        chars = solver_utilities._get_characters(tdata, characters_key)
        if chars is None:
            raise ValueError(
                "TreeData has no character matrix; store characters in "
                f"obsm[{characters_key or 'characters'!r}]."
            )
        missing = tdata.uns.get("missing_state_indicator", -1)
        priors = tdata.uns.get("priors", None)
        result = _compute_from_chars(chars, missing, priors, fn, prior_transformation, threads)
        tdata.obsp[key_added] = result.to_numpy()
    else:
        tdata.compute_dissimilarity_map(fn, prior_transformation, characters_key, threads=threads)
