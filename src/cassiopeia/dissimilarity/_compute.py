"""Optimized pairwise dissimilarity-map computation.

Moved here from ``cassiopeia.data.utilities``.  Computes the condensed pairwise
dissimilarity vector for a character matrix, JIT-compiling the per-pair metric
with ``numba`` and optionally parallelizing across processes.
"""

from __future__ import annotations

import functools
import multiprocessing
import warnings
from collections.abc import Callable
from multiprocessing import shared_memory

import numba
import numpy as np

from cassiopeia.mixins import CassiopeiaTreeWarning


@functools.cache
def _jit_metric(dissimilarity_function: Callable, ambiguous: bool) -> Callable:
    """Return (and cache) the ``numba``-compiled dissimilarity metric.

    Cached per ``(function, ambiguous)`` so the metric is compiled only once per
    process instead of on every :func:`compute_dissimilarity_map` call.
    """
    if not ambiguous:
        return numba.jit(dissimilarity_function, nopython=True)
    return numba.jit(dissimilarity_function, nopython=False, forceobj=True, parallel=True)


@functools.cache
def _jit_loop(dissimilarity_func: Callable, numbaize: bool, ambiguous: bool) -> Callable:
    """Return (and cache) the ``numba``-compiled pairwise inner loop.

    The loop calls *dissimilarity_func* on each pair.  Cached per
    ``(dissimilarity_func, numbaize, ambiguous)`` so it is compiled only once per
    process; previously a fresh closure was jitted on every call.
    """

    def _compute_dissimilarity_map(
        cm=np.array([[]]),
        batch_indices=np.array([]),
        missing_state_indicator=-1,
        nb_weights={},  # noqa: B006
    ):
        batch_results = np.zeros(len(batch_indices), dtype=np.float64)
        k = 0

        n = cm.shape[0]
        b = 1 - 2 * n
        for index in batch_indices:
            i = int(np.floor((-b - np.sqrt(b**2 - 8 * index)) / 2))
            j = int(index + i * (b + i + 2) / 2 + 1)
            s1 = cm[i, :]
            s2 = cm[j, :]
            batch_results[k] = dissimilarity_func(s1, s2, missing_state_indicator, nb_weights)
            k += 1
        return batch_indices, batch_results

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=numba.NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=numba.NumbaWarning)

        if not ambiguous:
            return numba.jit(_compute_dissimilarity_map, nopython=numbaize)
        return numba.jit(
            _compute_dissimilarity_map,
            nopython=False,
            forceobj=True,
            parallel=True,
        )


def compute_dissimilarity_map(
    cm: np.ndarray,
    C: int,
    dissimilarity_function: Callable,
    weights: dict[int, dict[int, float]] | None = None,
    missing_state_indicator: int = -1,
    threads: int = 1,
) -> np.array:
    """Compute the dissimilarity between all samples.

    An optimized function for computing pairwise dissimilarities between
    samples in a character matrix according to the dissimilarity function.

    Args:
        cm: Character matrix
        C: Number of samples
        weights: Weights to use for comparing states.
        dissimilarity_function: Dissimilarity function that returns the distance
            between two character states.
        missing_state_indicator: State indicating missing data
        threads: Number of threads to use for distance computation.

    Returns:
            A dissimilarity mapping as a flattened array.
    """
    # check to see if any ambiguous characters are present
    ambiguous_present = np.any([(cm[:, i].dtype == "object") for i in range(cm.shape[1])])

    # Compile (and cache) the dissimilarity function, falling back to python.
    numbaize = True
    try:
        dissimilarity_func = _jit_metric(dissimilarity_function, bool(ambiguous_present))
        if ambiguous_present:
            numbaize = False

    # When cluster_dissimilarity is used, the dissimilarity_function is wrapped
    # in a partial, which raises a TypeError when trying to numbaize.
    except TypeError:
        warnings.warn(
            "Failed to numbaize dissimilarity function. Falling back to Python.",
            CassiopeiaTreeWarning,
            stacklevel=2,
        )
        numbaize = False
        dissimilarity_func = dissimilarity_function

    if threads > 1:
        dm = np.zeros(C * (C - 1) // 2, dtype=np.float64)
        k, m = divmod(len(dm), threads)
        batches = [
            np.arange(len(dm))[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(threads)
        ]

        # load character matrix into shared memory
        shm = shared_memory.SharedMemory(create=True, size=cm.nbytes)
        shared_cm = np.ndarray(cm.shape, dtype=cm.dtype, buffer=shm.buf)
        shared_cm[:] = cm[:]

        with multiprocessing.Pool(processes=threads) as pool:
            results = list(
                pool.starmap(
                    __compute_dissimilarity_map_wrapper,
                    [
                        (
                            dissimilarity_func,
                            shared_cm,
                            batch,
                            weights,
                            missing_state_indicator,
                            numbaize,
                            ambiguous_present,
                        )
                        for batch in batches
                    ],
                ),
            )

        for batch_indices, batch_results in results:
            dm[batch_indices] = batch_results

        # Clean up shared memory buffer
        del shared_cm
        shm.close()
        shm.unlink()
    else:
        (_, dm) = __compute_dissimilarity_map_wrapper(
            dissimilarity_func,
            cm,
            np.arange(C * (C - 1) // 2),
            weights,
            missing_state_indicator,
            numbaize,
            ambiguous_present,
        )

    return dm


def __compute_dissimilarity_map_wrapper(
    dissimilarity_func: Callable,
    cm: np.ndarray,
    batch_indices: np.ndarray,
    weights: dict[int, dict[int, float]] | None = None,
    missing_state_indicator: int = -1,
    numbaize: bool = True,
    ambiguous_present: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper function for parallel computation of dissimilarity maps.

    This is a wrapper function that is intended to interface with
    compute_dissimilarity_map. The reason why this is necessary is because
    specific numba objects are not compatible with the multiprocessing library
    used for parallel computation of the dissimilarity matrix.

    While there is a minor hit when using multiple threads for a function
    that can be jit compiled with numba, this effect is negligible and the
    benefits of a parallel dissimilarity matrix computation for
    non-jit-compatible function far outweighs the minor slow down.

    Args:
        dissimilarity_func: A pre-compiled dissimilarity function.
        cm: Character matrix
        batch_indices: Batch indicies. These refer to a set of compressed
            indices for a final square dissimilarity matrix. This function
            will only compute the dissimilarity for these indices.
        missing_state_indicator: Integer value representing missing states.
        weights: Weights to use for comparing states.
        numbaize: Whether or not to numbaize the final dissimilarity map
            computation, based on whether or not the dissimilarity function
            was compatible with jit-compilation.
        ambiguous_present: Whether or not ambiguous states are present.

    Returns:
            A tuple of (batch_indices, batch_results) indicating the dissimilarities
            for the comparisons specified by batch_indices.
    """
    nb_weights = numba.typed.Dict.empty(
        numba.types.int64,
        numba.types.DictType(numba.types.int64, numba.types.float64),
    )
    if weights:
        for k, v in weights.items():
            nb_char_weights = numba.typed.Dict.empty(numba.types.int64, numba.types.float64)
            for state, prior in v.items():
                nb_char_weights[state] = prior
            nb_weights[k] = nb_char_weights

    compute = _jit_loop(dissimilarity_func, numbaize, ambiguous_present)
    return compute(cm, batch_indices, missing_state_indicator, nb_weights)
