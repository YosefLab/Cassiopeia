"""Module containing general utilities to be called by functions throughout the solver module."""

import time
from collections.abc import Callable, Generator
from hashlib import blake2b

import ete3
import numpy as np
import pandas as pd

from cassiopeia.mixins import PriorTransformationError


def node_name_generator() -> Generator[str, None, None]:
    """Generates unique node names for building the reconstructed tree.

    Creates a generator object that produces unique node names by hashing
    timestamps.

    Returns:
            A generator object
    """
    while True:
        k = str(time.time()).encode("utf-8")
        h = blake2b(key=k, digest_size=12)
        yield "cassiopeia_internal_node" + h.hexdigest()


def collapse_unifurcations(tree: ete3.Tree) -> ete3.Tree:
    """Collapse unifurcations.

    Collapse all unifurcations in the tree, namely any node with only one child
    should be removed and all children should be connected to the parent node.

    Args:
        tree: tree to be collapsed
    Returns:
        A collapsed tree.
    """
    collapse_fn = lambda x: (len(x.children) == 1)

    collapsed_tree = tree.copy()
    to_collapse = [n for n in collapsed_tree.traverse() if collapse_fn(n)]

    for n in to_collapse:
        n.delete()

    return collapsed_tree


def transform_priors(
    priors: dict[int, dict[int, float]] | None,
    prior_transformation: str = "negative_log",
) -> dict[int, dict[int, float]]:
    """Generates a dictionary of weights from priors.

    Generates a dictionary of weights from given priors for each character/state
    pair for use in algorithms that inherit the GreedySolver. Supported
    transformations include negative log, negative log square root, and inverse.

    Args:
        priors: A dictionary of prior probabilities for each character/state
            pair
        prior_transformation: A function defining a transformation on the priors
            in forming weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative log
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Returns:
            A dictionary of weights for each character/state pair
    """
    if prior_transformation not in [
        "negative_log",
        "inverse",
        "square_root_inverse",
    ]:
        raise PriorTransformationError("Please select one of the supported prior transformations.")

    prior_function = lambda x: -np.log(x)

    if prior_transformation == "square_root_inverse":
        prior_function = lambda x: (np.sqrt(1 / x))
    if prior_transformation == "inverse":
        prior_function = lambda x: 1 / x

    weights = {}
    for character in priors:
        state_weights = {}
        for state in priors[character]:
            p = priors[character][state]
            if p <= 0.0 or p > 1.0:
                raise PriorTransformationError(
                    "Please make sure all priors have a value between 0 and 1"
                )
            state_weights[state] = prior_function(p)
        weights[character] = state_weights
    return weights


def convert_sample_names_to_indices(names: list[str], samples: list[str]) -> list[int]:
    """Maps samples to their integer indices in a given set of names.

    Used to map sample string names to the their integer positions in the index
    of the original character matrix for efficient indexing operations.

    Args:
        names: A list of sample names, represented by their string names in the
            original character matrix
        samples: A list of sample names representing the subset to be mapped to
            integer indices

    Returns:
            A list of samples mapped to integer indices
    """
    name_to_index = dict(zip(names, range(len(names)), strict=False))

    return [name_to_index[x] for x in samples]


# ── Data-object helpers ───────────────────────────────────────────────────────


def _get_characters(data, characters_key=None):
    """Return the character matrix as a pd.DataFrame, or None if unavailable.

    Thin wrapper around :func:`cassiopeia.utils._get_characters` that maps the
    ``characters_key`` parameter name used throughout the solver module.
    """
    from cassiopeia.utils import _get_characters as _utils_get_characters

    return _utils_get_characters(data, key=characters_key)


def _get_digraph(data, tree_key=None):
    """Return the stored tree DiGraph, or None if not yet populated.

    Args:
        data: CassiopeiaTree or TreeData.
        tree_key: For TreeData, the obst key to look up.  Ignored for CassiopeiaTree.
    """
    from treedata import TreeData

    if isinstance(data, TreeData):
        return data.obst.get(tree_key) if tree_key else None
    return data.get_tree_topology()


def _set_tree(data, rooted, characters_key=None, tree_key=None):
    """Store *rooted* DiGraph in *data*, handling both CassiopeiaTree and TreeData.

    For CassiopeiaTree: infers the root node, drops it from the character matrix
    if it was an original sample (e.g. ``root_sample_name`` or synthetic ``'root'``),
    then calls ``populate_tree`` and ``collapse_unifurcations``.

    For TreeData: assigns ``rooted`` to ``data.obst[tree_key]``.

    Args:
        data: CassiopeiaTree or TreeData to modify in-place.
        rooted: Directed tree DiGraph to store.
        characters_key: Layer name for CassiopeiaTree's populate_tree.
        tree_key: obst key for TreeData.
    """
    from treedata import TreeData

    if isinstance(data, TreeData):
        data.obst[tree_key] = rooted
    else:
        roots = [n for n in rooted.nodes() if rooted.in_degree(n) == 0]
        root_node = roots[0] if roots else None
        if root_node is not None and root_node in data.character_matrix.index:
            data.character_matrix = data.character_matrix.drop(index=root_node)
        data.root_sample_name = root_node
        data.populate_tree(rooted, layer=characters_key)
        data.collapse_unifurcations()


def _get_missing_and_priors(data) -> tuple[int, dict | None]:
    """Return ``(missing_state_indicator, priors)`` for a data object.

    Reads from ``TreeData.uns`` or the corresponding ``CassiopeiaTree``
    attributes.
    """
    from treedata import TreeData

    if isinstance(data, TreeData):
        return data.uns.get("missing_state_indicator", -1), data.uns.get("priors", None)
    return data.missing_state_indicator, data.priors


def get_distance_map(
    data,
    dissimilarity_fn: Callable | None,
    characters_key: str | None = None,
    dissim_key: str | None = None,
    prior_transformation: str = "negative_log",
    threads: int = 1,
) -> pd.DataFrame:
    """Return a symmetric n×n distance ``pd.DataFrame`` for a data object.

    Resolves distances for distance-based solvers, handling both data types
    *without* side effects (it never writes the map back):

    - **TreeData**: uses ``obsp[dissim_key]`` when *dissim_key* is given,
      otherwise computes from ``obsm[characters_key]`` via
      :func:`cassiopeia.dissimilarity._pairwise`.
    - **CassiopeiaTree**: returns the cached dissimilarity map when present (and
      no explicit layer is requested), otherwise computes from the character
      matrix via :func:`cassiopeia.dissimilarity._pairwise`.

    Args:
        data: CassiopeiaTree or TreeData.
        dissimilarity_fn: Resolved dissimilarity callable, or ``None``.
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm`` key
            (TreeData).
        dissim_key: ``obsp`` key for precomputed distances (TreeData only).
        prior_transformation: Prior weight transformation name.
        threads: Threads for parallel computation.

    Raises:
        DistanceSolverError: If distances must be computed but no dissimilarity
            function is available.
    """
    from treedata import TreeData

    from cassiopeia.dissimilarity import _pairwise

    if isinstance(data, TreeData):
        if dissim_key is not None:
            names = list(data.obs_names)
            return pd.DataFrame(
                np.asarray(data.obsp[dissim_key], dtype=np.float64),
                index=names,
                columns=names,
            )
        chars = _get_characters(data, characters_key)
        if chars is None:
            raise ValueError(
                "TreeData has no character matrix; store characters in "
                f"obsm[{characters_key or 'characters'!r}] or provide dissim_key."
            )
        missing, priors = _get_missing_and_priors(data)
        return _pairwise(chars, dissimilarity_fn, missing, priors, prior_transformation, threads)

    # CassiopeiaTree
    cached = data.get_dissimilarity_map()
    if characters_key is None and cached is not None:
        return cached
    chars = _get_characters(data, characters_key)
    missing, priors = _get_missing_and_priors(data)
    return _pairwise(chars, dissimilarity_fn, missing, priors, prior_transformation, threads)


def save_distance_map(data, dist_df: pd.DataFrame, dissim_key: str | None = None) -> None:
    """Store a pairwise distance map on a data object.

    For :class:`~treedata.TreeData` the dense matrix is written to
    ``obsp[dissim_key or 'distances']``; for
    :class:`~cassiopeia.data.CassiopeiaTree` it is set via
    ``set_dissimilarity_map``.

    Args:
        data: CassiopeiaTree or TreeData to modify in-place.
        dist_df: Symmetric distance ``pd.DataFrame`` indexed by sample name.
        dissim_key: ``obsp`` key for the stored matrix (TreeData only).
    """
    from treedata import TreeData

    if isinstance(data, TreeData):
        names = list(data.obs_names)
        data.obsp[dissim_key or "distances"] = dist_df.loc[names, names].to_numpy()
    else:
        data.set_dissimilarity_map(dist_df)


def collapse_mutationless_edges(data, characters_key=None, tree_key=None):
    """Collapse edges with no inferred mutations (CassiopeiaTree only).

    Calls ``data.collapse_mutationless_edges(infer_ancestral_characters=True)``.

    Args:
        data: CassiopeiaTree to modify in-place.
        characters_key: Unused; present for API symmetry with other utils.
        tree_key: Unused; present for API symmetry with other utils.

    Raises:
        NotImplementedError: If *data* is a TreeData object.
    """
    from treedata import TreeData

    if isinstance(data, TreeData):
        raise NotImplementedError("collapse_mutationless_edges is not supported for TreeData.")
    data.collapse_mutationless_edges(infer_ancestral_characters=True)
