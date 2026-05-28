"""Module containing general utilities to be called by functions throughout the solver module."""

import time
from collections.abc import Generator
from hashlib import blake2b

import ete3
import networkx as nx
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
        raise NotImplementedError(
            "collapse_mutationless_edges is not supported for TreeData."
        )
    data.collapse_mutationless_edges(infer_ancestral_characters=True)


