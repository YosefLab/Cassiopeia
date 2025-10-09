"""Module containing general utilities to be called by functions throughout

the solver module
"""

import time
from collections.abc import Generator
from hashlib import blake2b

import ete3
import numpy as np
import pandas as pd

from cassiopeia.mixins import PriorTransformationError


def node_name_generator() -> Generator[str, None, None]:
    """Generates unique node names for building the reconstructed tree.

    Creates a generator object that produces unique node names by hashing
    timestamps.

    Returns
    -------
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

    Returns
    -------
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
                raise PriorTransformationError("Please make sure all priors have a value between 0 and 1")
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

    Returns
    -------
        A list of samples mapped to integer indices
    """
    name_to_index = dict(zip(names, range(len(names)), strict=False))

    return [name_to_index[x] for x in samples]


def save_dissimilarity_as_phylip(dissimilarity_map: pd.DataFrame, path: str) -> None:
    """Saves a dissimilarity map as a phylip file.

    Args:
        dissimilarity_map: A dissimilarity map
        path: The path to save the phylip file

    Returns
    -------
        None
    """
    dissimilarity_np = dissimilarity_map.to_numpy()
    n = dissimilarity_np.shape[0]
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = dissimilarity_np[i, : i + 1]
            formatted_values = "\t".join(map("{:.4f}".format, row))
            f.write(f"{dissimilarity_map.index[i]}\t{formatted_values}\n")
