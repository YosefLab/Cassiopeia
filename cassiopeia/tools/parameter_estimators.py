"""
This file stores functions for estimating lineage tracing parameters.
Currently, we'll support the estimation of mutation and missing data rates.
"""
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data.Layers import Layers
from cassiopeia.mixins import CassiopeiaTreeError, CassiopeiaTreeWarning


def get_proportion_of_missing_data(tree, layer: Optional[str] = None) -> float:
    """Calculates the proportion of missing entries in the character matrix.

    Calculates the proportion of cell/character pairs in the character
    matrix that have a non-missing state, as indicated by
    `tree.missing_state_indicator`.

    Args:
        tree: The tree on which to calculate the proportion of missing data
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns:
        The missing data proportion

    Raises:
        CassiopeiaTreeError if character matrix or layer doesn't exist
    """

    if layer:
        character_matrix = tree.layers[layer]
    else:
        character_matrix = tree.character_matrix

    if character_matrix is None:
        raise CassiopeiaTreeError(
            "No character matrix is detected in this tree."
        )

    num_dropped = (
        character_matrix.values == tree.missing_state_indicator
    ).sum()
    missing_proportion = num_dropped / (
        character_matrix.shape[0] * character_matrix.shape[1]
    )
    return missing_proportion


def get_proportion_of_mutation(tree, layer: Optional[str] = None) -> float:
    """Calculates the proportion of mutated entries in the character matrix.

    Calculates the proportion of non-missing cell/character pairs in the
    character matrix that have a non-uncut (non-0) state.

    Args:
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns:
        The proportion of mutated cell/character pairs

    Raises:
        CassiopeiaTreeError if character matrix or layer doesn't exist
    """

    if layer:
        character_matrix = tree.layers[layer]
    else:
        character_matrix = tree.character_matrix

    if character_matrix is None:
        raise CassiopeiaTreeError(
            "No character matrix is detected in this tree."
        )

    num_dropped = (
        character_matrix.values == tree.missing_state_indicator
    ).sum()

    num_mut = (
        character_matrix.shape[0] * character_matrix.shape[1]
        - num_dropped
        - (character_matrix.values == 0).sum()
    )
    mutation_proportion = num_mut / (
        character_matrix.shape[0] * character_matrix.shape[1] - num_dropped
    )
    return mutation_proportion


def estimate_mutation_rate(
    tree,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    layer: Optional[str] = None,
) -> float:
    """Estimates the mutation rate from tree and a character matrix.

    Infers the mutation rate using the proportion of the cell/character
    pairs in the leaves that have a non-uncut (non-0) state and the number of
    generations/the total time of the tree. We treat each lineage as
    independent and use the proportion as an estimate for the probability
    that an event occurs on a lineage.

    In the case where the rate is per-generation,
    it is estimated using:

        mutated proportion = 1 - (1 - mutation_rate) ^ (average depth of tree)

    In the case when the rate is continuous, it is estimated using:

        mutated proportion = ExponentialCDF(total time of tree, mutation rate)

    Note that these naive estimates perform better when the tree is ultrametric
    in depth or time.

    This function attempts to consume the mutation rate as `mutation_rate` in
    `tree.parameters` in order to infer the mutated proportion. If this field
    is not populated, it is inferred using `get_proportion_of_mutation`.

    In the inference, we need to consider whether to assume an implicit root,
    specified by `assume_root_implicit_branch`. In the case where the tree does
    not have a single leading edge from the root representing the progenitor
    cell before cell division begins, this additional edge is added to the
    total time in calculating the estimate if `assume_root_implicit_branch` is
    True.

    Args:
        tree: The CassioepiaTree specifying the tree and the character matrix
        continuous: Whether to calculate a continuous mutation rate,
            accounting for branch lengths. Otherwise, calculates a
            discrete mutation rate using the number of generations
        assume_root_implicit_branch: Whether to assume that there is an
            implicit branch leading from the root, if it doesn't exist
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns:
        The estimated mutation rate

    Raises:
        CassiopeiaTreeError if the mutation proportion parameter is not
        between 0 and 1
    """
    if "mutated_proportion" not in tree.parameters:
        mutation_proportion = get_proportion_of_mutation(tree, layer)
    else:
        mutation_proportion = tree.parameters["mutated_proportion"]
    if mutation_proportion < 0 or mutation_proportion > 1:
        raise CassiopeiaTreeError(
            "Mutation proportion must be between 0 and 1."
        )
    if not continuous:
        mean_depth = tree.get_mean_depth_of_tree()
        # We account for the added depth of the implicit branch leading
        # from the root, if it is to be added
        if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
            mean_depth += 1
        mutation_rate = 1 - (1 - mutation_proportion) ** (1 / mean_depth)
    else:
        times = tree.get_times()
        mean_time = np.mean([times[l] for l in tree.leaves])
        if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
            mean_time += np.mean(
                [tree.get_branch_length(u, v) for u, v in tree.edges]
            )
        mutation_rate = -np.log(1 - mutation_proportion) / mean_time
    return mutation_rate


def estimate_stochastic_missing_data_probability(
    tree,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    proportion_of_missing_as_stochastic: float = 0.5,
    layer: Optional[str] = None,
) -> float:
    """Estimates the stochastic missing probability from a tree and a character
    matrix.

    Infers the stochastic missing data probability using the proportion of the
    cell/character pairs in the leaves. We use the proportion as an estimate
    for the probability that a character at a leaf has stochastic missing data.

    As missing data can be either heritable or stochastic, we assume both
    contribute to the total missing proportion as:

        total missing proportion = heritable proportion + stochastic proportion
            - heritable proportion * stochastic proportion

    This function attempts to consume the missing proportion as
    `missing_proportion` in `tree.parameters`, inferring it using
    `get_proportion_of_missing_data` if it is not populated. Additionally, it
    attempts to consume the heritable missing rate as `heritable_missing_rate`
    in `tree.parameters` in order to infer the stochastic proportion using the
    convolution above. If this field is not populated, then the stochastic
    proportion is assumed to be `proportion_of_missing_as_stochastic` * the
    total missing proportion.

    In calculating the heritable proportion from the heritable missing rate,
    we need to consider whether to assume an implicit root. This is specified
    by `assume_root_implicit_branch`. In the case where the tree does not have
    a single leading edge from the root representing the progenitor cell before
    cell division begins, this additional edge is added to the total time in
    calculating the estimate if `assume_root_implicit_branch` is True.

    Args:
        tree: The CassiopeiaTree specifying the tree and the character matrix
        continuous: Whether to calculate a continuous missing rate,
            accounting for branch lengths. Otherwise, calculates a
            discrete missing rate based on the number of generations
        assume_root_implicit_branch: Whether to assume that there is an
            implicit branch leading from the root, if it doesn't exist

    Returns:
        The stochastic missing probability.

    Raises:
        CassiopeiaTreeError if the missing proportion parameter is not
        between 0 and 1, CassiopeiaTreeWarning if the estimated rates
        are negative
    """
    if (
        proportion_of_missing_as_stochastic < 0
        or proportion_of_missing_as_stochastic > 1
    ):
        raise CassiopeiaTreeError(
            "Proportion of missing as stochastic must be between 0 and 1"
        )

    if "missing_proportion" not in tree.parameters:
        total_missing_proportion = get_proportion_of_missing_data(tree, layer)
    else:
        total_missing_proportion = tree.parameters["missing_proportion"]
    if total_missing_proportion < 0 or total_missing_proportion > 1:
        raise CassiopeiaTreeError("Missing proportion must be between 0 and 1.")

    heritable_missing_rate = None
    stochastic_missing_probability = None
    if "heritable_missing_rate" in tree.parameters:
        heritable_missing_rate = tree.parameters["heritable_missing_rate"]
        if heritable_missing_rate < 0:
            raise CassiopeiaTreeError(
                "Heritable missing data rate must be > 0."
            )
        if not continuous and heritable_missing_rate > 1:
            raise CassiopeiaTreeError(
                "Per-generation heritable missing data rate must be < 1."
            )

    if heritable_missing_rate is None:
        stochastic_missing_probability = (
            proportion_of_missing_as_stochastic * total_missing_proportion
        )
    else:
        if not continuous:
            mean_depth = tree.get_mean_depth_of_tree()
            # We account for the added depth of the implicit branch leading
            # from the root, if it is to be added
            if (
                assume_root_implicit_branch
                and len(tree.children(tree.root)) != 1
            ):
                mean_depth += 1
            heritable_proportion = 1 - (1 - heritable_missing_rate) ** (
                mean_depth
            )
            stochastic_missing_probability = (
                total_missing_proportion - heritable_proportion
            ) / (1 - heritable_proportion)
        else:
            times = tree.get_times()
            mean_time = np.mean([times[l] for l in tree.leaves])
            if (
                assume_root_implicit_branch
                and len(tree.children(tree.root)) != 1
            ):
                mean_time += np.mean(
                    [tree.get_branch_length(u, v) for u, v in tree.edges]
                )

            heritable_proportion = 1 - np.exp(
                -heritable_missing_rate * mean_time
            )
            stochastic_missing_probability = (
                total_missing_proportion - heritable_proportion
            ) / (1 - heritable_proportion)

    if stochastic_missing_probability < 0:
        raise CassiopeiaTreeWarning(
            "Estimate of the stochastic missing rate using "
            "the heritable rate resulted in a negative "
            "stochastic missing rate."
        )

    return stochastic_missing_probability


def estimate_heritable_missing_data_rate(
    tree,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    proportion_of_missing_as_stochastic: float = 0.5,
    layer: Optional[str] = None,
):
    """Estimates the heritable missing rate from a tree and a character matrix.

    Infers the heritable mutation rate using the proportion of the
    cell/character pairs in the leaves that have missing data and the number
    of generations/the total time of the tree. We treat each lineage as
    independent and use the proportion as an estimate for the probability
    that an event occurs on a lineage.

    As missing data can be either heritable or stochastic, we assume both
    contribute to the total missing proportion as:

        total missing proportion = heritable proportion + stochastic proportion
            - heritable proportion * stochastic proportion

    In the case where the rate is per-generation, it is estimated using:

        heritable missing proportion =
            1 - (1 - heritable missing rate) ^ (average depth of tree)

    In the case where the rate is continuous, it is estimated using:

        heritable_missing_proportion =
            ExponentialCDF(total time of tree, heritable missing rate)

    Note that these naive estimates perform better when the tree is ultrametric
    in depth or time.

    This function attempts to consume the missing proportion as
    `missing_proportion` in `tree.parameters`, inferring it using
    `get_proportion_of_missing_data` if it is not populated. Additionally,
    it attempts to consume the stochastic as proportion
    `stochastic_missing_probability` in `tree.parameters`. If this field is
    not populated, then the stochastic proportion is assumed to be
    `proportion_of_missing_as_stochastic` * the total missing proportion.

    In the inference, we need to consider whether to assume an implicit root,
    specified by `assume_root_implicit_branch`. In the case where the tree does
    not have a single leading edge from the root representing the progenitor
    cell before cell division begins, this additional edge is added to the
    total time in calculating the estimate if `assume_root_implicit_branch` is
    True.

    Args:
        continuous: Whether to calculate a continuous missing rate,
            accounting for branch lengths. Otherwise, calculates a
            discrete missing rate based on the number of generations
        assume_root_implicit_branch: Whether to assume that there is an
            implicit branch leading from the root, if it doesn't exist

    Returns:
        None, populates the `parameters` attribute as
        "stochastic_missing_probability" and "heritable_missing_rate"

    Raises:
        CassiopeiaTreeError if the missing proportion parameter is not
        between 0 and 1, CassiopeiaTreeWarning if the estimated rates
        are negative
    """
    if (
        proportion_of_missing_as_stochastic < 0
        or proportion_of_missing_as_stochastic > 1
    ):
        raise CassiopeiaTreeError(
            "Proportion of missing as stochastic must be between 0 and 1"
        )

    if "missing_proportion" not in tree.parameters:
        total_missing_proportion = get_proportion_of_missing_data(tree, layer)
    else:
        total_missing_proportion = tree.parameters["missing_proportion"]
    if total_missing_proportion < 0 or total_missing_proportion > 1:
        raise CassiopeiaTreeError("Missing proportion must be between 0 and 1.")

    stochastic_missing_probability = None
    heritable_missing_rate = None
    if "stochastic_missing_probability" in tree.parameters:
        stochastic_missing_probability = tree.parameters[
            "stochastic_missing_probability"
        ]
        if stochastic_missing_probability is not None:
            if stochastic_missing_probability < 0:
                raise CassiopeiaTreeError(
                    "Stochastic missing data rate must be > 0."
                )
            if stochastic_missing_probability > 1:
                raise CassiopeiaTreeError(
                    "Stochastic missing data rate must be < 1."
                )
    if stochastic_missing_probability is None:
        stochastic_missing_probability = (
            proportion_of_missing_as_stochastic * total_missing_proportion
        )

    if not continuous:
        mean_depth = tree.get_mean_depth_of_tree()
        # We account for the added depth of the implicit branch leading
        # from the root, if it is to be added
        if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
            mean_depth += 1
        heritable_missing_rate = 1 - (
            (1 - total_missing_proportion)
            / (1 - stochastic_missing_probability)
        ) ** (1 / mean_depth)
    else:
        times = tree.get_times()
        mean_time = np.mean([times[l] for l in tree.leaves])
        if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
            mean_time += np.mean(
                [tree.get_branch_length(u, v) for u, v in tree.edges]
            )
        heritable_missing_rate = (
            -np.log(
                (1 - total_missing_proportion)
                / (1 - stochastic_missing_probability)
            )
            / mean_time
        )

    if heritable_missing_rate < 0:
        raise CassiopeiaTreeWarning(
            "Estimate of the heritable missing rate using "
            "the stochastic rate resulted in a negative "
            "heritable missing rate."
        )

    return heritable_missing_rate
