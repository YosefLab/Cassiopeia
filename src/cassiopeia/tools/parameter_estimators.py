"""
This file stores functions for estimating lineage tracing parameters.
Currently, we'll support the estimation of mutation and missing data rates.
"""

import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning


def get_proportion_of_missing_data(tree: CassiopeiaTree, layer: str | None = None) -> float:
    """Calculates the proportion of missing entries in the character matrix.

    Calculates the proportion of cell/character entries in the character matrix
    that have a non-missing state, with the missing state being indicated by
    `tree.missing_state_indicator`.

    Args:
        tree: The CassiopeiaTree specifying the tree and the character matrix
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns
    -------
        The proportion of missing cell/character entries

    Raises
    ------
        ParameterEstimateError if character matrix or layer doesn't exist
    """
    if layer:
        character_matrix = tree.layers[layer]
    else:
        character_matrix = tree.character_matrix

    if character_matrix is None:
        raise ParameterEstimateError("No character matrix is detected in this tree.")

    num_dropped = (character_matrix.values == tree.missing_state_indicator).sum()
    missing_proportion = num_dropped / (character_matrix.shape[0] * character_matrix.shape[1])
    return missing_proportion


def get_proportion_of_mutation(tree: CassiopeiaTree, layer: str | None = None) -> float:
    """Calculates the proportion of mutated entries in the character matrix.

    Calculates the proportion of cell/character entries in the character matrix
    that have a non-uncut (non-0) state, normalizing over non-missing entries.
    Hence, missing entries are not considered in calculating the proportion.

    Args:
        tree: The CassiopeiaTree specifying the tree and the character matrix
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns
    -------
        The proportion of non-missing cell/character entries that are mutated

    Raises
    ------
        ParameterEstimateError if character matrix or layer doesn't exist
    """
    if layer:
        character_matrix = tree.layers[layer]
    else:
        character_matrix = tree.character_matrix

    if character_matrix is None:
        raise ParameterEstimateError("No character matrix is detected in this tree.")

    num_dropped = (character_matrix.values == tree.missing_state_indicator).sum()

    num_mut = character_matrix.shape[0] * character_matrix.shape[1] - num_dropped - (character_matrix.values == 0).sum()
    mutation_proportion = num_mut / (character_matrix.shape[0] * character_matrix.shape[1] - num_dropped)
    return mutation_proportion


def estimate_mutation_rate(
    tree: CassiopeiaTree,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    layer: str | None = None,
) -> float:
    """Estimates the mutation rate from a tree and character matrix.

    Infers the mutation rate using the proportion of the cell/character
    entries in the leaves that have a non-uncut (non-0) state and the node
    depth/the total time of the tree. The mutation rate is either a
    continuous or per-generation rate according to which lineages accumulate
    mutations.

    In estimating the mutation rate, we use the observed proportion of mutated
    entries in the character matrix as an estimate of the probability that a
    mutation occurs on a lineage. Using this probability, we can then infer
    the mutation rate.

    This function attempts to consume the observed mutation proportion as
    `mutation_proportion` in `tree.parameters`. If this field is not populated,
    it is inferred using `get_proportion_of_mutation`.

    In the case where the rate is per-generation (probability a mutation occurs
    on an edge), it is estimated using:

        mutated proportion = 1 - (1 - mutation_rate) ^ (average depth of tree)

    In the case when the rate is continuous, it is estimated using:

        mutated proportion = ExponentialCDF(average time of tree, mutation rate)

    Note that these naive estimates perform better when the tree is ultrametric
    in depth or time. The average depth/lineage time of the tree is used as a
    proxy for the depth/total time when the tree is not ultrametric.

    In the inference, we need to consider whether to assume an implicit root,
    specified by `assume_root_implicit_branch`. In the case where the tree does
    not have a single leading edge from the root representing the progenitor
    cell before cell division begins, this additional edge is added to the
    total time in calculating the estimate if `assume_root_implicit_branch` is
    True.

    Args:
        tree: The CassiopeiaTree specifying the tree and the character matrix
        continuous: Whether to calculate a continuous mutation rate,
            accounting for branch lengths. Otherwise, calculates a
            discrete mutation rate using the node depths
        assume_root_implicit_branch: Whether to assume that there is an
            implicit branch leading from the root, if it doesn't exist
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns
    -------
        The estimated mutation rate

    Raises
    ------
        ParameterEstimateError if the `mutation_proportion` parameter is not
            between 0 and 1
    """
    if "mutated_proportion" not in tree.parameters:
        mutation_proportion = get_proportion_of_mutation(tree, layer)
    else:
        mutation_proportion = tree.parameters["mutated_proportion"]
    if mutation_proportion < 0 or mutation_proportion > 1:
        raise ParameterEstimateError("Mutation proportion must be between 0 and 1.")
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
            mean_time += np.mean([tree.get_branch_length(u, v) for u, v in tree.edges])
        mutation_rate = -np.log(1 - mutation_proportion) / mean_time
    return mutation_rate


def estimate_missing_data_rates(
    tree: CassiopeiaTree,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    stochastic_missing_probability: float | None = None,
    heritable_missing_rate: float | None = None,
    layer: str | None = None,
) -> tuple[float, float]:
    """
    Estimates both missing data parameters given one of the two from a tree.

    The stochastic missing probability is the probability that any given
    cell/character pair acquires stochastic missing data in the character
    matrix due to low-capture in single-cell RNA sequencing. The heritable
    missing rate is either a continuous or per-generation rate according to
    which lineages accumulate heritable missing data events, such as
    transcriptional silencing or resection.

    In most instances, the two types of missing data are convolved and we
    determine whether any single occurrence of missing data is due to stochastic
    or heritable missing data. We assume both contribute to the total amount of
    missing data as:

        total missing proportion = heritable proportion + stochastic proportion
            - heritable proportion * stochastic proportion

    This function attempts to consume the amount of missing data (the total
    missing proportion) as `missing_proportion` in `tree.parameters`, inferring
    it using `get_proportion_of_missing_data` if it is not populated.

    Additionally, as the two types of data are convolved, we need to know the
    contribution of one of the types of missing data in order to estimate the
    other. This function attempts to consume the heritable missing rate as
    `heritable_missing_rate` in `tree.parameters` and the stochastic missing
    probability as `stochastic_missing_probability` in `tree.parameters`.
    If they are not provided on the tree, then they may be provided as
    function arguments. If neither or both parameters are provided by either of
    these methods, the function errors.

    In estimating the heritable missing rate from the stochastic missing data
    probability, we take the proportion of stochastic missing data in the
    character matrix as equal to the stochastic probability. Then using the
    total observed proportion of missing data as well as the estimated
    proportion of stochastic missing data we can estimate the proportion
    of heritable missing data using the expression above. Finally, we use the
    heritable proportion as an estimate of the probability a lineage acquires
    a missing data event by the end of the phylogeny, and using this
    probability we can estimate the rate.

    In the case where the rate is per-generation (probability a heritable
    missing data event occurs on an edge), it is estimated using:

        heritable missing proportion =
            1 - (1 - heritable missing rate) ^ (average depth of tree)

    In the case where the rate is continuous, it is estimated using:

        heritable_missing_proportion =
            ExponentialCDF(average time of tree, heritable missing rate)

    Note that these naive estimates perform better when the tree is ultrametric
    in depth or time. The average depth/lineage time of the tree is used as a
    proxy for the depth/total time when the tree is not ultrametric.

    In calculating the heritable proportion from the heritable missing rate,
    we need to consider whether to assume an implicit root. This is specified
    by `assume_root_implicit_branch`. In the case where the tree does not have
    a single leading edge from the root representing the progenitor cell before
    cell division begins, this additional edge is added to the total time in
    calculating the estimate if `assume_root_implicit_branch` is True.

    In estimating the stochastic missing probability from the heritable missing
    rate, we calculate the expected proportion of heritable missing data using
    the heritable rate in the same way, and then as above use the total
    proportion of missing data to estimate the stochastic proportion, which we
    assume is equal to the probability.

    Args:
        tree: The CassiopeiaTree specifying the tree and the character matrix
        continuous: Whether to calculate a continuous missing rate,
            accounting for branch lengths. Otherwise, calculates a
            discrete missing rate based on the number of generations
        assume_root_implicit_branch: Whether to assume that there is an
            implicit branch leading from the root, if it doesn't exist
        stochastic_missing_probability: The stochastic missing probability.
            Will override the value on the tree. Observed probabilites of
            stochastic missing data range between 10-20%
        heritable_missing_rate: The heritable missing rate. Will override the
            value on the tree
        layer: Layer to use for character matrix. If this is None,
            then the current `character_matrix` variable will be used.

    Returns
    -------
        The stochastic missing probability and heritable missing rate. One of
        these will be the parameter as provided, the other will be an estimate

    Raises
    ------
        ParameterEstimateError if the `total_missing_proportion`,
            `stochastic_missing_probability`, or `heritable_missing_rate` that
            are provided have invalid values, or if both or neither of
            `stochastic_missing_probability`, and `heritable_missing_rate` are
            provided. ParameterEstimateWarning if the estimated parameter is
            negative
    """
    if "missing_proportion" not in tree.parameters:
        total_missing_proportion = get_proportion_of_missing_data(tree, layer)
    else:
        total_missing_proportion = tree.parameters["missing_proportion"]
    if total_missing_proportion < 0 or total_missing_proportion > 1:
        raise ParameterEstimateError("Missing proportion must be between 0 and 1.")

    if stochastic_missing_probability is None:
        if "stochastic_missing_probability" in tree.parameters:
            stochastic_missing_probability = tree.parameters["stochastic_missing_probability"]

    if heritable_missing_rate is None:
        if "heritable_missing_rate" in tree.parameters:
            heritable_missing_rate = tree.parameters["heritable_missing_rate"]

    if heritable_missing_rate is None and stochastic_missing_probability is None:
        raise ParameterEstimateError(
            "Neither `heritable_missing_rate` nor "
            "`stochastic_missing_probability` were provided as arguments or "
            "found in `tree.parameters`. Please provide one of these "
            "parameters, otherwise they are convolved and cannot be estimated"
        )

    if heritable_missing_rate is not None and stochastic_missing_probability is not None:
        raise ParameterEstimateError(
            "Both `heritable_missing_rate` and `stochastic_missing_probability`"
            " were provided as parameters or found in `tree.parameters`. "
            "Please only supply one of the two"
        )

    if heritable_missing_rate is None:
        if stochastic_missing_probability < 0:
            raise ParameterEstimateError("Stochastic missing data rate must be > 0.")
        if stochastic_missing_probability > 1:
            raise ParameterEstimateError("Stochastic missing data rate must be < 1.")

        if not continuous:
            mean_depth = tree.get_mean_depth_of_tree()
            # We account for the added depth of the implicit branch leading
            # from the root, if it is to be added
            if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
                mean_depth += 1
            heritable_missing_rate = 1 - ((1 - total_missing_proportion) / (1 - stochastic_missing_probability)) ** (
                1 / mean_depth
            )

        else:
            times = tree.get_times()
            mean_time = np.mean([times[l] for l in tree.leaves])
            if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
                mean_time += np.mean([tree.get_branch_length(u, v) for u, v in tree.edges])
            heritable_missing_rate = (
                -np.log((1 - total_missing_proportion) / (1 - stochastic_missing_probability)) / mean_time
            )

    if stochastic_missing_probability is None:
        if heritable_missing_rate < 0:
            raise ParameterEstimateError("Heritable missing data rate must be > 0.")
        if not continuous and heritable_missing_rate > 1:
            raise ParameterEstimateError("Per-generation heritable missing data rate must be < 1.")

        if not continuous:
            mean_depth = tree.get_mean_depth_of_tree()
            # We account for the added depth of the implicit branch leading
            # from the root, if it is to be added
            if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
                mean_depth += 1

            heritable_proportion = 1 - (1 - heritable_missing_rate) ** (mean_depth)

        else:
            times = tree.get_times()
            mean_time = np.mean([times[l] for l in tree.leaves])
            if assume_root_implicit_branch and len(tree.children(tree.root)) != 1:
                mean_time += np.mean([tree.get_branch_length(u, v) for u, v in tree.edges])

            heritable_proportion = 1 - np.exp(-heritable_missing_rate * mean_time)

        stochastic_missing_probability = (total_missing_proportion - heritable_proportion) / (1 - heritable_proportion)

    if stochastic_missing_probability < 0:
        raise ParameterEstimateWarning(
            "Estimate of the stochastic missing probability using this "
            "heritable rate resulted in a negative stochastic missing "
            "probability. It may be that this heritable rate is too high."
        )

    if heritable_missing_rate < 0:
        raise ParameterEstimateWarning(
            "Estimate of the heritable rate using this stochastic missing "
            "probability resulted in a negative heritable rate. It may be that "
            "this stochastic missing probability is too high."
        )

    return stochastic_missing_probability, heritable_missing_rate
