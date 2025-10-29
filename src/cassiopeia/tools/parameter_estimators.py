"""Utilities for estimating lineage tracing parameters."""

import warnings

import numpy as np
from treedata import TreeData

from cassiopeia import utils
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning


def get_proportion_of_missing_data(
    tree: CassiopeiaTree | TreeData,
    characters_key: str = "characters",
    missing_state=None,
    **kwargs,
) -> float:
    """Calculate the proportion of missing entries in the character matrix.

    Calculates the proportion of cell/character entries in the character matrix
    that have a missing state, with the missing state being indicated by
    the tree's missing_state_indicator.

    Args:
        tree: CassiopeiaTree or TreeData object containing the character matrix
        characters_key: Key for the character matrix. For CassiopeiaTree, if "characters",
            uses the default character_matrix attribute; otherwise looks in layers.
            For TreeData, specifies the obsm key. Default is "characters".
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a list of values. If None, uses the tree's missing_state_indicator
            attribute, or defaults to [-1, "-1", "NA", "-"]. Default is None.
        **kwargs: Deprecated arguments. Use 'characters_key' instead of 'layer'.

    Returns:
        Proportion of missing cell/character entries (between 0 and 1)

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        characters_key = kwargs.pop("layer")

    character_matrix = utils._get_character_matrix(tree, characters_key)
    missing_state_indicator = utils._get_missing_state_indicator(tree, missing_state)

    num_dropped = utils._count_entries(character_matrix, missing_state_indicator)
    missing_proportion = num_dropped / (character_matrix.shape[0] * character_matrix.shape[1])
    return missing_proportion


def get_proportion_of_mutation(
    tree: CassiopeiaTree | TreeData,
    characters_key: str = "characters",
    missing_state=None,
    unmodified_state=None,
    **kwargs,
) -> float:
    """Calculate the proportion of mutated entries in the character matrix.

    Calculates the proportion of cell/character entries in the character matrix
    that have a non-uncut (non-0) state, normalizing over non-missing entries.
    Hence, missing entries are not considered in calculating the proportion.

    Args:
        tree: CassiopeiaTree or TreeData object containing the character matrix
        characters_key: Key for the character matrix. For CassiopeiaTree, if "characters",
            uses the default character_matrix attribute; otherwise looks in layers.
            For TreeData, specifies the obsm key. Default is "characters".
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a list of values. If None, uses the tree's missing_state_indicator
            attribute, or defaults to [-1, "-1", "NA", "-"]. Default is None.
        unmodified_state: Value(s) to consider as unmodified/uncut states. Can be
            a single value or a list of values. If None, defaults to [0, "0", "*"]
            for flexibility with both integer and string character matrices.
            Default is None.
        **kwargs: Deprecated arguments. Use 'characters_key' instead of 'layer'.

    Returns:
        Proportion of non-missing cell/character entries that are mutated (between 0 and 1)

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        characters_key = kwargs.pop("layer")

    character_matrix = utils._get_character_matrix(tree, characters_key)
    missing_state_indicator = utils._get_missing_state_indicator(tree, missing_state)

    if unmodified_state is None:
        unmodified_state = [0, "0", "*"]

    num_dropped = utils._count_entries(character_matrix, missing_state_indicator)
    num_unmodified = utils._count_entries(character_matrix, unmodified_state)

    num_mut = character_matrix.shape[0] * character_matrix.shape[1] - num_dropped - num_unmodified
    mutation_proportion = num_mut / (
        character_matrix.shape[0] * character_matrix.shape[1] - num_dropped
    )
    return mutation_proportion


def estimate_mutation_rate(
    tree: CassiopeiaTree | TreeData,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    characters_key: str = "characters",
    depth_key: str = "depth",
    tree_key: str = "tree",
    missing_state=None,
    unmodified_state=None,
    **kwargs,
) -> float:
    """Calculate the proportion of mutated entries in the character matrix.

    Calculates the proportion of cell/character entries in the character matrix
    that have a non-uncut (non-0) state, normalizing over non-missing entries.
    Hence, missing entries are not considered in calculating the proportion.

    Args:
        tree: CassiopeiaTree or TreeData object containing tree topology and character matrix
        continuous: If True, calculate a continuous mutation rate accounting for branch
            lengths. If False, calculate a discrete mutation rate using node depths.
            Default is True.
        assume_root_implicit_branch: If True, assume an implicit branch leading from
            the root if it doesn't exist (i.e., if root has multiple children). This
            branch is added to the total time when calculating the estimate. Default is True.
        characters_key: Key for the character matrix. For CassiopeiaTree, if "characters",
            uses the default character_matrix attribute; otherwise looks in layers.
            For TreeData, specifies the obsm key. Default is "characters".
        depth_key: Node attribute key containing depth values (e.g., "depth" for
            generation count, "time" for evolutionary time). Default is "depth".
        tree_key: Tree key to use if tree is a TreeData object with multiple trees.
            Only required if multiple trees are present. Default is "tree".
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a list of values. If None, uses the tree's missing_state_indicator
            attribute, or defaults to [-1, "-1", "NA", "-"]. Default is None.
        unmodified_state: Value(s) to consider as unmodified/uncut states. Can be
            a single value or a list of values. If None, defaults to [0, "0", "*"]
            for flexibility with both integer and string character matrices.
            Default is None.
        **kwargs: Deprecated arguments. Use 'characters_key' instead of 'layer'.

    Returns:
        Proportion of non-missing cell/character entries that are mutated (between 0 and 1)

    Warns:
        UserWarning: If continuous=True but branch lengths are integers, suggesting
            a mismatch between the continuous parameter and discrete branch lengths.

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        characters_key = kwargs.pop("layer")

    t, _ = utils._get_digraph(tree, tree_key=tree_key)
    mutation_proportion = utils._get_tree_parameter(tree, "mutation_proportion")
    if mutation_proportion is None:
        mutation_proportion = get_proportion_of_mutation(
            tree, characters_key, missing_state, unmodified_state
        )

    if mutation_proportion < 0 or mutation_proportion > 1:
        raise ParameterEstimateError("Mutation proportion must be between 0 and 1.")

    edges = list(t.edges())

    utils._check_continuous_not_int(t, edges, continuous)

    root = utils.get_root(tree, tree_key=tree_key)
    mean_depth = utils.get_mean_depth(tree, depth_key, tree_key=tree_key)

    if assume_root_implicit_branch and t.out_degree(root) != 1:
        mean_depth += (
            1 if not continuous else np.mean(np.mean([t[u][v]["length"] for u, v in edges]))
        )

    if not continuous:
        mutation_rate = 1 - (1 - mutation_proportion) ** (1 / mean_depth)
    else:
        mutation_rate = -np.log(1 - mutation_proportion) / mean_depth

    return mutation_rate


def estimate_missing_data_rates(
    tree: CassiopeiaTree | TreeData,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    stochastic_missing_probability: float | None = None,
    heritable_missing_rate: float | None = None,
    characters_key: str = "characters",
    depth_key: str = "depth",
    tree_key: str = "tree",
    missing_state=None,
    **kwargs,
) -> tuple[float, float]:
    """Estimates both missing data parameters given one of the two from a tree.

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

    Since the two types of data are convolved, we need to know the contribution of one
    type to estimate the other. This function attempts to retrieve the heritable missing
    rate and stochastic missing probability from tree parameters, or they may be provided
    as function arguments. Exactly one of these parameters must be provided; if neither
    or both are provided, the function raises an error.

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
        tree: CassiopeiaTree or TreeData object containing tree topology and character matrix
        continuous: If True, calculate a continuous missing rate accounting for branch
            lengths. If False, calculate a discrete missing rate based on the number of
            generations. Default is True.
        assume_root_implicit_branch: If True, assume an implicit branch leading from
            the root if it doesn't exist (i.e., if root has multiple children). This
            branch is added to the total time when calculating the estimate. Default is True.
        stochastic_missing_probability: The stochastic missing probability. Will override
            the value stored in tree parameters if provided. Observed probabilities of
            stochastic missing data typically range between 10-20%. Default is None.
        heritable_missing_rate: The heritable missing rate. Will override the value
            stored in tree parameters if provided. Default is None.
        characters_key: Key for the character matrix. For CassiopeiaTree, if "characters",
            uses the default character_matrix attribute; otherwise looks in layers.
            For TreeData, specifies the obsm key. Default is "characters".
        depth_key: Node attribute key containing depth values (e.g., "depth" for
            generation count, "time" for evolutionary time). Default is "depth".
        tree_key: Tree key to use if tree is a TreeData object with multiple trees.
            Only required if multiple trees are present. Default is "tree".
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a list of values. If None, uses the tree's missing_state_indicator
            attribute, or defaults to [-1, "-1", "NA", "-"]. Default is None.
        unmodified_state: Value(s) to consider as unmodified/uncut states. Can be
            a single value or a list of values. If None, defaults to [0, "0", "*"]
            for flexibility with both integer and string character matrices.
            Default is None.
        **kwargs: Deprecated arguments. Use 'characters_key' instead of 'layer'.

    Warns:
        UserWarning: If continuous=True but branch lengths are integers, suggesting
            a mismatch between the continuous parameter and discrete branch lengths.

    Raises:
        ParameterEstimateError: If the total missing proportion is not between 0 and 1,
            if stochastic missing probability or heritable missing rate have invalid values,
            or if both or neither of these parameters are provided.
        ParameterEstimateWarning: If the estimated parameter is negative, suggesting
            that the provided parameter may be too high.
    """
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        characters_key = kwargs.pop("layer")

    t, _ = utils._get_digraph(tree, tree_key=tree_key)
    total_missing_proportion = utils._get_tree_parameter(tree, "missing_proportion")
    if total_missing_proportion is None:
        total_missing_proportion = get_proportion_of_missing_data(
            tree, characters_key, missing_state
        )

    if total_missing_proportion < 0 or total_missing_proportion > 1:
        raise ParameterEstimateError("Missing proportion must be between 0 and 1.")

    if stochastic_missing_probability is None:
        stochastic_missing_probability = utils._get_tree_parameter(
            tree, "stochastic_missing_probability"
        )

    if heritable_missing_rate is None:
        heritable_missing_rate = utils._get_tree_parameter(tree, "heritable_missing_rate")

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

    edges = list(t.edges())

    utils._check_continuous_not_int(t, edges, continuous)

    root = utils.get_root(tree, tree_key=tree_key)
    mean_depth = utils.get_mean_depth(tree, depth_key, tree_key=tree_key)

    if heritable_missing_rate is None:
        if stochastic_missing_probability < 0:
            raise ParameterEstimateError("Stochastic missing data rate must be > 0.")
        if stochastic_missing_probability > 1:
            raise ParameterEstimateError("Stochastic missing data rate must be < 1.")

        mean_depth = utils.get_mean_depth(tree, depth_key, tree_key=tree_key)

        if assume_root_implicit_branch and t.out_degree(root) != 1:
            if not continuous:
                mean_depth += 1
            else:
                mean_depth += np.mean([t[u][v]["length"] for u, v in edges])

        if not continuous:
            heritable_missing_rate = 1 - (
                (1 - total_missing_proportion) / (1 - stochastic_missing_probability)
            ) ** (1 / mean_depth)
        else:
            heritable_missing_rate = (
                -np.log((1 - total_missing_proportion) / (1 - stochastic_missing_probability))
                / mean_depth
            )

    if stochastic_missing_probability is None:
        if heritable_missing_rate < 0:
            raise ParameterEstimateError("Heritable missing data rate must be > 0.")
        if not continuous and heritable_missing_rate > 1:
            raise ParameterEstimateError("Per-generation heritable missing data rate must be < 1.")

        mean_depth = utils.get_mean_depth(tree, depth_key, tree_key=tree_key)

        if assume_root_implicit_branch and t.out_degree(root) != 1:
            if not continuous:
                mean_depth += 1
            else:
                mean_depth += np.mean([t[u][v]["length"] for u, v in edges])

        if not continuous:
            heritable_proportion = 1 - (1 - heritable_missing_rate) ** mean_depth
        else:
            heritable_proportion = 1 - np.exp(-heritable_missing_rate * mean_depth)

        stochastic_missing_probability = (total_missing_proportion - heritable_proportion) / (
            1 - heritable_proportion
        )

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
