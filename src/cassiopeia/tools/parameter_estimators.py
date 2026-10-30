"""Utilities for estimating lineage tracing parameters."""

import warnings
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia import utils
from cassiopeia.mixins import ParameterEstimateError, ParameterEstimateWarning
from cassiopeia.tools.topology import mean_depth as _mean_depth


def _resolve_layer(characters_key: str, kwargs: dict) -> str:
    """Map the deprecated ``layer`` keyword onto ``characters_key``.

    Raises:
        TypeError: If *kwargs* contains anything other than ``layer``.
    """
    unexpected = set(kwargs) - {"layer"}
    if unexpected:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(sorted(unexpected))}")
    if "layer" not in kwargs:
        return characters_key
    warnings.warn(
        "'layer' is deprecated and will be removed in a future release; use "
        "'characters_key' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return kwargs["layer"]


def fraction_missing(
    tdata: TreeData,
    characters_key: str = "characters",
    missing_state: str | int | Sequence[str | int] | None = None,
    key_added: str | None = "fraction_missing",
    **kwargs,
) -> float:
    """Calculate the fraction of missing entries in the character matrix.

    Computes, per cell, the fraction of its character entries that have a
    missing state and (when ``key_added`` is given) stores this per-cell value
    in ``tdata.obs[key_added]``. Returns the overall pooled fraction of missing
    cell/character entries across the whole matrix.

    Args:
        tdata: TreeData object containing the character matrix.
        characters_key: The ``obsm`` key for the character matrix.
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a sequence of values. If not provided, uses ``tdata.uns['missing_state']``,
            or defaults to (-1, "-1", "NA", "-").
        key_added: Column in ``tdata.obs`` under which to store the per-cell
            missing fraction. Pass ``None`` to skip writing per-cell values
            (e.g. when called internally).
        **kwargs: Deprecated ``layer`` alias for ``characters_key``.

    Returns:
        Overall fraction of missing cell/character entries (between 0 and 1).

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    characters_key = _resolve_layer(characters_key, kwargs)
    character_matrix = utils._get_characters(tdata, characters_key)
    missing_state_indicator = utils._get_parameter(tdata, "missing_state", value=missing_state)

    missing_mask = _entry_mask(character_matrix, missing_state_indicator)
    n_characters = character_matrix.shape[1]

    if key_added is not None:
        per_cell = missing_mask.sum(axis=1) / n_characters
        utils._get_cell_meta(tdata)[key_added] = pd.Series(per_cell, index=character_matrix.index)

    num_dropped = int(missing_mask.sum())
    return num_dropped / (character_matrix.shape[0] * n_characters)


def fraction_mutated(
    tdata: TreeData,
    characters_key: str = "characters",
    missing_state: str | int | Sequence[str | int] | None = None,
    unmodified_state: str | int | Sequence[str | int] | None = None,
    key_added: str | None = "fraction_mutated",
    **kwargs,
) -> float:
    """Calculate the fraction of mutated entries in the character matrix.

    Computes, per cell, the fraction of its non-missing character entries that
    have a non-unmodified (mutated) state and (when ``key_added`` is given)
    stores this per-cell value in ``tdata.obs[key_added]``. Returns the overall
    pooled fraction of mutated entries, normalizing over non-missing entries
    (missing entries are not considered).

    Args:
        tdata: TreeData object containing the character matrix.
        characters_key: The ``obsm`` key for the character matrix.
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a sequence of values. If not provided, uses ``tdata.uns['missing_state']``,
            or defaults to (-1, "-1", "NA", "-").
        unmodified_state: Value(s) to consider as unmodified/uncut states. Can be
            a single value or a sequence of values. If not provided, uses
            ``tdata.uns['unmodified_state']``, or defaults to (0, "0", "*").
        key_added: Column in ``tdata.obs`` under which to store the per-cell
            mutated fraction. Pass ``None`` to skip writing per-cell values
            (e.g. when called internally).
        **kwargs: Deprecated ``layer`` alias for ``characters_key``.

    Returns:
        Overall fraction of non-missing cell/character entries that are mutated
        (between 0 and 1).

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    characters_key = _resolve_layer(characters_key, kwargs)
    character_matrix = utils._get_characters(tdata, characters_key)
    missing_state_indicator = utils._get_parameter(tdata, "missing_state", value=missing_state)
    unmodified_state_indicator = utils._get_parameter(
        tdata, "unmodified_state", value=unmodified_state
    )

    missing_mask = _entry_mask(character_matrix, missing_state_indicator)
    unmodified_mask = _entry_mask(character_matrix, unmodified_state_indicator)
    n_characters = character_matrix.shape[1]

    if key_added is not None:
        n_nonmissing_row = n_characters - missing_mask.sum(axis=1)
        n_mutated_row = n_nonmissing_row - unmodified_mask.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell = np.where(n_nonmissing_row > 0, n_mutated_row / n_nonmissing_row, 0.0)
        utils._get_cell_meta(tdata)[key_added] = pd.Series(per_cell, index=character_matrix.index)

    num_dropped = int(missing_mask.sum())
    num_unmodified = int(unmodified_mask.sum())
    n_total = character_matrix.shape[0] * n_characters
    num_mut = n_total - num_dropped - num_unmodified
    return num_mut / (n_total - num_dropped)


def estimate_mutation_rate(
    tdata: TreeData,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    characters_key: str = "characters",
    depth_key: str | None = None,
    time_key: str = "time",
    tree_key: str | None = None,
    missing_state: str | int | Sequence[str | int] | None = None,
    unmodified_state: str | int | Sequence[str | int] | None = None,
    **kwargs,
) -> float:
    """Estimate the mutation rate of a tree from its observed mutations.

    Estimates the mutation rate from the fraction of mutated (non-missing,
    non-unmodified) entries in the character matrix and the mean depth
    (``continuous=False``) or time (``continuous=True``) of the leaves, either as
    a per-generation (discrete) or instantaneous (continuous) rate.

    Args:
        tdata: TreeData object containing tree topology and character matrix.
        continuous: If True, calculate a continuous mutation rate from node times
            (``time_key``). If False, calculate a discrete mutation rate from node
            depths (``depth_key``). Default is True.
        assume_root_implicit_branch: If True, assume an implicit branch leading from
            the root if it doesn't exist (i.e., if root has multiple children). This
            branch is added to the total time when calculating the estimate. Default is True.
        characters_key: The ``obsm`` key for the character matrix. Default is "characters".
        depth_key: (``continuous=False``) Node attribute holding the number of
            generations from the root. If ``None`` (default), it is the number of
            edges from the root.
        time_key: (``continuous=True``) Node attribute holding node times; branch
            lengths are the differences between them. Default is "time".
        tree_key: The ``obst`` key of the tree to use if ``tdata`` contains multiple
            trees.
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a sequence of values. If not provided, uses ``tdata.uns['missing_state']``,
            or defaults to (-1, "-1", "NA", "-").
        unmodified_state: Value(s) to consider as unmodified/uncut states. Can be
            a single value or a sequence of values. If not provided, uses
            ``tdata.uns['unmodified_state']``, or defaults to (0, "0", "*").
        **kwargs: Deprecated ``layer`` alias for ``characters_key``.

    Returns:
        The estimated mutation rate.

    Warns:
        UserWarning: If continuous=True but branch lengths are integers, suggesting
            a mismatch between the continuous parameter and discrete branch lengths.

    Raises:
        ParameterEstimateError: If character matrix or layer doesn't exist
    """
    characters_key = _resolve_layer(characters_key, kwargs)
    t, _ = utils._get_digraph(tdata, tree_key=tree_key)
    mutation_proportion = utils._get_parameter(tdata, "mutation_proportion")
    if mutation_proportion is None:
        mutation_proportion = fraction_mutated(
            tdata, characters_key, missing_state, unmodified_state, key_added=None
        )

    if mutation_proportion < 0 or mutation_proportion > 1:
        raise ParameterEstimateError("Mutation proportion must be between 0 and 1.")

    mean_depth = _tree_depth(t, continuous, depth_key, time_key, assume_root_implicit_branch)

    if not continuous:
        mutation_rate = 1 - (1 - mutation_proportion) ** (1 / mean_depth)
    else:
        mutation_rate = -np.log(1 - mutation_proportion) / mean_depth

    return mutation_rate


def estimate_missing_rates(
    tdata: TreeData,
    continuous: bool = True,
    assume_root_implicit_branch: bool = True,
    stochastic_missing_probability: float | None = None,
    heritable_missing_rate: float | None = None,
    characters_key: str = "characters",
    depth_key: str | None = None,
    time_key: str = "time",
    tree_key: str | None = None,
    missing_state: str | int | Sequence[str | int] | None = None,
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
    missing proportion) as `missing_proportion` in `tdata.uns`, inferring
    it using `fraction_missing` if it is not populated.

    Since the two types of data are convolved, we need to know the contribution of one
    type to estimate the other. This function attempts to retrieve the heritable missing
    rate and stochastic missing probability from ``tdata.uns``, or they may be provided
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
        tdata: TreeData object containing tree topology and character matrix
        continuous: If True, calculate a continuous missing rate from node times
            (``time_key``). If False, calculate a discrete missing rate from the
            number of generations (``depth_key``). Default is True.
        assume_root_implicit_branch: If True, assume an implicit branch leading from
            the root if it doesn't exist (i.e., if root has multiple children). This
            branch is added to the total time when calculating the estimate. Default is True.
        stochastic_missing_probability: The stochastic missing probability. Will override
            the value stored in ``tdata.uns`` if provided. Observed probabilities of
            stochastic missing data typically range between 10-20%. Default is None.
        heritable_missing_rate: The heritable missing rate. Will override the value
            stored in ``tdata.uns`` if provided. Default is None.
        characters_key: The ``obsm`` key for the character matrix. Default is "characters".
        depth_key: (``continuous=False``) Node attribute holding the number of
            generations from the root. If ``None`` (default), it is the number of
            edges from the root.
        time_key: (``continuous=True``) Node attribute holding node times; branch
            lengths are the differences between them. Default is "time".
        tree_key: The ``obst`` key of the tree to use if ``tdata`` contains multiple
            trees.
        missing_state: Value(s) to consider as missing data. Can be a single value
            or a sequence of values. If not provided, uses ``tdata.uns['missing_state']``,
            or defaults to (-1, "-1", "NA", "-").
        **kwargs: Deprecated ``layer`` alias for ``characters_key``.

    Returns:
        The stochastic missing probability and the heritable missing rate.

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
    characters_key = _resolve_layer(characters_key, kwargs)
    t, _ = utils._get_digraph(tdata, tree_key=tree_key)
    total_missing_proportion = utils._get_parameter(tdata, "missing_proportion")
    if total_missing_proportion is None:
        total_missing_proportion = fraction_missing(
            tdata, characters_key, missing_state, key_added=None
        )

    if total_missing_proportion < 0 or total_missing_proportion > 1:
        raise ParameterEstimateError("Missing proportion must be between 0 and 1.")

    if stochastic_missing_probability is None:
        stochastic_missing_probability = utils._get_parameter(
            tdata, "stochastic_missing_probability"
        )

    if heritable_missing_rate is None:
        heritable_missing_rate = utils._get_parameter(tdata, "heritable_missing_rate")

    if heritable_missing_rate is None and stochastic_missing_probability is None:
        raise ParameterEstimateError(
            "Neither `heritable_missing_rate` nor "
            "`stochastic_missing_probability` were provided as arguments or "
            "found in `tdata.uns`. Please provide one of these "
            "parameters, otherwise they are convolved and cannot be estimated"
        )

    if heritable_missing_rate is not None and stochastic_missing_probability is not None:
        raise ParameterEstimateError(
            "Both `heritable_missing_rate` and `stochastic_missing_probability`"
            " were provided as parameters or found in `tdata.uns`. "
            "Please only supply one of the two"
        )

    mean_depth = _tree_depth(t, continuous, depth_key, time_key, assume_root_implicit_branch)

    if heritable_missing_rate is None:
        if stochastic_missing_probability < 0:
            raise ParameterEstimateError("Stochastic missing data rate must be > 0.")
        if stochastic_missing_probability > 1:
            raise ParameterEstimateError("Stochastic missing data rate must be < 1.")

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


def _entry_mask(character_matrix: pd.DataFrame, indicator) -> np.ndarray:
    """Boolean ndarray mask of entries matching ``indicator`` (scalar or sequence)."""
    if not isinstance(indicator, (list, tuple, set)):
        return (character_matrix == indicator).values
    if pd.api.types.is_integer_dtype(character_matrix.values.dtype):
        indicator = [x for x in indicator if isinstance(x, (int, np.integer))]
    else:
        indicator = [str(x) for x in indicator]
    if not indicator:
        return np.zeros(character_matrix.shape, dtype=bool)
    return np.isin(character_matrix.values, indicator)


def _tree_depth(
    t: nx.DiGraph,
    continuous: bool,
    depth_key: str | None,
    time_key: str,
    assume_root_implicit_branch: bool,
) -> float:
    """Mean depth (generations) or time of the leaves below the root.

    With ``assume_root_implicit_branch`` and a root with several children, an
    implicit branch above the root adds one generation, or the mean branch
    length.
    """
    root = utils._get_root(t)
    if continuous:
        missing = [n for n in t.nodes if time_key not in t.nodes[n]]
        if missing:
            raise ParameterEstimateError(
                f"continuous=True needs node times in the {time_key!r} node attribute, "
                f"which {len(missing)} node(s) lack (e.g. {missing[0]!r}). Pass time_key=, "
                "estimate them with cas.tl.estimate_branch_lengths, or use continuous=False."
            )
        lengths = [t.nodes[v][time_key] - t.nodes[u][time_key] for u, v in t.edges]
        _check_continuous_not_int(lengths, continuous)
        leaves = [n for n in t.nodes if t.out_degree(n) == 0]
        depth = float(np.mean([t.nodes[n][time_key] for n in leaves])) - t.nodes[root][time_key]
        implicit = float(np.mean(lengths)) if lengths else 0.0
    else:
        depth = _mean_depth(t, depth_key)
        implicit = 1
    if assume_root_implicit_branch and t.out_degree(root) != 1:
        depth += implicit
    return depth


def _check_continuous_not_int(lengths: Sequence[float], continuous: bool = True) -> None:
    """Warn if continuous=True but branch lengths are discrete integers."""
    if not len(lengths):
        return

    if continuous and float(lengths[0]).is_integer():
        warnings.warn(
            "continuous=True with discrete branches may produce incorrect estimates. "
            "Consider using continuous=False",
            UserWarning,
            stacklevel=3,
        )
