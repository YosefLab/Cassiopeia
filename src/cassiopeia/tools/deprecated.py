import warnings


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"{old}() is deprecated and will be removed in a future release; use {new}() instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def get_proportion_of_missing_data(*args, **kwargs):
    """Deprecated alias for :func:`cassiopeia.tl.fraction_missing`."""
    _warn_renamed("get_proportion_of_missing_data", "fraction_missing")
    from .parameter_estimators import fraction_missing

    kwargs.setdefault("key_added", None)
    return fraction_missing(*args, **kwargs)


def get_proportion_of_mutation(*args, **kwargs):
    """Deprecated alias for :func:`cassiopeia.tl.fraction_mutated`."""
    _warn_renamed("get_proportion_of_mutation", "fraction_mutated")
    from .parameter_estimators import fraction_mutated

    kwargs.setdefault("key_added", None)
    return fraction_mutated(*args, **kwargs)


def estimate_missing_data_rates(*args, **kwargs):
    """Deprecated alias for :func:`cassiopeia.tl.estimate_missing_rates`."""
    _warn_renamed("estimate_missing_data_rates", "estimate_missing_rates")
    from .parameter_estimators import estimate_missing_rates

    return estimate_missing_rates(*args, **kwargs)


def get_lineage_tracing_parameters(
    tree,
    continuous,
    assume_root_implicit_branch,
    layer="characters",
    depth_key="time",
    **kwargs,
):
    """Deprecated alias for :func:`cassiopeia.tl.get_tracing_parameters`."""
    _warn_renamed("get_lineage_tracing_parameters", "get_tracing_parameters")
    from .tree_metrics import get_tracing_parameters

    return get_tracing_parameters(
        tree,
        continuous,
        assume_root_implicit_branch,
        characters_key=layer,
        time_key=depth_key,
        **kwargs,
    )


def calculate_likelihood_discrete(
    tree, use_internal_character_states=False, layer="characters", depth_key="time", **kwargs
):
    """Deprecated alias for ``calculate_likelihood(..., model="discrete")``."""
    _warn_renamed("calculate_likelihood_discrete", "calculate_likelihood")
    from .tree_metrics import calculate_likelihood

    return calculate_likelihood(
        tree,
        model="discrete",
        use_internal_character_states=use_internal_character_states,
        characters_key=layer,
        time_key=depth_key,
        **kwargs,
    )


def calculate_likelihood_continuous(
    tree, use_internal_character_states=False, layer="characters", depth_key="time", **kwargs
):
    """Deprecated alias for ``calculate_likelihood(..., model="continuous")``."""
    _warn_renamed("calculate_likelihood_continuous", "calculate_likelihood")
    from .tree_metrics import calculate_likelihood

    return calculate_likelihood(
        tree,
        model="continuous",
        use_internal_character_states=use_internal_character_states,
        characters_key=layer,
        time_key=depth_key,
        **kwargs,
    )


def plot_matplotlib(*args, **kwargs):
    """Placeholder for the removed :func:`plot_matplotlib` API."""
    raise NotImplementedError(
        "plot_matplotlib() was removed in v3.0.0. Use Pycea plotting functions instead: https://pycea.readthedocs.io/"
    )


def plot_plotly(*args, **kwargs):
    """Placeholder for the removed :func:`plot_plotly` API."""
    raise NotImplementedError(
        "plot_plotly() was removed in v3.0.0. Use Pycea plotting functions instead: https://pycea.readthedocs.io/"
    )


def upload_and_export_itol(*args, **kwargs):
    """Placeholder for the removed :func:`upload_and_export_itol` API."""
    raise NotImplementedError(
        "upload_and_export_itol() was removed in v3.0.0. "
        "Use Pycea plotting functions instead: https://pycea.readthedocs.io/"
    )


def labels_from_coordinates(*args, **kwargs):
    """Placeholder for the removed :func:`labels_from_coordinates` API."""
    raise NotImplementedError(
        "labels_from_coordinates() was removed in v3.0.0. "
        "Use Pycea plotting functions instead: https://pycea.readthedocs.io/"
    )


class IIDExponentialBayesian:
    """Deprecated stub retained for backwards compatibility."""

    def __init__(*args, **kwargs):
        raise NotImplementedError(
            "IIDExponentialBayesian() was removed in v3.0.0. "
            "Use ConvexML instead: https://github.com/songlab-cal/ConvexML"
        )


class IIDExponentialMLE:
    """Deprecated stub retained for backwards compatibility."""

    def __init__(*args, **kwargs):
        raise NotImplementedError(
            "IIDExponentialMLE() was removed in v3.0.0. Use ConvexML instead: https://github.com/songlab-cal/ConvexML"
        )


class FitnessEstimator:
    """Deprecated stub retained for backwards compatibility."""

    def __init__(*args, **kwargs):
        raise NotImplementedError(
            "FitnessEstimator() was removed in v3.0.0. Use pycea.tl.fitness instead: https://pycea.readthedocs.io/"
        )


def compute_evolutionary_coupling(*args, **kwargs):
    """Placeholder for the removed :func:`compute_evolutionary_coupling` API."""
    raise NotImplementedError(
        "compute_evolutionary_coupling() was removed in v3.0.0. "
        "Use pycea.tl.ancestral_linkage instead instead: https://pycea.readthedocs.io/"
    )


def compute_morans_i(*args, **kwargs):
    """Placeholder for the removed :func:`compute_morans_i` API."""
    raise NotImplementedError(
        "compute_morans_i() was removed in v3.0.0. "
        "Use pycea.tl.autocorr instead instead: https://pycea.readthedocs.io/"
    )


def compute_expansion_pvalues(*args, **kwargs):
    """Placeholder for the removed :func:`compute_expansion_pvalues` API."""
    raise NotImplementedError(
        "compute_expansion_pvalues() was removed in v3.0.0. "
        "Use pycea.tl.expansion_test instead: https://pycea.readthedocs.io/"
    )


def compute_cophenetic_correlation(*args, **kwargs):
    """Placeholder for the removed :func:`compute_cophenetic_correlation` API."""
    raise NotImplementedError(
        "compute_cophenetic_correlation() was removed in v3.0.0. "
        "Use pycea.tl.compare_distance instead: https://pycea.readthedocs.io/"
    )


def fitch_hartigan(*args, **kwargs):
    """Placeholder for the removed :func:`fitch_hartigan` API."""
    raise NotImplementedError(
        "fitch_hartigan() was removed in v3.0.0. "
        "Use pycea.tl.ancestral_states instead: https://pycea.readthedocs.io/"
    )


def score_small_parsimony(*args, **kwargs):
    """Placeholder for the removed :func:`score_small_parsimony` API."""
    raise NotImplementedError(
        "score_small_parsimony() was removed in v3.0.0. "
        "Use pycea.tl.parsimony_score instead: https://pycea.readthedocs.io/"
    )


def fitch_count(*args, **kwargs):
    """Placeholder for the removed :func:`fitch_count` API."""
    raise NotImplementedError(
        "fitch_count() was removed in v3.0.0. "
        "Use pycea.tl.transition_rates instead: https://pycea.readthedocs.io/"
    )
