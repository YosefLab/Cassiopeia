"""Top level for tools."""

from .ancestral_characters import ancestral_characters
from .branch_lengths import estimate_branch_lengths
from .deprecated import (
    FitnessEstimator,
    IIDExponentialBayesian,
    IIDExponentialMLE,
    calculate_likelihood_continuous,
    calculate_likelihood_discrete,
    compute_cophenetic_correlation,
    compute_evolutionary_coupling,
    compute_expansion_pvalues,
    compute_morans_i,
    estimate_missing_data_rates,
    fitch_count,
    fitch_hartigan,
    get_lineage_tracing_parameters,
    get_proportion_of_missing_data,
    get_proportion_of_mutation,
    score_small_parsimony,
)
from .parameter_estimators import (
    estimate_missing_rates,
    estimate_mutation_rate,
    fraction_missing,
    fraction_mutated,
)
from .topology import (
    collapse_edges,
    count_edge_mutations,
    get_leaves,
    get_root,
    mean_depth,
    rescale_node_times,
)
from .tree_metrics import (
    calculate_likelihood,
    calculate_parsimony,
    get_tracing_parameters,
)

__all__ = [
    # Branch lengths
    "ancestral_characters",
    "estimate_branch_lengths",
    "rescale_node_times",
    # Metrics
    "calculate_likelihood",
    "calculate_parsimony",
    "count_edge_mutations",
    "get_tracing_parameters",
    # Parameter estimation
    "estimate_missing_rates",
    "estimate_mutation_rate",
    "fraction_missing",
    "fraction_mutated",
    # Topology
    "collapse_edges",
    "get_leaves",
    "get_root",
    "mean_depth",
    # Deprecated
    "FitnessEstimator",
    "IIDExponentialBayesian",
    "IIDExponentialMLE",
    "calculate_likelihood_continuous",
    "calculate_likelihood_discrete",
    "compute_cophenetic_correlation",
    "compute_evolutionary_coupling",
    "compute_expansion_pvalues",
    "compute_morans_i",
    "estimate_missing_data_rates",
    "fitch_count",
    "fitch_hartigan",
    "get_lineage_tracing_parameters",
    "get_proportion_of_missing_data",
    "get_proportion_of_mutation",
    "score_small_parsimony",
]
