"""Deprecated module. Re-exports :mod:`cassiopeia.data.deprecated` for backward compatibility."""

from cassiopeia.data.deprecated import (  # noqa: F401
    compute_inter_cluster_distances,
    compute_phylogenetic_weight_matrix,
    get_lca_characters,
    net_relatedness_index,
    newick_to_networkx,
    sample_bootstrap_allele_tables,
    sample_bootstrap_character_matrices,
    to_newick,
)
