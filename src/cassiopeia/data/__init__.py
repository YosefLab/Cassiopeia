"""Top level for data.

:class:`CassiopeiaTree` and the utilities in this module are deprecated; use
:class:`~treedata.TreeData` instead.
"""

from cassiopeia.dissimilarity._compute import compute_dissimilarity_map

from .CassiopeiaTree import CassiopeiaTree
from .deprecated import (
    compute_inter_cluster_distances,
    compute_phylogenetic_weight_matrix,
    get_lca_characters,
    net_relatedness_index,
    newick_to_networkx,
    sample_bootstrap_allele_tables,
    sample_bootstrap_character_matrices,
    to_newick,
)
