===========
Data
===========

.. module:: cassiopeia.data
.. currentmodule:: cassiopeia

CassiopeiaTrees
~~~~~~~~~~~~~~~~~~~

The main data structure that Cassiopeia uses for all tree-based analyses is the CassiopeiaTree:

.. autosummary::
   :toctree: reference/

   data.CassiopeiaTree

Utilities
~~~~~~~~~~~~~~~~~~~

We also have several utilities that are useful for working with various data related to phylogenetics:

.. autosummary::
   :toctree: reference/

   data.compute_dissimilarity_map
   data.compute_phylogenetic_weight_matrix
   data.get_lca_characters
   data.sample_bootstrap_allele_tables
   data.sample_bootstrap_character_matrices
   data.to_newick
