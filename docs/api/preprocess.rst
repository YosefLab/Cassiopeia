===========
Preprocess
===========
.. module:: cassiopeia.pp
.. currentmodule:: cassiopeia

Data Preprocessing
~~~~~~~~~~~~~~~~~~~

We have several functions that are part of our pipeline for processing sequencing data from single-cell lineage tracing technologies:

.. autosummary::
   :toctree: reference/
   
   pp.align_sequences
   pp.call_alleles
   pp.call_lineage_groups
   pp.collapse_umis
   pp.error_correct_umis
   pp.filter_molecule_table
   pp.filter_cells
   pp.filter_umis
   pp.resolve_umi_sequence
   
   


Data Utilities
~~~~~~~~~~~~~~~~~~~

We also have several functions that are useful for converting between data formats for downstream analyses:

.. autosummary::
   :toctree: reference/
   
   pp.compute_empirical_indel_priors
   pp.convert_alleletable_to_character_matrix
   pp.convert_alleletable_to_lineage_profile
   pp.convert_lineage_profile_to_character_matrix
