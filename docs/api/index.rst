.. module:: cassiopeia
.. automodule:: cassiopeia
   :noindex:

===
API
===

Import cassiopeia as::

   import cassiopeia

Preprocessing: ``pp``
------------------------

.. module:: cassiopeia.pp
.. currentmodule:: cassiopeia

Filtering of cells and all main pipeline procedures.

.. autosummary::
   :toctree: reference/

   pp.align_sequences
   pp.call_alleles
   pp.collapse_umis
   pp.convert_alleletable_to_character_matrix
   pp.error_correct_umis
   pp.filter_cells
   pp.filter_umis
   pp.resolve_umi_sequence
   pp.filter_molecule_table
   pp.call_lineage_groups

Tree Inference: ``solver``
----------------------------

.. module:: cassiopeia.solver
.. currentmodule:: cassiopeia

Filtering of cells and all main pipeline procedures.

.. autosummary::
   :toctree: reference/
   
   solver.collapse_tree
   solver.MaxCutSolver
   solver.NeighborJoiningSolver
   solver.SpectralSolver
   solver.VanillaGreedySolver
