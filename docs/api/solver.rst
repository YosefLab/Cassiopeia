===========
Solver
===========
.. module:: cassiopeia.solver
.. currentmodule:: cassiopeia

CassiopeiaSolvers
~~~~~~~~~~~~~~~~~~~

We have several algorithms available for solving phylogenies:

.. autosummary::
   :toctree: reference/
   
   solver.HybridSolver
   solver.ILPSolver
   solver.MaxCutSolver
   solver.MaxCutGreedySolver
   solver.NeighborJoiningSolver
   solver.PercolationSolver
   solver.SharedMutationJoiningSolver
   solver.SpectralSolver
   solver.SpectralGreedySolver
   solver.UPGMASolver
   solver.VanillaGreedySolver


Dissimilarity Maps
~~~~~~~~~~~~~~~~~~~

For use in our distance-based solver and for comparing character states, we also have available several dissimilarity functions:

.. autosummary::
   :toctree: reference/
   
   solver.dissimilarity_functions.cluster_dissimilarity
   solver.dissimilarity_functions.hamming_distance
   solver.dissimilarity_functions.hamming_similarity_normalized_over_missing
   solver.dissimilarity_functions.hamming_similarity_without_missing
   solver.dissimilarity_functions.weighted_hamming_distance
   solver.dissimilarity_functions.weighted_hamming_similarity