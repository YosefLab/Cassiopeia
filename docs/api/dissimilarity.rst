==============
Dissimilarity
==============
.. currentmodule:: cassiopeia

Pairwise dissimilarity metrics for comparing character states, and the
:func:`~cassiopeia.dissimilarity.pairwise` helper for computing and storing the
full pairwise dissimilarity map on a tree.

.. autosummary::
   :toctree: reference/

   dissimilarity.pairwise
   dissimilarity.weighted_hamming_distance
   dissimilarity.hamming_distance
   dissimilarity.hamming_similarity_without_missing
   dissimilarity.hamming_similarity_normalized_over_missing
   dissimilarity.weighted_hamming_similarity
   dissimilarity.exponential_negative_hamming_distance
   dissimilarity.cluster_dissimilarity
   dissimilarity.cluster_dissimilarity_weighted_hamming_distance_min_linkage
