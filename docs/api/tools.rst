==========
Tools
==========

.. currentmodule:: cassiopeia

This library stores code for post-reconstruction analysis of trees. We are
always in the process of developing new statistics and tools for helping us
interpret trees, and adding them to this library.

Metrics
~~~~~~~~
.. autosummary::
   :toctree: reference/

   tl.calculate_likelihood
   tl.calculate_parsimony
   tl.count_edge_mutations
   tl.get_tracing_parameters

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   tl.estimate_missing_rates
   tl.estimate_mutation_rate
   tl.fraction_missing
   tl.fraction_mutated


Small-Parsimony
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   tl.fitch_count
   tl.fitch_hartigan
   tl.score_small_parsimony

Topology
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: reference/

   tl.compute_expansion_pvalues
   tl.collapse_edges
   tl.mean_depth

Ancestral Characters
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: reference/

   tl.ancestral_characters
