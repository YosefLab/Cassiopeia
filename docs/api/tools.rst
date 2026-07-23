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
   tl.calculate_cPHS

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   tl.estimate_missing_rates
   tl.estimate_mutation_rate
   tl.fraction_missing
   tl.fraction_mutated
   tl.get_tracing_parameters

Topology
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: reference/

   tl.collapse_edges
   tl.get_leaves
   tl.get_root
   tl.mean_depth
   tl.rescale_node_times

Ancestral Characters
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: reference/

   tl.ancestral_characters
