==========
Tools
==========

.. module:: cassiopeia.tl
.. currentmodule:: cassiopeia

This library stores code for post-reconstruction analysis of trees. We are
always in the process of developing new statistics and tools for helping us
interpret trees, and adding them to this library.

Autocorrelation
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   tl.compute_morans_i

Branch Length Estimation (BLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/
      
   tl.IIDExponentialBayesian
   tl.IIDExponentialMLE

Coupling
~~~~~~~~~~~

.. autosummary::
   :toctree: reference/
   
   tl.compute_evolutionary_coupling

Metrics
~~~~~~~~
.. autosummary::
   :toctree: reference/
      
   tl.calculate_likelihood_continuous
   tl.calculate_likelihood_discrete
   tl.calculate_parsimony

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/
      
   tl.estimate_missing_data_rates
   tl.estimate_mutation_rate


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