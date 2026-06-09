=========
Simulator
=========
.. currentmodule:: cassiopeia

Cassiopeia provides a composable, functional API for simulating cell lineages.
A typical pipeline builds a tree topology, overlays lineage tracing data,
simulates data dropout, and then subsamples leaves:

.. code-block:: python

   import cassiopeia as cas

   tdata = cas.sim.complete_binary(num_cells=256)
   cas.sim.stochastic_tracing(tdata, number_of_cassettes=10, size_of_cassette=3)
   cas.sim.missing_data(tdata, stochastic_missing_rate = 0.1)
   tdata = cas.sim.sample_uniform(tdata, ratio=0.5)


Tree Topology Simulators
~~~~~~~~~~~~~~~~~~~~~~~~

These functions simulate tree topologies and return a
:class:`treedata.TreeData` object with the tree stored in ``obst[tree_key]``.

.. autosummary::
   :toctree: reference/

   sim.complete_binary
   sim.birth_death_process
   sim.simple_fit_subclone


Lineage Tracing Simulators
~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions overlay simulated lineage tracing data onto an existing
topology.

.. autosummary::
   :toctree: reference/

   sim.stochastic_tracing
   sim.missing_data
   sim.noise


Expression Simulators
~~~~~~~~~~~~~~~~~~~~~~

These functions overlay simulated gene-expression data onto an existing
topology, optionally sampling counts from a Poisson or negative-binomial
observation model.

.. autosummary::
   :toctree: reference/

   sim.brownian_expression
   sim.trajectory_expression


Spatial Simulators
~~~~~~~~~~~~~~~~~~

These functions add spatial coordinates to an existing topology.

.. autosummary::
   :toctree: reference/

   sim.brownian_spatial
   sim.clonal_spatial


Leaf Subsampling
~~~~~~~~~~~~~~~~

Utilities for subsampling leaves, e.g. for benchmarking or to mimic
incomplete capture.

.. autosummary::
   :toctree: reference/

   sim.sample_uniform
   sim.sample_spatial
   sim.sample_supercellular


Deprecated
~~~~~~~~~~

The class-based API from v2 is retained for backwards compatibility but will
be removed in a future release. Use the functional API above instead.

.. autosummary::
   :toctree: reference/

   sim.BirthDeathFitnessSimulator
   sim.CompleteBinarySimulator
   sim.SimpleFitSubcloneSimulator
   sim.Cas9LineageTracingDataSimulator
   sim.BrownianSpatialDataSimulator
   sim.ClonalSpatialDataSimulator
   sim.UniformLeafSubsampler
   sim.SpatialLeafSubsampler
   sim.SupercellularSampler
   sim.fate_tree_expression
