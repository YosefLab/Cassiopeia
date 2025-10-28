===========
Simulator
===========
.. currentmodule:: cassiopeia


Our simulators for cassiopeia are split up into those that simulate topologies and those that simulate data on top of the topologies.

Tree Simulators
~~~~~~~~~~~~~~~~~~~

We have several frameworks available for simulating topologies:

.. autosummary::
   :toctree: reference/

   sim.BirthDeathFitnessSimulator
   sim.ecDNABirthDeathSimulator
   sim.CompleteBinarySimulator
   sim.SimpleFitSubcloneSimulator


Data Simulators
~~~~~~~~~~~~~~~~~~~

These simulators are subclasses of the `DataSimulator` class and implement the `overlay_data` method which simulates data according to a given topology.

.. autosummary::
   :toctree: reference/

   sim.Cas9LineageTracingDataSimulator


Spatial Simulators
~~~~~~~~~~~~~~~~~~~
These simulators are subclasses of the `SpatialSimulator` class and implement the `overlay_data` method which adds spatial coordinates to a given topology. `SpatialSimulator`s are a special sublcass of `DataSimulator` and can be used in addition to other `DataSimulator`s that simulate lineage tracing data.

.. autosummary::
   :toctree: reference/

   sim.BrownianSpatialDataSimulator
   sim.ClonalSpatialDataSimulator


Leaf SubSamplers
~~~~~~~~~~~~~~~~~~~
These are utilities for subsampling lineages for benchmarking purposes. For example, sampling a random proportion of leaves or grouping together cells into clades to model spatial data.

.. autosummary::
   :toctree: reference/

   sim.SupercellularSampler
   sim.SpatialLeafSubsampler
   sim.UniformLeafSubsampler
