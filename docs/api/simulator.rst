===========
Simulator
===========
.. module:: cassiopeia.sim
.. currentmodule:: cassiopeia


Our simulators for cassiopeia are split up into those that simulate topologies and those that simulate data on top of the topologies.

Tree Simulators
~~~~~~~~~~~~~~~~~~~

We have several frameworks available for simulating topologies:

.. autosummary::
   :toctree: reference/
   
   sim.BirthDeathFitnessSimulator
   sim.SimpleFitSubcloneSimulator


Data Simulators
~~~~~~~~~~~~~~~~~~~

These simulators are subclasses of the `DataSimulator` class and implement the `overlay_data` method which simulates data according to a given topology.

.. autosummary::
   :toctree: reference/
