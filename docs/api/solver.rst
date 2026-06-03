===========
Solver
===========
.. currentmodule:: cassiopeia

Solvers
~~~~~~~~~~~~~~~~~~~

Phylogeny reconstruction is performed with functional solvers that operate on a
:class:`treedata.TreeData` object in place, storing the inferred tree in
``tdata.obst[tree_key]``:

.. autosummary::
   :toctree: reference/

   solver.nj
   solver.upgma
   solver.greedy
   solver.ilp
   solver.hybrid

Rooting
~~~~~~~~~~~~~~~~~~~

Trees can be (re)rooted with a choice of procedures (``outgroup``, ``midpoint``,
``centroid``, ``shared_mutation``):

.. autosummary::
   :toctree: reference/

   solver.reroot

Deprecated solver classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The object-oriented solver classes are deprecated in favor of the functional API
above and are retained only for backward compatibility. ``NeighborJoiningSolver``,
``UPGMASolver``, ``VanillaGreedySolver``, ``ILPSolver``, and ``HybridSolver`` warn
on use and delegate to the corresponding function. The remaining classes
(``MaxCutSolver``, ``MaxCutGreedySolver``, ``SpectralSolver``,
``SpectralGreedySolver``, ``SharedMutationJoiningSolver``, ``PercolationSolver``,
``SpectralNeighborJoiningSolver``) have been removed and raise on ``solve``.
