===========
Solver
===========
.. currentmodule:: cassiopeia

Solvers
~~~~~~~~~~~~~~~~~~~

Phylogeny reconstruction is performed with functional solvers that operate on a
:class:`treedata.TreeData` object in place (or return a copy when ``copy=True``),
storing the inferred tree in ``tdata.obst[key_added]``:

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
