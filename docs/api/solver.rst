===========
Solver
===========
.. module:: cassiopeia.solver
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
.. module:: cassiopeia.solver.rooting
.. currentmodule:: cassiopeia

:func:`~cassiopeia.solver.reroot` re-roots a reconstructed tree, and :func:`~cassiopeia.solver.nj`
roots the tree it builds (``root=``), with one of the procedures in :mod:`cassiopeia.solver.rooting`:

``"centroid"``
    Root on the edge that most evenly splits the leaves (the default of ``reroot``).
``"midpoint"``
    Root at the midpoint of the longest leaf-to-leaf path, using branch lengths.
``"outgroup"``
    Root on the branch to the outgroup leaf named by ``outgroup=``. Without ``outgroup``,
    :func:`~cassiopeia.solver.nj` adds a synthetic unedited leaf and roots the tree there.
``"shared_mutation"``
    Root on the edge that best separates the cells carrying the most common mutation.

.. autosummary::
   :toctree: reference/

   solver.reroot

Keyword arguments of ``reroot`` (for example ``outgroup``) are passed on to the procedure.
New procedures can be added with :func:`~cassiopeia.solver.rooting.register`.

.. autosummary::
   :toctree: reference/

   solver.rooting.centroid
   solver.rooting.midpoint
   solver.rooting.outgroup
   solver.rooting.shared_mutation
   solver.rooting.register
