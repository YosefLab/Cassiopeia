===========
Data
===========

.. module:: cassiopeia.data
.. currentmodule:: cassiopeia

CassiopeiaTree
~~~~~~~~~~~~~~~~~~~

.. deprecated:: 3.0.0
   :class:`~cassiopeia.data.CassiopeiaTree` is deprecated in favor of
   :class:`treedata.TreeData` and will be removed in v3.1.0. All Cassiopeia
   functions operate on :class:`~treedata.TreeData`; convert existing objects
   with :meth:`CassiopeiaTree.to_treedata() <cassiopeia.data.CassiopeiaTree.to_treedata>`.

.. autosummary::
   :toctree: reference/

   data.CassiopeiaTree
