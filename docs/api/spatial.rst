==========
Spatial
==========

.. currentmodule:: cassiopeia

Spatial Analyses
~~~~~~~~~~~~~~~~~

This library stores code for analyzing lineage-tracing data from spatial
genomics approaches. While some relevant code is stored elsewhere (e.g.,
the `Spatial Simulators`) we have specific functions here.

Before calling :func:`sp.impute_alleles_spatial`, compute a spatial
connectivity graph with ``squidpy.gr.spatial_neighbors`` and store the
result in ``tdata.obsp``::

    import squidpy as sq
    sq.gr.spatial_neighbors(adata, key_added="spatial")
    tdata.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]

.. autosummary::
   :toctree: reference/

   sp.impute_alleles_spatial

Deprecated
~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   sp.impute_alleles_from_spatial_data
