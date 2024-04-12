==========
Plotting
==========

.. currentmodule:: cassiopeia

Plotting
~~~~~~~~~~~~~~~~~~~

Plotting functionality is divided into two broad categories: local and remote
(a.k.a. iTOL). Previously, we only supported tree visualization using the rich
iTOL framework. However, we are now in the process of deprecating the use of
this service for most use cases. We recommend all users to visualize their
trees using the local plotting functions, which either use Matplotlib or
Plotly, as this option is free and is more reminiscent of plotting in other
packages such as Scanpy.

.. autosummary::
   :toctree: reference/

   pl.labels_from_coordinates
   pl.plot_matplotlib
   pl.plot_plotly
   pl.Tree3D
   pl.upload_and_export_itol
