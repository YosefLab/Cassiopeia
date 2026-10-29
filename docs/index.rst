==========
Cassiopeia
==========

Cassiopeia [Jones20]_ is a package for processing and analyzing lineage tracing data.
It provides an end-to-end pipeline for converting raw sequencing reads into character matrices and
reconstructing lineage trees, along with tools for simulating lineage tracing data and benchmarking
reconstruction methods. Cassiopeia is maintained by the
`Yosef Lab <https://yoseflab.github.io/>`_ at the Weizmann Institute.

Cassiopeia is part of the lineageverse ecosystem:

* `TreeData <https://treedata.readthedocs.io/en/latest/>`_, a data structure for storing lineage trees
  alongside cell-level data. Cassiopeia functions read and write TreeData objects.
* `Pycea <https://pycea.readthedocs.io/en/latest/>`_, a toolkit for visualizing and analyzing
  lineage trees stored in TreeData objects.

Cassiopeia (``import cassiopeia as cas``) is organized into the following modules:

* :doc:`cas.pp <api/preprocess>`: process sequencing reads into allele tables and character matrices.
* :doc:`cas.solver <api/solver>`: reconstruct trees with neighbor joining, UPGMA, greedy, ILP or hybrid
  solvers.
* :doc:`cas.dissimilarity <api/dissimilarity>`: dissimilarities between cells for the distance-based
  solvers.
* :doc:`cas.tl <api/tools>`: infer ancestral states and branch lengths, edit tree topologies and estimate
  lineage tracing parameters.
* :doc:`cas.sim <api/simulator>`: simulate trees, lineage tracing data, gene expression and spatial
  positions.
* :doc:`cas.sp <api/spatial>`: analyze spatial lineage tracing data.
* :doc:`cas.critique <api/critique>`: compare reconstructed trees with a ground truth.

If you find Cassiopeia useful for your research, please consider citing [Jones20]_.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`download;1.5em;sd-mr-1` Installation
      :link: installation
      :link-type: doc

      New to Cassiopeia? Check out the installation guide.

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Tutorials
      :link: user_guide
      :link-type: doc

      Learn to preprocess reads, reconstruct trees, simulate data and benchmark solvers.

   .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` API reference
      :link: api/index
      :link-type: doc

      A detailed description of every function in Cassiopeia.

   .. grid-item-card:: :octicon:`mark-github;1.5em;sd-mr-1` Questions & issues
      :link: https://github.com/YosefLab/Cassiopeia/issues

      Have a question or found a bug? File an issue on GitHub.


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   api/index
   user_guide
   contributing
   changelog
   references
