==========
Cassiopeia
==========

Cassiopeia [Jones20]_ is a package for end-to-end analysis of single-cell lineage tracing data: from
sequencing reads to character matrices, tree reconstruction and branch length estimation, and
simulations for benchmarking new methods. It is maintained by the `Yosef Lab <https://yoseflab.github.io/>`_
at the Weizmann Institute.

Cassiopeia is part of an ecosystem of packages for lineage tracing:

* `TreeData <https://treedata.readthedocs.io/en/latest/>`_ stores trees together with cell-level data. It
  extends `AnnData <https://anndata.readthedocs.io/en/latest/>`_ with trees in ``obst``, and every
  Cassiopeia function reads and writes TreeData objects.
* `pycea <https://pycea.readthedocs.io/en/latest/>`_ plots trees and analyzes them downstream, for
  example to find clades, estimate fitness or test whether cell states are heritable.

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
