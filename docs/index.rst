.. Cassiopeia documentation master file, created by
   sphinx-quickstart on Sat Jan 26 12:35:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================
Welcome!
========================

This website serves as documentation to the Cassiopeia software suite, maintained by the `Yosef Lab
<https://yoseflab.github.io/>`_ at the Weizmann Institute.

Cassiopeia [Jones20]_ is a package for end-to-end phylogenetic reconstruction of single-cell lineage tracing data. The package is composed of the following modules:

* ``preprocess`` for processing sequencing FASTQ data to character matrices.
* ``solver`` for performing tree inference.
* ``simulator`` for simulating trees and character-level data.
* ``plotting`` for plotting trees.
* ``spatial`` for performing analyses with spatial data.
* ``tools`` for analyzing phylogenies, for example with paired RNA-seq data.

If you find this useful for your research, please consider citing Cassiopeia [Jones20]_.

.. raw:: html

    <div class="container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <div class="card text-center intro-card shadow">
                <img src="_static/computer-24px.svg" class="card-img-top" alt="installation with cassiopeia action icon" height="52">
                <div class="card-body flex-fill">
                    <h5 class="card-title">Installation</h5>
                    <p class="card-text">New to <em>Cassiopeia</em>? Check out the installation guide.
                    </p>

.. container:: custom-button

    :doc:`To the installation guide<installation>`

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <div class="card text-center intro-card shadow">
                <img src="_static/play_circle_outline-24px.svg" class="card-img-top" alt="cassiopeia user guide action icon" height="52">
                <div class="card-body flex-fill">
                    <h5 class="card-title">User guide</h5>
                    <p class="card-text">The tutorials provide in-depth information on running Cassiopeia.</p>

.. container:: custom-button

    :doc:`To the user guide<user_guide>`

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <div class="card text-center intro-card shadow">
                <img src="_static/library_books-24px.svg" class="card-img-top" alt="api of scvi action icon" height="52">
                <div class="card-body flex-fill">
                    <h5 class="card-title">API reference</h5>
                    <p class="card-text">The API reference contains a detailed description of
                    the Cassiopeia API.</p>

.. container:: custom-button

    :doc:`To the API reference<api/index>`

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <div class="card text-center intro-card shadow">
                <img src="_static/question-mark-svgrepo-com.svg" class="card-img-top" alt="questions about cassiopeia" height="52">
                <div class="card-body flex-fill">
                    <h5 class="card-title">Questions & Issues</h5>
                    <p class="card-text">Have a question or found a bug? File an issue.</p>

.. container:: custom-button

    `File an issue <https://github.com/YosefLab/Cassiopeia/issues>`_

.. raw:: html

                </div>
                </div>
            </div>
        </div>
    </div>


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   api/index
   user_guide
   contributing
   changelog
   references
