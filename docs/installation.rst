Installation
============

Cassiopeia requires Python 3.10 or later. Install it from `PyPI <https://pypi.org/project/cassiopeia-lineage/>`_:

.. code-block:: bash

    pip install cassiopeia-lineage

Optional dependencies
---------------------

Some features need additional packages:

* Branch length estimation (:func:`cassiopeia.tl.estimate_branch_lengths`) uses `cvxpy <https://www.cvxpy.org/>`_:

  .. code-block:: bash

      pip install "cassiopeia-lineage[convexml]"

* Building spatial neighbor graphs for :func:`cassiopeia.sp.impute_alleles_spatial` uses
  `squidpy <https://squidpy.readthedocs.io/en/stable/>`_:

  .. code-block:: bash

      pip install "cassiopeia-lineage[spatial]"

* The ILP solver (:func:`cassiopeia.solver.ilp`), which is also the default bottom solver of
  :func:`cassiopeia.solver.hybrid`, requires `Gurobi <https://www.gurobi.com/>`_: install it with
  ``pip install gurobipy`` and obtain a license, which is
  `free for academic users <https://www.gurobi.com/academia/academic-program-and-licenses/>`_.

* The tutorials use `pycea <https://pycea.readthedocs.io/en/latest/>`_ (Python 3.11 or later) for plotting and
  downstream analysis:

  .. code-block:: bash

      pip install pycea-lineage

Development version
-------------------

To install the latest development version from GitHub (building from source compiles C++
extensions and needs a C++20 compiler and zlib):

.. code-block:: bash

    pip install git+https://github.com/YosefLab/Cassiopeia.git

To contribute, clone the repository, install it in editable mode with the test dependencies and run the
tests (see the :doc:`contributing guide<contributing>`):

.. code-block:: bash

    git clone https://github.com/YosefLab/Cassiopeia.git
    cd Cassiopeia
    pip install -e ".[test]"
    pytest
