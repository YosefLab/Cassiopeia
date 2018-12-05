
Single Cell Lineage Tracing
===

Single Cell Lineage Tracing 

* Free Software: MIT License

Quick Start
-----------

1. Clone the package as so: ``git clone https://github.com/YosefLab/SingleCellLineageTracing.git``

2. Make sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``

3. Install the package using the following commands:
    * ``python setup.py build``
    * ``python setup.py install --user``

You can then load in the package to a python session with ``import SingleCellLineageTracing`` if you'd like to use the package from within an existing session. We also provide a command-line interface for reconstructing and post-processing the lineage groups. From command line, you can use ``reconstruct-lineage`` and ``post-process``. Use the ``-h`` flag for documentation (further documentation forthcoming).

