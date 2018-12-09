
Single Cell Lineage Tracing
===

Single Cell Lineage Tracing 

* Free Software: MIT License

Installation
-----------

1. Clone the package as so: ``git clone https://github.com/YosefLab/SingleCellLineageTracing.git``

2. Make sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``

3. Install the package using the following commands:
    * ``python setup.py build``
    * ``python setup.py bdist_wheel``
    * ``python -m pip install . --user``
    
To verify that it installed correctly, try using the package in a python session: ``import SingleCellLineageTracing``. Then, to make sure that the command-line tools work, try ``reconstruct-lineage -h`` and confirm that you get the usage details.

Simulation Trees
-----------

To simulate trees, use the ``simulate-tree`` command line tool.

Reconstruct Trees
-----------------

To reconstruct trees, use the ``reconstruct-lineage`` command line tool.

Post Processing Trees
---------------------

To post process trees, use the ``post-process-tree`` command line tool.

Stress Test
-----------

To run a stress test, use the ``stress-test`` command line tool.

Lineage Group Assignments
-------------------------

To run the lineage group caller, use the ``call-lineages`` command line tool.

Filtering Molecule Tables
-------------------------

To run the filter molecule table pipeline, use the ``filter-molecule-table`` command line tool.



