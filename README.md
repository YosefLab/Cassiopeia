
Single Cell Lineage Tracing
===

Single Cell Lineage Tracing 

* Free Software: MIT License

Installation
-----------

1. Clone the package as so: ``git clone https://github.com/YosefLab/SingleCellLineageTracing.git``

2. Ensure that you have python3.6 installed. You can install this via pip.

3. Make sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``
    
4. Make sure that Emboss is properly configurd and installed; oftentimes users may see a "command not found" error when attempting to align with the `align_sequences` function we have provided. This is most likely due to the fact that you have not properly added the binary file to your path variable. For details on how to download, configure, and install the Emboss package, refer to this [tutorial](http://emboss.open-bio.org/html/adm/ch01s01.html).

5. Install the package using the following commands:
    * ``python3.6 setup.py build``
    * ``python3.6 setup.py bdist_wheel``
    * ``python3.6 -m pip install . --user``
    
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



