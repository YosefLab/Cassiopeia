
Cassiopeia
============

This is a software suite for proecessing data from single cell lineage tracing experiments. This suite comes equipped with three main modules:

- **Target Site Sequencing Pipeline**: a pipeline for extracing lineage information from raw fastqs produced from a lineage tracing experiment.
- **Phylogeny Reconstruction**: a collection of tools for constructing phylogenies. We support 5 algorithms currently: a greedy algorithm based on multi-state compatibility, an exact Steiner-Tree solver, Cassiopeia (the combination of these two), Neighbor-Joining, and Camin-Sokal Maximum Parsimony. 
- **Benchmarking**: a set of tools for benchmarking; a simulation framework and tree comparsion tools. 

You can find all documentation [here](https://cassiopeia-lineage.readthedocs.io/en/latest/readme.html)

You can also find example notebooks in this repository:

- [processing fastqs](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/process_fastq.ipynb)
- [reconstructing trees](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/reconstruct_lineages.ipynb)
- [simulating trees and stress testing](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/simulate_and_stress_test.ipynb)

Free Software: MIT License

Installation
--------------

1. Clone the package as so: ``git clone https://github.com/YosefLab/Cassiopeia.git``

2. Ensure that you have python3.6 installed. You can install this via pip.

3. Make sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``
    
4. Make sure that Emboss is properly configurd and installed; oftentimes users may see a "command not found" error when attempting to align with the `align_sequences` function we have provided. This is most likely due to the fact that you have not properly added the binary file to your path variable. For details on how to download, configure, and install the Emboss package, refer to this [tutorial](http://emboss.open-bio.org/html/adm/ch01s01.html).

5. One of Cassiopeia's dependencies, pysam, requires HTSLib to be installed. You can read about pysam's requirements [here](https://pysam.readthedocs.io/en/latest/installation.html#requirements).

6. Ensure the Cython is installed. You can do this via ``python3.6 pip install --user cython``. 

7. While we get pip working, it's best to first clone the package and then follow these instructions:
    * ``python3.6 setup.py build``
    * ``python3.6 setup.py bdist_wheel``
    * ``python3.6 setup.py build_ext --inplace``
    * ``python3.6 -m pip install . --user``
    
    
To verify that it installed correctly, try using the package in a python session: ``import cassiopeia``. Then, to make sure that the command-line tools work, try ``reconstruct-lineage -h`` and confirm that you get the usage details.

Command Line Tools
-------------------

In addition to allowing users to use Cassiopeia from a python session, we provide five unique command line tools for common pipeline procedures:

- `reconstruct-lineage`: Reconstructs a lineage from a provided character matrix (consisting of cells x characters where each element is the observed state of that character in that cell).
- `post-process-tree`: Post-process trees after reconstructing to assign sample identities back to leaves of the tree and removing any leaves that don't correspond to a sample in the character matrix.
- `stress-test`: Conduct stress testing on a given simulated tree. Writes out a new tree file after inferring a tree from the unique leaves of the "true", simulated tree.
- `call-lineages`: Perform lineage group calling from a molecule table.
- `filter-molecule-table`: Perform molecule table filtering. 

All usage details can be found by using the `-h` flag. 
