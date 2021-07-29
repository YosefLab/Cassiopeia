Cassiopeia: A pipeline for single-cell lineage tracing data
=============================================================

Cassiopeia is an end-to-end pipeline for single-cell lineage tracing experiments.
The software contained here comes equipped with modules for processing sequencing reads,
reconstructing & plotting trees, analyzing these trees, and benchmarking new algorithms.

You can find all of our [documentation here](https://cassiopeia-lineage.readthedocs.io/en/testdeployment/index.html).

We also have provided tutorials for three modules:

- [processing fastqs](https://github.com/YosefLab/Cassiopeia/blob/testdeployment/notebooks/preprocess.ipynb)
- [reconstructing trees](https://github.com/YosefLab/Cassiopeia/blob/testdeployment/notebooks/reconstruct.ipynb)
- [simulating trees and benchmarking](https://github.com/YosefLab/Cassiopeia/blob/testdeployment/notebooks/benchmark.ipynb)


You can also find our originally describing Cassiopeia published in [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8)) 

Free Software: MIT License

Installation
--------------

1. Clone the package as so: ``git clone https://github.com/YosefLab/Cassiopeia.git``

2. Ensure that you have python3.6 installed. You can install this via pip.

3. Make sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``
    
4. Install Cassiopeia by running `make install` from the directory where you have Cassiopeia installed.
    
To verify that it installed correctly, try running our tests with `make test`.

Reference
----------------------

If you've found Cassiopeia useful for your research, please consider citing our paper published in Genome Biology:


Matthew G Jones*, Alex Khodaverdian*, Jeffrey J Quinn*, Michelle M Chan, Jeffrey A Hussmann, Robert Wang, Chenling Xu, Jonatahn S Weissman, Nir Yosef. (2020), *Inference of single-cell phylogenies from lineage tracing data using Cassiopeia*, [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8)