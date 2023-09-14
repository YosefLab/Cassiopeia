<img src="https://github.com/YosefLab/cassiopeia/blob/master/docs/_static/logo.png?raw=true" width="400" alt="cassiopeia">

[![Stars](https://img.shields.io/github/stars/YosefLab/cassiopeia?logo=GitHub&color=yellow)](https://github.com/YosefLab/cassiopeia/stargazers)
[![Documentation Status](https://readthedocs.org/projects/cassiopeia/badge/?version=latest)](https://cassiopeia.readthedocs.io/en/stable/?badge=stable)
![Build
Status](https://github.com/YosefLab/cassiopeia/workflows/cassiopeia/badge.svg)
[![Coverage](https://codecov.io/gh/YosefLab/cassiopeia/branch/master/graph/badge.svg)](https://codecov.io/gh/YosefLab/cassiopeia)

Cassiopeia: A pipeline for single-cell lineage tracing data
=============================================================

Cassiopeia is an end-to-end pipeline for single-cell lineage tracing experiments.
The software contained here comes equipped with modules for processing sequencing reads,
reconstructing & plotting trees, analyzing these trees, and benchmarking new algorithms.

You can find all of our [documentation here](https://cassiopeia-lineage.readthedocs.io/en/latest/index.html).

We also have provided tutorials for three modules:

- [processing fastqs](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/preprocess.ipynb)
- [reconstructing trees](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/reconstruct.ipynb)
- [simulating trees and benchmarking](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/benchmark.ipynb)


You can also find our originally describing Cassiopeia published in [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8).

Free Software: MIT License

Installation
--------------

For users:

```
pip install git+https://github.com/YosefLab/Cassiopeia@master#egg=cassiopeia-lineage
```

For developers:

1. Clone the package as so: ``git clone https://github.com/YosefLab/Cassiopeia.git``

2. Ensure that you have Python >= 3.8 installed. (Due to dependencies no longer supporting Python 3.7, we have now stopped supporting Python <= 3.7) We prefer using [miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. [Optional] If you intend to use the CassiopeiaILP solver, please be sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``

4. [Optional] To use fast versions of Neighbor-Joining and UPGMA, install [CCPhylo](https://bitbucket.org/genomicepidemiology/ccphylo/src/master/) then set ccphylo_path in the config.ini file in the cassiopeia directory.

5. Install Cassiopeia by first changing into the Cassiopeia directory and then `pip3 install .`. To install dev and docs requirements, you can run `pip3 install .[dev,docs]`.

6. [Optional] To use tools built for the analysis of spatial lineage tracing datasets, you can install Cassiopeia with `pip install .[spatial]`. Please note that we recommend using Python >= 3.9 for these analyses as some features might not be available otherwise, due to package dependencies (especially 3D visualization).

To verify that it installed correctly, install `pytest` (`pip install pytest`) and try running our tests with `make test`.

Reference
----------------------

If you've found Cassiopeia useful for your research, please consider citing our paper published in Genome Biology:


Matthew G Jones*, Alex Khodaverdian*, Jeffrey J Quinn*, Michelle M Chan, Jeffrey A Hussmann, Robert Wang, Chenling Xu, Jonathan S Weissman, Nir Yosef. (2020), [*Inference of single-cell phylogenies from lineage tracing data using Cassiopeia*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8), Genome Biology

