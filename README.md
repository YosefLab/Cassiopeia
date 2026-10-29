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

We also provide tutorials:

- [preprocessing sequencing reads](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/preprocess.ipynb)
- [basic benchmarking pipeline](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/benchmark.ipynb)
- [basic tree reconstruction](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/reconstruct.ipynb)
- [gene expression simulation](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/expr_simulation.ipynb)
- [branch length estimation](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/branch_lengths.ipynb)
- [choosing the best solver](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/solvers.ipynb)
- [advanced simulation: ecDNA](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/simulate_ecDNA.ipynb)
- [spatially-resolved lineage tracing](https://github.com/YosefLab/Cassiopeia/blob/master/notebooks/spatial.ipynb)

Trees are stored in [TreeData](https://treedata.readthedocs.io/en/latest/) objects and can be plotted and analyzed with [pycea](https://pycea.readthedocs.io/en/latest/).


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

2. Ensure that you have Python >= 3.10 installed. (Due to dependencies and ongoing updates to Python, we will not guarantee support for python <= 3.9) We prefer using [miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. [Optional] If you intend to use the CassiopeiaILP solver, please be sure that Gurobi is installed. You can follow the instructions listed [here](http://www.gurobi.com/academia/for-universities). To verify that it's working correctly, use the following tests:
    * Run the command ``gurobi.sh`` from a terminal window
    * From the Gurobi installation directory (where there is a setup.py file), use ``python setup.py install --user``

4. Install Cassiopeia by first changing into the Cassiopeia directory and then `pip3 install .` or `make install`. To install dev and docs requirements, you can run `pip3 install .[dev,docs]`.

5. [Optional] To use tools built for the analysis of spatial lineage tracing datasets, you can install Cassiopeia with `pip install .[spatial]`. Please note that we recommend using Python >= 3.9 for these analyses as some features might not be available otherwise, due to package dependencies (especially 3D visualization).

6. [Optional] To estimate branch lengths with `cas.tl.estimate_branch_lengths`, install Cassiopeia with `pip install .[convexml]`, which adds [cvxpy](https://www.cvxpy.org/).

To verify that it installed correctly, install `pytest` (`pip install pytest`) and try running our tests with `make test`.

Reference
----------------------

If you've found Cassiopeia useful for your research, please consider citing our paper published in Genome Biology:


Matthew G Jones*, Alex Khodaverdian*, Jeffrey J Quinn*, Michelle M Chan, Jeffrey A Hussmann, Robert Wang, Chenling Xu, Jonathan S Weissman, Nir Yosef. (2020), [*Inference of single-cell phylogenies from lineage tracing data using Cassiopeia*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8), Genome Biology
