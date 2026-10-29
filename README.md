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

Cassiopeia requires Python >= 3.10. Install it from [PyPI](https://pypi.org/project/cassiopeia-lineage/):

```
pip install cassiopeia-lineage
```

Optional dependencies:

- Branch length estimation (`cas.tl.estimate_branch_lengths`): `pip install "cassiopeia-lineage[convexml]"`, which adds [cvxpy](https://www.cvxpy.org/).
- Spatial neighbor graphs for `cas.sp.impute_alleles_spatial`: `pip install "cassiopeia-lineage[spatial]"`, which adds [squidpy](https://squidpy.readthedocs.io/).
- The ILP solver (`cas.solver.ilp`, the default bottom solver of `cas.solver.hybrid`) requires [Gurobi](https://www.gurobi.com/): `pip install gurobipy` and a license, which is [free for academic users](https://www.gurobi.com/academia/academic-program-and-licenses/).

For developers, clone the repository, install it in editable mode with the test dependencies, and run the tests:

```
git clone https://github.com/YosefLab/Cassiopeia.git
cd Cassiopeia
pip install -e ".[test]"
pytest
```

Reference
----------------------

If you've found Cassiopeia useful for your research, please consider citing our paper published in Genome Biology:


Matthew G Jones*, Alex Khodaverdian*, Jeffrey J Quinn*, Michelle M Chan, Jeffrey A Hussmann, Robert Wang, Chenling Xu, Jonathan S Weissman, Nir Yosef. (2020), [*Inference of single-cell phylogenies from lineage tracing data using Cassiopeia*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02000-8), Genome Biology
