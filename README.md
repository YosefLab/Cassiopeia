<img src="https://github.com/YosefLab/cassiopeia/blob/master/docs/_static/logo.png?raw=true" width="400" alt="cassiopeia">

[![Stars](https://img.shields.io/github/stars/YosefLab/Cassiopeia?logo=GitHub&color=yellow)](https://github.com/YosefLab/Cassiopeia/stargazers)
[![Documentation Status](https://readthedocs.org/projects/cassiopeia-lineage/badge/?version=stable)](https://cassiopeia-lineage.readthedocs.io/en/stable/)
[![Tests](https://github.com/YosefLab/Cassiopeia/actions/workflows/test.yaml/badge.svg)](https://github.com/YosefLab/Cassiopeia/actions/workflows/test.yaml)
[![Coverage](https://codecov.io/gh/YosefLab/Cassiopeia/graph/badge.svg)](https://codecov.io/gh/YosefLab/Cassiopeia)

Cassiopeia: A pipeline for single-cell lineage tracing data
=============================================================

Cassiopeia is an end-to-end pipeline for single-cell lineage tracing experiments.
It provides modules for processing sequencing reads into character matrices, reconstructing and
analyzing lineage trees, simulating lineage tracing data, and benchmarking new algorithms.

You can find all of our [documentation here](https://cassiopeia-lineage.readthedocs.io/en/stable/).

We also provide tutorials:

- [preprocessing sequencing reads](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/preprocess.html)
- [basic benchmarking pipeline](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/benchmark.html)
- [basic tree reconstruction](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/reconstruct.html)
- [gene expression simulation](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/expr_simulation.html)
- [branch length estimation](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/branch_lengths.html)
- [choosing the best solver](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/solvers.html)
- [advanced simulation: ecDNA](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/simulate_ecDNA.html)
- [spatially-resolved lineage tracing](https://cassiopeia-lineage.readthedocs.io/en/stable/notebooks/spatial.html)

Trees are stored in [TreeData](https://treedata.readthedocs.io/en/latest/) objects and can be plotted and analyzed with [pycea](https://pycea.readthedocs.io/en/latest/).

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

For developers, clone the repository, install it in editable mode with the test dependencies, and run the tests.
Building from source compiles C++ extensions and needs a C++20 compiler and zlib:

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
