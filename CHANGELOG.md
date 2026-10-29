# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [3.0.0]

- **TreeData and a functional API.** Every module now works on
  [TreeData](https://treedata.readthedocs.io/en/latest/) objects (AnnData with trees in `obst`)
  through functions such as `cas.solver.nj(tdata)`, with shared arguments (`tree_key`,
  `characters_key`, `key_added`, `copy`) and parameters such as the missing and unmodified states
  stored in `tdata.uns`. `CassiopeiaTree` and the class-based solvers, simulators and tools are
  deprecated (they warn and delegate where possible) and will be removed in 3.1.
- **New preprocessing pipeline (`cas.pp`).** C++ kernels take sequencing reads to an allele table
  and a character matrix: target-site construct presets (Cassiopeia/KP-tracer, GESTALT, PEtracer,
  DNA Typewriter, BASELINE), alignment with `pp.align` or STARsolo/Cell Ranger BAMs, allele
  calling, QC filters with plots, clone calling, doublet detection, and `pp.run` /
  `cassiopeia-preprocess` to run it all. The pre-3.0 pipeline is removed.
- **Solvers.** `solver.nj`, `upgma`, `greedy`, `ilp` and `hybrid`, with fast Cython neighbor
  joining, UPGMA and greedy kernels (replacing CCPhylo), `solver.reroot` for rooting trees, the
  dissimilarities in a new `cas.dissimilarity` module, and support for ambiguous (tuple) states.
  The MaxCut, spectral, percolation and shared-mutation-joining solvers are deprecated.
- **Branch lengths and ancestral states.** `tl.estimate_branch_lengths` estimates branch lengths by
  maximum likelihood with [ConvexML](https://github.com/songlab-cal/ConvexML), replacing the
  `IIDExponential` estimators, and `tl.ancestral_characters` adds conservative and Sankoff
  parsimony. New tools for tree topologies and lineage tracing parameters include
  `tl.collapse_edges`, `tl.count_edge_mutations` and `tl.calculate_likelihood`.
- **Simulation.** Functional simulators for trees (`sim.birth_death_process`, with fitness and
  per-division hooks), lineage tracing data for many technologies (`sim.stochastic_tracing`),
  missing data and sequencing errors (`sim.missing_data`, `sim.noise`), gene expression, spatial
  positions, and sampling of cells, spatial regions, capture spots and timepoints.
- **Priors** for character states are handled the same way everywhere: stored in
  `tdata.uns["priors"]` and selected with `priors=None|True|False|dict` in every function that
  uses them.
- **Plotting and downstream analysis moved to [pycea](https://pycea.readthedocs.io/en/latest/).**
  The `cas.pl` plotting module, `FitnessEstimator` and the tree utilities covered by pycea are
  deprecated.
- **Spatial.** `sp.impute_alleles_spatial` imputes missing alleles from neighbors in a spatial graph
  stored in `tdata.obsp`.
- **Documentation and packaging.** New and rewritten tutorials (tree reconstruction, branch length
  estimation, choosing a solver, ecDNA and spatial simulation), and binary wheels for Linux and
  macOS on PyPI. Supports Python 3.10 to 3.14 and requires NumPy 2.
