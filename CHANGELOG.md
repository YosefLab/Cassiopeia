# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [3.0.0] - (unreleased)

<!-- On release, replace "(unreleased)" with the release date as YYYY-MM-DD. -->

### Added

- **TreeData and a functional API.** Every module now works on
  [TreeData](https://treedata.readthedocs.io/en/latest/) objects (AnnData with trees in `obst`)
  through functions such as `cas.solver.nj(tdata)`, with shared arguments (`tree_key`,
  `characters_key`, `key_added`, `copy`) and parameters such as the missing and unmodified states
  stored in `tdata.uns`.
- **New preprocessing pipeline (`cas.pp`).** C++ kernels take sequencing reads to an allele table
  and a character matrix: target-site construct presets (Cassiopeia/KP-tracer, GESTALT, PEtracer,
  DNA Typewriter, BASELINE), alignment with `pp.align` or STARsolo/Cell Ranger BAMs, allele
  calling, QC filters with plots, clone calling, doublet detection, and `pp.run` /
  `cassiopeia-preprocess` to run it all.
- **Solvers.** `solver.nj`, `upgma`, `greedy`, `ilp` and `hybrid`, with fast Cython neighbor
  joining, UPGMA and greedy kernels (replacing CCPhylo), `solver.reroot` for rooting trees, the
  dissimilarities in a new `cas.dissimilarity` module (with `dissimilarity.pairwise` and the new
  `nonmissing_hamming` metric), and support for ambiguous (tuple) states.
- **Branch lengths and ancestral states.** `tl.estimate_branch_lengths` estimates branch lengths by
  maximum likelihood with [ConvexML](https://github.com/songlab-cal/ConvexML), replacing the
  `IIDExponential` estimators, and `tl.ancestral_characters` adds conservative and Sankoff
  parsimony. New tools for tree topologies and lineage tracing parameters include
  `tl.collapse_edges`, `tl.count_edge_mutations`, `tl.rescale_node_times` and
  `tl.calculate_likelihood`.
- **Simulation.** Functional simulators for trees (`sim.birth_death_process`, with fitness and
  per-division hooks), lineage tracing data for many technologies (`sim.stochastic_tracing`),
  missing data and sequencing errors (`sim.missing_data`, `sim.noise`), gene expression, spatial
  positions, and sampling of cells, spatial regions, capture spots and timepoints.
- **Priors** for character states are handled the same way everywhere: stored in
  `tdata.uns["priors"]`, keyed by character (column) name so they stay matched to their characters
  after subsetting, and selected with `priors=None|True|False|dict` in every function that uses
  them.
- **Spatial.** `sp.impute_alleles_spatial` imputes missing alleles from neighbors in a spatial graph
  stored in `tdata.obsp`.
- **Documentation and packaging.** New and rewritten tutorials (tree reconstruction, branch length
  estimation, choosing a solver, ecDNA and spatial simulation), and binary wheels for Linux and
  macOS on PyPI.

### Changed

- Plotting and downstream tree analysis moved to [pycea](https://pycea.readthedocs.io/en/latest/).
- The dissimilarity metrics moved from `cas.solver.dissimilarity_functions` to `cas.dissimilarity`
  and take `missing_state` instead of `missing_state_indicator` (the old keyword still works, with
  a warning). Renamed metrics (the old names remain in the deprecated
  `cas.solver.dissimilarity_functions`):
  - `weighted_hamming_distance` -> `weighted_hamming`
  - `hamming_distance` -> `hamming`
  - `cluster_dissimilarity_weighted_hamming_distance_min_linkage` -> `cluster_weighted_hamming`
- Renamed functions (the old names warn and call the new functions until 3.1):
  - `tl.get_proportion_of_missing_data` -> `tl.fraction_missing`
  - `tl.get_proportion_of_mutation` -> `tl.fraction_mutated`
  - `tl.estimate_missing_data_rates` -> `tl.estimate_missing_rates`
  - `tl.get_lineage_tracing_parameters` -> `tl.get_tracing_parameters`
  - `tl.calculate_likelihood_discrete` and `tl.calculate_likelihood_continuous` ->
    `tl.calculate_likelihood(model="discrete" | "continuous")`
  - `sim.fate_tree_expression` -> `sim.trajectory_expression`
  - `sp.impute_alleles_from_spatial_data` -> `sp.impute_alleles_spatial`
- Supports Python 3.10 to 3.14 and requires NumPy 2.
- Reconstructed trees name their internal nodes `root`, `node1`, `node2`, ... (in depth-first
  order, so the names are the same on every run) instead of `cassiopeia_internal_node...`.
- Empty (NaN) entries of a character matrix are treated as missing data.
- `tl.calculate_likelihood` works with string states (as produced by `cas.sim` and `cas.pp`) and
  takes the priors, states and rates as arguments; in the continuous model, it and the rate
  estimators read branch lengths from node times (`time_key`).

<!-- Add further API changes for 3.0.0 here. -->

### Deprecated

These warn when used and will be removed in 3.1:

- `CassiopeiaTree`: convert existing trees with `CassiopeiaTree.to_treedata()`.
- The class-based solvers `NeighborJoiningSolver`, `UPGMASolver`, `VanillaGreedySolver`,
  `ILPSolver` and `HybridSolver`: use `solver.nj`, `upgma`, `greedy`, `ilp` and `hybrid`.
- The class-based simulators: use the functions in `cas.sim`.
- `cas.solver.dissimilarity_functions` and its alias `cas.solver.dissimilarity`: use
  `cas.dissimilarity`.
- `data.get_lca_characters` and `data.newick_to_networkx`.
- The renamed functions listed above.

### Removed

- The MaxCut, MaxCutGreedy, Spectral, SpectralGreedy, SpectralNeighborJoining, Percolation and
  SharedMutationJoining solvers: their classes remain only as stubs that raise
  `NotImplementedError`. Use `solver.nj` or `solver.greedy`.
- The `cas.pl` plotting module (`plot_matplotlib`, `plot_plotly`, `upload_and_export_itol`,
  `labels_from_coordinates`, `Tree3D`): plot trees with pycea.
- The `tl.IIDExponentialBayesian` and `tl.IIDExponentialMLE` branch length estimators (use
  `tl.estimate_branch_lengths`), and the tools covered by pycea: `tl.FitnessEstimator`,
  `compute_morans_i`, `compute_evolutionary_coupling`, `compute_expansion_pvalues`,
  `compute_cophenetic_correlation`, `fitch_hartigan`, `fitch_count` and `score_small_parsimony`.
- The data utilities covered by pycea (`compute_phylogenetic_weight_matrix`,
  `net_relatedness_index`, `compute_inter_cluster_distances`), `data.to_newick` (use
  `treedata.io.write_nwk`) and the bootstrap samplers (`sample_bootstrap_character_matrices`,
  `sample_bootstrap_allele_tables`).
- `sim.ecDNABirthDeathSimulator`: use `sim.birth_death_process` with an `on_division` hook (see
  the ecDNA tutorial).
- The similarity metrics `hamming_similarity_without_missing`,
  `hamming_similarity_normalized_over_missing`, `weighted_hamming_similarity` and
  `exponential_negative_hamming_distance`, used by the removed solvers.
- `sp.get_spatial_graph_from_anndata`: build the spatial graph with
  `squidpy.gr.spatial_neighbors`.
- The pre-3.0 preprocessing pipeline and its functions, replaced by `cas.pp`.
- The CCPhylo dependency for fast neighbor joining.
