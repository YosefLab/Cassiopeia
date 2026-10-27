# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unrealeased]

### Added

- Added a ``cassiopeia.utils`` module with helpers for retrieving leaves and roots
  and collapsing unifurcations across multiple tree representations.
- Added pytest coverage for the new utility helpers.
- New preprocessing pipeline (`cassiopeia.pp`) with C++ kernels:
  - Declarative target-site constructs (`pp.Construct`) with presets for the
    Weissman lab Cas9 target site (`weissman`; Cassiopeia/KP-tracer), GESTALT,
    PEtracer, DNA Typewriter and BASELINE, and four edit models (`indel`,
    `insertion_lookup`, `sequential_insertion`, `substitution`).
  - `pp.build_reference` writes FASTA/GTF references and STARsolo / Cell Ranger
    commands; target-site BAMs from STARsolo or Cell Ranger can be used directly.
  - `pp.align`: FASTQ -> deduplicated, Smith-Waterman aligned, CB/UB-tagged BAM
    in one streaming pass (cell barcode and UMI correction included). Follows
    the pre-3.0 UMI collapse, including its banded-alignment merge of reads with
    sequencing indels, and the pre-3.0 alignment scoring. With
    `pp.call_alleles(legacy=True, umi_max_dist=2)` it reproduces the pre-3.0
    allele calls and read counts on simulated reads. On published data
    (KP-tracer, PEtracer) read counts match for 92-98% of molecules; most of the
    rest keep more reads, because the old collapse drops reads from UMIs in
    which no read sequence repeats. About 30x faster than the old pipeline.
  - Read layouts are preset names (`10xv3`, `dropseq`, `slideseq2`, ...) or
    pattern strings with one token per FASTQ, e.g. `16C12U,T` (`C` cell
    barcode, `U` UMI, `X` skip, `T` target site); `pp.build_reference`
    derives the matching STAR / STARsolo arguments from them.
  - `pp.call_alleles`: streaming BAM -> molecule table (pandas) that reads only
    the target-site contigs of indexed BAMs of any size.
  - Molecule-table steps that annotate rather than drop, each with a built-in
    QC plot: `pp.correct_intbcs`, `pp.filter_molecules`, `pp.filter_cells`,
    `pp.call_clones` (NovaClone, single clone or user-supplied),
    `pp.filter_integrations` and `pp.filter_doublets` (allele conflicts, clone
    mixtures, UMI outliers, custom scores).
  - `pp.alleles_to_characters` returns a TreeData ready for the solvers, with
    states written as strings (`"*"` unedited, `"-"` missing, `"1"`, `"2"`,
    ...), integers, or the alleles themselves (`mapping=`).
  - `pp.run` / `cassiopeia-preprocess run config.yaml` run the full pipeline.
  - `cassiopeia.preprocess.datasets` fetches small subsets of published
    datasets for testing.

### Changed

- Removed the pre-3.0 preprocessing pipeline (`pp.convert_fastqs_to_unmapped_bam`,
  `pp.filter_bam`, `pp.error_correct_cellbcs_to_whitelist`, `pp.collapse_umis`,
  `pp.resolve_umi_sequence`, `pp.align_sequences`, `pp.call_alleles`,
  `pp.error_correct_intbcs_to_whitelist`, `pp.error_correct_umis`,
  `pp.filter_molecule_table`, `pp.call_lineage_groups`, `pp.filter_cells`,
  `pp.filter_umis` and the `.cfg`-based `cassiopeia-preprocess`) in favor of the
  new pipeline above; `pp.call_alleles` and `pp.filter_cells` now refer to the
  new functions. The legacy allele-table converters
  (`pp.convert_alleletable_to_character_matrix`,
  `pp.convert_alleletable_to_lineage_profile`,
  `pp.convert_lineage_profile_to_character_matrix`,
  `pp.convert_character_matrix_to_allele_table`) are removed;
  `pp.compute_empirical_indel_priors` now also accepts the molecule table. `ngs-tools`, `hits` and `pyseq-align` are no longer
  dependencies; `pysam>=0.22` and `pyarrow` are.
- Priors stored in `tdata.uns["priors"]` with string keys (as written by h5ad/h5td)
  are converted back to integers by the solvers.
- Pairwise dissimilarities with priors encode string character matrices (e.g.
  `"*"` / `"-"` / `"1"`) to integers, so they use the fast numba path.
- `cassiopeia.mixins.logger` no longer depends on `ngs-tools`.

- Removed plotting module. Use [Pycea](https://pycea.readthedocs.io/en/latest/index.html) for all plotting needs.
- Removed `IIDExponentialBayesian` and `IIDExponentialBayesian` branch length estimation classes. The latest version of these are part of the [ConvexML](https://github.com/songlab-cal/ConvexML) package.
- Removed  `FitnessEstimator` fitness estimation class. This is now implemented by [pycea.tl.fitness](https://pycea.readthedocs.io/en/latest/generated/pycea.tl.fitness.html#pycea.tl.fitness)
- Updated the Robinson-Foulds implementation to rely on the shared utilities
  while continuing to accept mixed tree types and support string keys via
  ``TreeData`` without changing the public API.
- Updated :meth:`cassiopeia.data.CassiopeiaTree.collapse_unifurcations` to use
  the shared utility backend for consistent topology updates.

### Fixed

- .c files for cython extensions are now included in the source distribution.
