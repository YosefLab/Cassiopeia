===========
Preprocess
===========
.. currentmodule:: cassiopeia

The preprocessing module turns sequencing reads into per-cell character
matrices. It supports Cas9 indel recorders (Cassiopeia / KP-tracer, GESTALT),
prime-editing recorders (PEtracer), sequential recorders (DNA Typewriter) and
base-editing recorders (BASELINE) through a declarative :class:`~cassiopeia.pp.Construct`.

Reads reach an aligned, CB/UB-tagged BAM through any of three paths:

* **Cell Ranger** (``cellranger mkref`` + ``count``) or **STARsolo**, using the
  reference written by :func:`~cassiopeia.pp.build_reference`;
* :func:`~cassiopeia.pp.align`, a self-contained FASTQ -> deduplicated
  Smith-Waterman alignment that follows the pre-3.0 Cassiopeia collapse and
  alignment, so ``pp.call_alleles(legacy=True)`` reproduces its allele calls.

Read layouts (where the cell barcode, UMI and target site are in the reads)
are given as a preset name or a pattern string such as ``"16C12U,T"`` (see
:func:`~cassiopeia.pp.align`).

All paths converge in :func:`~cassiopeia.pp.call_alleles`, which produces a
molecule table (a pandas DataFrame). The remaining steps annotate that table
(``pass_*`` columns; nothing is dropped until the character matrix is built)
and each has a built-in QC plot (``plot=True`` / ``save=``).

Constructs and references
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.Construct
   pp.Site
   pp.TapeArray
   pp.load_construct
   pp.list_presets
   pp.build_reference

Reads to molecules
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.align
   pp.call_alleles
   pp.call_read
   pp.correct_intbcs

Filtering, clones and doublets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.filter_molecules
   pp.filter_cells
   pp.gmm_threshold
   pp.call_clones
   pp.clone_sizes
   pp.filter_integrations
   pp.integration_metrics
   pp.filter_doublets
   pp.doublet_scores

Character matrices
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.alleles_to_characters
   pp.to_allele_table
   pp.run
   pp.read_molecules

QC metrics and plots
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.qc.cell_metrics
   pp.qc.molecule_metrics
   pp.plot_umi_reads
   pp.plot_intbc_detection
   pp.plot_integrations
   pp.plot_clone_intbc_heatmap
   pp.plot_doublet_scores
   pp.plot_edit_frequency

Data utilities
~~~~~~~~~~~~~~

Converters for legacy allele tables (see :func:`~cassiopeia.pp.to_allele_table`):

.. autosummary::
   :toctree: reference/

   pp.compute_empirical_indel_priors
   pp.convert_alleletable_to_character_matrix
   pp.convert_alleletable_to_lineage_profile
   pp.convert_character_matrix_to_allele_table
   pp.convert_lineage_profile_to_character_matrix
