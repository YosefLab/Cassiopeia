===========
Preprocess
===========
.. currentmodule:: cassiopeia

The preprocessing module turns sequencing reads into per-cell character
matrices. It supports Cas9 indel recorders (Cassiopeia / KP-tracer, GESTALT),
prime-editing recorders (PEtracer), sequential recorders (DNA Typewriter) and
base-editing recorders (BASELINE) through a declarative :class:`~cassiopeia.pp.Construct`.

Reads reach an aligned, CB/UB-tagged BAM through any of three paths:

* :func:`~cassiopeia.pp.align`, a self-contained FASTQ -> deduplicated
  Smith-Waterman alignment that follows the pre-3.0 Cassiopeia collapse and
  alignment. Use it for Cas9 indel recorders (Cassiopeia / KP-tracer,
  GESTALT); it works for every recorder.
* **Cell Ranger** (``cellranger mkref`` + ``count``) or **STARsolo**, using the
  reference written by :func:`~cassiopeia.pp.build_reference`. Recommended
  only for insertion and substitution recorders (PEtracer, DNA Typewriter,
  BASELINE): STAR cannot align reads across large Cas9 deletions, so with
  indel recorders the most edited molecules are lost.

Read layouts (where the cell barcode, UMI and target site are in the reads)
are given as a preset name or a pattern string such as ``"16C12U,T"`` (see
:func:`~cassiopeia.pp.align`).

All paths converge in :func:`~cassiopeia.pp.call_alleles`, which produces the
**allele table**: a pandas DataFrame with one row per allele (cell, intBC and
site alleles) and one column per edit site. UMIs with fewer than
``min_reads_per_umi`` reads are dropped while the BAM is read. Each row holds
the number of UMIs and reads supporting the allele and ``allele_frac``, the
fraction of the (cell, intBC)'s UMIs that carry it. ``collapse_umis=False``
gives one row per UMI instead; every later step works on either form. The
remaining steps annotate the allele table (``pass_*`` columns; nothing is
dropped until the character matrix is built).
:func:`~cassiopeia.pp.filter_alleles` keeps the most abundant allele of each
(cell, intBC) by default, so the character matrix has one state per cell and
character. :func:`~cassiopeia.pp.call_clones` is only
needed for data with several clones; otherwise all cells are treated as one
clone. Every step draws a QC plot (``plot=True`` by default; ``save=`` writes
it to a file).

Running the pipeline
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.run
   pp.read_alleles
   pp.alleles_to_characters

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

Reads to alleles
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.align
   pp.call_alleles
   pp.collapse_umis
   pp.call_read
   pp.correct_intbcs

Filtering, clones and doublets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   pp.filter_alleles
   pp.filter_cells
   pp.call_clones
   pp.filter_integrations
   pp.filter_doublets
   pp.doublet_scores

QC metrics and plots
~~~~~~~~~~~~~~~~~~~~

:func:`~cassiopeia.pp.cell_metrics`, :func:`~cassiopeia.pp.clone_metrics`
and :func:`~cassiopeia.pp.integration_metrics` summarize the allele table per
cell, per clone and per integration; their docstrings describe every column.

.. autosummary::
   :toctree: reference/

   pp.cell_metrics
   pp.clone_metrics
   pp.integration_metrics
   pp.compute_indel_priors
   pp.plot_edit_frequency
