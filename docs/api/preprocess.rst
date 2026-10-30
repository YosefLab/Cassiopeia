===========
Preprocess
===========
.. currentmodule:: cassiopeia

The preprocessing module turns sequencing reads into per-cell character
matrices. It supports lineage tracing technologies, including Cas9 indel (Weissman, GESTALT),
prime-editing (PEtracer), sequential (DNA Typewriter) and
base-editing recorders (BASELINE) through a declarative :class:`~cassiopeia.pp.Construct`.

Reads reach an aligned, CB/UB-tagged BAM through two alternative paths:

* :func:`~cassiopeia.pp.align`, a self-contained FASTQ -> deduplicated
  Smith-Waterman alignment that follows the pre-3.0 Cassiopeia collapse and
  alignment procedure.
* **Cell Ranger**  or **STARsolo**, using the
  reference written by :func:`~cassiopeia.pp.build_reference`. Recommended
  only for insertion and substitution recorders since STAR cannot align reads
  across large Cas9 deletions.

All paths converge through :func:`~cassiopeia.pp.call_alleles`, which produces the
**allele table**: a pandas DataFrame with one row per allele (cell, intBC, and
per site edit calls). These rows correspond to either UMIs or sets of UMIs
sharing the same edit calls depending on the value of ``collapse_umis``.

Quality control and filtering of the **allele table** are performed using the
functions below. Each function flags low quality data using a ``pass_*`` column
rather than filtering, and most draw QC plots to help select thresholds.
:func:`~cassiopeia.pp.run` runs them in this order:
:func:`~cassiopeia.pp.correct_intbcs`, :func:`~cassiopeia.pp.filter_alleles`,
:func:`~cassiopeia.pp.filter_cells`, :func:`~cassiopeia.pp.call_clones` (only
for data with several clones), :func:`~cassiopeia.pp.filter_integrations`,
optionally :func:`~cassiopeia.pp.filter_cells` with ``min_intbc_frac``, and
:func:`~cassiopeia.pp.filter_doublets`. :func:`~cassiopeia.pp.filter_doublets`
and :func:`~cassiopeia.pp.filter_integrations` can run in either order;
running :func:`~cassiopeia.pp.filter_doublets` first keeps doublets out of the
integrations' conflict rates.

Finally the **character matrix** is constructed from the filtered allele
table using :func:`~cassiopeia.pp.alleles_to_characters`.

Each allele table holds one sample. The steps identify cells by ``cellBC``
alone and need the categorical columns and ``df.attrs`` that ``pd.concat``
drops, so do not concatenate the allele tables of several samples. Process
each sample separately and combine the character matrices instead, with
``mapping="allele"`` (or a shared dict) so that states mean the same in every
sample, and with unique cell names (e.g. prefixed with the sample name).

Running the pipeline
~~~~~~~~~~~~~~~~~~~~

:func:`~cassiopeia.pp.run` runs every step from reads to the filtered allele
table and saves it as parquet. Load it with :func:`~cassiopeia.pp.read_alleles`
and build the character matrix with :func:`~cassiopeia.pp.alleles_to_characters`.

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

.. autosummary::
   :toctree: reference/

   pp.cell_metrics
   pp.clone_metrics
   pp.integration_metrics
   pp.compute_indel_priors
   pp.plot_edit_frequency
