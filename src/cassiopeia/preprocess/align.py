"""FASTQ -> deduplicated, aligned, CB/UB-tagged BAM."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pysam

from cassiopeia.mixins import PreprocessError, logger

from . import _native
from ._utils import default_threads, finish_plot
from .construct import Construct, load_construct
from .layout import ReadLayout, get_layout


def _lanes(fastqs, n_reads: int) -> list[list[str]]:
    if isinstance(fastqs, (str, Path)):
        fastqs = [fastqs]
    fastqs = list(fastqs)
    if fastqs and isinstance(fastqs[0], (str, Path)):
        if len(fastqs) % n_reads:
            raise PreprocessError(
                f"Expected a multiple of {n_reads} FASTQ files, got {len(fastqs)}."
            )
        lanes = [fastqs[i : i + n_reads] for i in range(0, len(fastqs), n_reads)]
    else:
        lanes = [list(lane) for lane in fastqs]
    for lane in lanes:
        if len(lane) != n_reads:
            raise PreprocessError(f"Each lane needs {n_reads} FASTQ files, got {lane}.")
        for p in lane:
            if not Path(p).exists():
                raise PreprocessError(f"FASTQ not found: {p}")
    return [[str(p) for p in lane] for lane in lanes]


def align(
    fastqs: str | Sequence[str] | Sequence[Sequence[str]],
    output_bam: str | Path,
    construct: Construct | str,
    *,
    layout: str | ReadLayout | dict = "10xv3",
    whitelist: str | Path | None = None,
    cb_translation: str | Path | None = None,
    min_bc_qual: int = 10,
    min_base_qual: int = 20,
    umi_max_dist: int = 1,
    dedup: Literal["umi", "sequence"] = "umi",
    max_mismatches: int = 3,
    max_indels: int = 2,
    method: Literal["local", "global"] = "local",
    match: int = 5,
    mismatch: int = -4,
    gap_open: int = 20,
    gap_extend: int = 1,
    min_score: int = 1,
    strand: Literal["forward", "reverse", "both"] = "forward",
    cb_confidence: float = 0.975,
    n_threads: int | None = None,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> dict:
    """Align target-site reads and write a deduplicated, CB/UB-tagged BAM.

    A self-contained alternative to STARsolo / Cell Ranger for target-site
    libraries that reproduces the legacy Cassiopeia alignment. Steps, all in
    one streaming pass with no intermediate files:

    1. Parse the cell barcode, UMI and target-site read from the FASTQs
       (``layout``); drop reads with N or low-quality barcode/UMI bases.
    2. Collapse reads with the same raw cell barcode and UMI.
    3. Correct cell barcodes to the whitelist (1 mismatch, posterior using
       base qualities and barcode abundance, as in Cell Ranger).
    4. Merge molecules that now share a cell barcode and UMI, then correct
       UMIs within each cell (directional, as in UMI-tools / STARsolo).
    5. Deduplicate: one consensus sequence per UMI (``dedup="umi"``) or one
       record per distinct sequence (``dedup="sequence"``). As in the legacy
       collapse, a UMI's reads are clustered around its most common sequence
       (``max_mismatches``) and smaller clusters are merged into the biggest
       when an alignment shows at most ``max_indels`` gap bases, so reads with
       a sequencing indel still count towards their UMI.
    6. Smith-Waterman (or global) alignment of each distinct sequence to the
       construct reference, with legacy scoring by default.
    7. Write a coordinate-sorted, indexed BAM. Tags: ``CB``, ``UB``, ``ZR``
       (reads), ``ZU`` (UMIs merged), ``ZF`` (read fraction of the consensus)
       and ``AS`` (alignment score).

    Args:
        fastqs: FASTQ paths in read order for one lane (e.g. ``[R1, R2]``), a
            flat list for several lanes, or a list of lanes.
        output_bam: Output BAM path.
        construct: Construct, preset name or YAML path.
        layout: Read layout name (``"10xv3"``, ``"10xv2"``, ``"indropsv3"``,
            ``"slideseq2"``, ``"dropseq"``, ``"bulk"``, ...) or definition.
        whitelist: Cell barcode whitelist (plain or gzipped). Without one,
            cell barcodes are not corrected.
        cb_translation: Two-column barcode translation table (reported
            barcode, barcode in the reads), for libraries captured with 10x
            Feature Barcoding (capture sequence), whose bead barcodes differ
            from the gene expression barcodes (Cell Ranger's
            ``barcodes/translation/3M-february-2018.txt.gz``). Reads are
            corrected to the second column and reported as the first, so cell
            barcodes match the gene expression library. Replaces ``whitelist``.
        min_bc_qual: Minimum Phred quality of every barcode/UMI base.
        min_base_qual: Target-site bases below this quality do not count as
            mismatches when clustering a UMI's reads and have little weight
            in its consensus (as in the legacy collapse).
        umi_max_dist: Hamming distance for UMI correction (0 disables).
        dedup: ``"umi"`` (one consensus per UMI) or ``"sequence"``.
        max_mismatches: High-quality mismatches allowed to join a UMI's
            cluster or merge two of its clusters.
        max_indels: Gap bases allowed when merging a UMI's clusters (-1 skips
            the alignment-based merge).
        method: ``"local"`` (Smith-Waterman) or ``"global"`` (free end gaps).
        match: Match score.
        mismatch: Mismatch score.
        gap_open: Penalty for the first base of a gap.
        gap_extend: Penalty for each additional gap base.
        min_score: Records scoring lower are written unaligned.
        strand: Orientation of the target-site read relative to the reference.
        cb_confidence: Posterior required to correct a cell barcode.
        n_threads: Threads (default: ``SLURM_CPUS_PER_TASK`` or up to 8).
        plot: Plot barcode rank, reads per UMI and alignment scores.
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        Summary dict with ``stats`` and per-cell ``cells``/``cell_reads``/
        ``cell_umis`` arrays.
    """
    t0 = time.time()
    c = load_construct(construct)
    lay = get_layout(layout)
    lanes = _lanes(fastqs, lay.n_reads)
    if lay.cb and whitelist is None and cb_translation is None:
        logger.warning(
            "No cell barcode whitelist given; cell barcodes will not be corrected"
            + (f" (usual whitelist: {lay.whitelist_hint})." if lay.whitelist_hint else ".")
        )
    output_bam = str(output_bam)
    Path(output_bam).parent.mkdir(parents=True, exist_ok=True)
    params = {
        "layout": lay.name,
        "dedup": dedup,
        "method": method,
        "umi_max_dist": umi_max_dist,
        "min_bc_qual": min_bc_qual,
        "scoring": [match, mismatch, gap_open, gap_extend],
    }
    header_extra = (
        "@PG\tID:cassiopeia.pp.align\tPN:cassiopeia\t"
        f"CL:{json.dumps(params, separators=(',', ':'))}\n"
    )
    options = {
        "lanes": lanes,
        "layout": lay._native(),
        "output_bam": output_bam,
        "whitelist": str(whitelist) if whitelist else "",
        "cb_translation": str(cb_translation) if cb_translation else "",
        "min_bc_qual": min_bc_qual,
        "min_base_qual": min_base_qual,
        "umi_max_dist": umi_max_dist if lay.umi else 0,
        "dedup_umi": dedup == "umi",
        "max_mismatches": max_mismatches,
        "max_indels": max_indels,
        "global": method == "global",
        "scoring": {
            "match": match,
            "mismatch": mismatch,
            "n_score": 0,
            "gap_open": gap_open,
            "gap_extend": gap_extend,
        },
        "min_score": min_score,
        "strand": {"forward": 0, "reverse": 1, "both": 2}[strand],
        "n_threads": default_threads(n_threads),
        "ref_name": c.name,
        "ref_seq": c.reference,
        "cb_confidence": cb_confidence,
        "header_extra": header_extra,
    }
    res = _native.align_fastqs(options)
    pysam.index(output_bam)
    stats = res["stats"]
    elapsed = time.time() - t0
    logger.info(
        f"align: {stats['reads']:,} reads -> {stats['molecules']:,} molecules, "
        f"{stats['records_aligned']:,}/{stats['records_written']:,} records aligned ({elapsed:.1f}s)"
    )
    res["bam"] = output_bam
    res["elapsed"] = elapsed
    if plot or save:
        from .plotting import plot_align

        fig = plot_align(res, min_score=min_score)
        finish_plot(fig, show and bool(plot), save)
    return res
