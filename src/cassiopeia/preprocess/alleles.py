"""Allele calling: aligned BAM -> molecule table."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from . import _native
from ._utils import as_list, default_threads, finish_plot, record_params
from .construct import Construct, load_construct


def _resolve_contigs(construct: Construct, names: list[str], contigs) -> list[int]:
    if contigs is None:
        return []
    if isinstance(contigs, str) and contigs != "auto":
        contigs = [contigs]
    if contigs == "auto":
        selected = [i for i, n in enumerate(names) if n == construct.name]
        if construct.contig_pattern:
            pat = re.compile(construct.contig_pattern)
            selected += [i for i, n in enumerate(names) if pat.search(n) and n != construct.name]
        if not selected:
            shown = ", ".join(names[:20]) + (" ..." if len(names) > 20 else "")
            raise PreprocessError(
                f"No contig named {construct.name!r}"
                + (f" or matching {construct.contig_pattern!r}" if construct.contig_pattern else "")
                + f" in the BAM header ({shown}). Pass `contigs=` explicitly or `contigs=None` "
                "to scan all reads."
            )
        return sorted(set(selected))
    index = {n: i for i, n in enumerate(names)}
    missing = [c for c in contigs if c not in index]
    if missing:
        raise PreprocessError(f"Contigs not in BAM header: {missing}")
    return sorted(index[c] for c in contigs)


def call_alleles(
    bam: str | Path,
    construct: Construct | str,
    *,
    cb_tag: str | None = "CB",
    umi_tag: str | None = "UB",
    count_tag: str | None = "ZR",
    contigs: Literal["auto"] | Iterable[str] | None = "auto",
    intbc_from: Literal["auto", "sequence", "contig"] = "auto",
    min_mapq: int = 0,
    use_index: bool = True,
    cells: str | Path | Iterable[str] | None = None,
    umi_max_dist: int = 0,
    sample: str | None = None,
    legacy: bool = False,
    normalize_indels: bool = True,
    clip_missing: bool = True,
    n_threads: int | None = None,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Call per-molecule alleles from an aligned target-site BAM.

    Works with BAMs from :func:`cassiopeia.pp.align`, STARsolo and Cell Ranger.
    Reads are streamed, so BAMs of any size can be processed; with an index,
    only the construct contigs are read. For each read the intBC is read from
    the bases aligned to the masked intBC (or from the contig name) and every
    edit site is called from the alignment. Reads are then grouped into
    molecules (cell, intBC, UMI) and each site is called by majority vote.

    Args:
        bam: Aligned BAM (coordinate-sorted and indexed for fast contig access).
        construct: Construct, preset name or YAML path.
        cb_tag: Cell barcode tag; ``None`` for bulk data.
        umi_tag: UMI tag; ``None`` for data without UMIs (each distinct allele
            becomes one row with its read count).
        count_tag: Integer tag with the number of reads a record represents
            (``ZR`` from :func:`cassiopeia.pp.align`); records without it count once.
        contigs: Contigs to read: ``"auto"`` (the construct contig and contigs
            matching ``construct.contig_pattern``), a list of names, or ``None``
            to scan all reads.
        intbc_from: Read the intBC from the read sequence or use the contig
            name. ``"auto"`` uses the contig name for contigs matching
            ``construct.contig_pattern``.
        min_mapq: Minimum mapping quality.
        use_index: Use the BAM index (``.bai``/``.csi``) to read only the
            selected contigs. ``False`` streams the whole file (still
            filtering by contig); used automatically when there is no index.
        cells: Cell barcodes to keep (list or file, e.g. Cell Ranger's
            ``barcodes.tsv.gz``).
        umi_max_dist: Merge UMIs within this Hamming distance inside each
            (cell, intBC), as the legacy pipeline did. 0 disables (STARsolo and
            Cell Ranger already correct UMIs).
        sample: Optional sample name, stored in a ``sample`` column.
        legacy: Reproduce the legacy Cassiopeia indel calls exactly
            (implies no indel normalization and no clip handling).
        normalize_indels: Left-align indels before calling (makes calls
            independent of the aligner).
        clip_missing: Call a site missing (instead of unedited) when the
            alignment does not cover its window, e.g. because of soft clipping.
        n_threads: Threads (default: ``SLURM_CPUS_PER_TASK`` or up to 8).
        plot: Plot UMIs vs. reads per UMI and per-site allele frequencies.
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        Molecule table with categorical ``cellBC``, ``intBC``, ``UMI`` and one
        column per edit site (NaN = missing), plus ``n_reads`` and
        ``allele_frac`` (lowest fraction of reads supporting a site call).
        ``df.attrs`` holds the construct, site names and read statistics.
    """
    t0 = time.time()
    c = load_construct(construct)
    bam = str(bam)
    n_threads = default_threads(n_threads)
    names, lengths = _native.references(bam)
    tids = _resolve_contigs(c, names, contigs)
    for t in tids:
        if names[t] == c.name and lengths[t] != len(c.reference):
            raise PreprocessError(
                f"Contig {names[t]!r} has length {lengths[t]} but the construct reference has "
                f"{len(c.reference)}bp; the BAM was aligned to a different reference."
            )
    if intbc_from == "auto":
        from_contig = bool(c.contig_pattern) and any(names[t] != c.name for t in tids)
    else:
        from_contig = intbc_from == "contig"
    if c.intbc is None and not from_contig:
        logger.info(f"Construct {c.name!r} has no intBC; using the contig name as intBC.")
    cell_list = as_list(cells)
    options = {
        "path": bam,
        "tids": tids,
        "use_index": use_index,
        "cb_tag": cb_tag or "",
        "umi_tag": umi_tag or "",
        "count_tag": count_tag or "",
        "min_mapq": min_mapq,
        "n_threads": n_threads,
        "cells": cell_list or [],
        "umi_max_dist": umi_max_dist if umi_tag else 0,
        "intbc_from_contig": from_contig,
        "default_intbc": c.name,
    }
    native = c._native(
        legacy=legacy,
        normalize_indels=normalize_indels and not legacy,
        clip_missing=clip_missing and not legacy,
    )
    res = _native.call_alleles(native, options)

    def cat(codes, cats):
        return pd.Categorical.from_codes(codes, categories=pd.Index(cats, dtype=object))

    data = {}
    codes, cats = res["cellBC"]
    if cb_tag:
        data["cellBC"] = cat(codes, cats)
    else:
        data["cellBC"] = pd.Categorical([sample or "bulk"] * len(codes))
    data["intBC"] = cat(*res["intBC"])
    if umi_tag:
        data["UMI"] = cat(*res["UMI"])
    else:
        data["UMI"] = pd.Categorical([None] * len(codes), categories=pd.Index([], dtype=object))
    for site, (scodes, scats) in zip(c.site_names, res["sites"], strict=True):
        data[site] = cat(scodes, scats)
    data["n_reads"] = res["n_reads"]
    data["allele_frac"] = res["allele_frac"]
    df = pd.DataFrame(data)
    if sample is not None:
        df.insert(0, "sample", pd.Categorical([sample] * len(df)))
    stats = dict(res["stats"])
    stats["site_missing_reads"] = dict(zip(c.site_names, res["site_missing"].tolist(), strict=True))
    stats["site_clipped_reads"] = dict(zip(c.site_names, res["site_clipped"].tolist(), strict=True))
    df.attrs["construct"] = c.name
    df.attrs["sites"] = c.site_names
    df.attrs["edit_models"] = {s.name: s.edit_model for s in c.sites}
    df.attrs["site_states"] = {s.name: s.edited_states for s in c.sites if s.lookup}
    df.attrs["read_stats"] = stats
    record_params(df, umi_max_dist=umi_max_dist)
    elapsed = time.time() - t0
    used = stats.get("reads_used", 0)
    logger.info(
        f"call_alleles: {stats.get('records', 0):,} records read, {used:,} reads used, "
        f"{len(df):,} molecules in {df['cellBC'].nunique():,} cells ({elapsed:.1f}s)"
    )
    if stats.get("records_on_target", 0) and used / max(stats["records_on_target"], 1) < 0.5:
        logger.warning(
            "Fewer than half of the on-target reads were usable; check the construct "
            f"(read statistics: {stats})."
        )
    if plot or save:
        from .plotting import plot_call_alleles

        fig = plot_call_alleles(df)
        finish_plot(fig, show and bool(plot), save)
    return df


def call_read(
    construct: Construct | str,
    ref_pos: int,
    cigar: list[tuple[int, int]],
    seq: str,
    qual: Iterable[int] | None = None,
    legacy: bool = False,
) -> dict | None:
    """Call the intBC and site alleles of a single aligned read (for inspection).

    Args:
        construct: Construct, preset name or YAML path.
        ref_pos: 0-based alignment start on the construct reference.
        cigar: CIGAR as ``(op, length)`` pairs using BAM op codes (0=M, 1=I,
            2=D, 3=N, 4=S).
        seq: Read sequence.
        qual: Base qualities (Phred).
        legacy: Use legacy allele calling.
    """
    c = load_construct(construct)
    q = None if qual is None else bytes(np.asarray(list(qual), dtype=np.uint8))
    return _native.call_read_py(
        c._native(legacy=legacy, normalize_indels=not legacy, clip_missing=not legacy),
        ref_pos,
        list(cigar),
        seq,
        q,
    )
