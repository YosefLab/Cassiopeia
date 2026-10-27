"""Allele calling: aligned BAM -> allele table."""

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
from ._utils import (
    as_list,
    default_threads,
    finish_plot,
    record_params,
    site_columns,
    umi_counts,
)
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


def _group_rows(
    keys: dict[str, np.ndarray], umis: np.ndarray, reads: np.ndarray, frac: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group rows with identical integer keys.

    Returns the first row of each group (in order of appearance), the summed
    UMIs and reads, and the read-weighted allele fraction.
    """
    if len(umis) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, np.zeros(0)
    gid = pd.DataFrame(keys, copy=False).groupby(list(keys), sort=False).ngroup().to_numpy()
    n = int(gid.max()) + 1
    _, first = np.unique(gid, return_index=True)
    reads = np.asarray(reads, dtype=np.float64)
    n_umis = np.bincount(gid, weights=np.asarray(umis, dtype=np.float64), minlength=n)
    n_reads = np.bincount(gid, weights=reads, minlength=n)
    support = np.bincount(gid, weights=np.asarray(frac, dtype=np.float64) * reads, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        allele_frac = np.where(n_reads > 0, support / n_reads, 1.0)
    return first, n_umis.astype(np.int64), n_reads.astype(np.int64), allele_frac


def _collapse(df: pd.DataFrame, read_cap: int | None = None) -> pd.DataFrame:
    """Merge rows with the same cell, intBC, allele, read class and pass flags."""
    if read_cap is None:
        read_cap = int(df.attrs.get("qc_params", {}).get("read_cap", 5))
    sites = site_columns(df)
    names = [c for c in ("sample", "cellBC", "intBC") if c in df.columns] + sites
    names += [c for c in df.columns if c == "pass" or c.startswith("pass_")]
    keys = {
        c: df[c].cat.codes.to_numpy()
        if isinstance(df[c].dtype, pd.CategoricalDtype)
        else df[c].to_numpy()
        for c in names
    }
    umis = umi_counts(df)
    reads = df["n_reads"].to_numpy()
    keys["_reads"] = np.minimum(reads // np.maximum(umis, 1), read_cap)
    first, n_umis, n_reads, frac = _group_rows(keys, umis, reads, df["allele_frac"].to_numpy())
    out = df.iloc[first].drop(columns=["UMI", "n_umis"], errors="ignore").reset_index(drop=True)
    out["n_umis"] = n_umis.astype(np.int32)
    out["n_reads"] = n_reads.astype(np.int32)
    out["allele_frac"] = frac.astype(np.float32)
    lead = [c for c in names if c in out.columns and not c.startswith("pass")]
    lead += ["n_umis", "n_reads", "allele_frac"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.attrs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in df.attrs.items()}
    record_params(out, read_cap=read_cap)
    return out


def collapse_umis(df: pd.DataFrame, read_cap: int = 5) -> pd.DataFrame:
    """Collapse UMIs with the same cell, intBC and allele into one row.

    The allele table from :func:`cassiopeia.pp.call_alleles` has one row per
    UMI. Collapsing merges UMIs with the same cell, intBC, allele and number
    of reads into one row with ``n_umis`` and ``n_reads`` (summed over its
    UMIs); UMIs with ``read_cap`` or more reads are merged regardless of their
    read count. Keeping read counts apart means
    :func:`cassiopeia.pp.filter_alleles` filters UMIs by reads exactly as on
    the uncollapsed table, for ``min_reads_per_umi`` up to ``read_cap``. The
    table is typically 2-6x smaller, and every later step works on either
    form. ``allele_frac`` becomes the fraction of the row's reads that
    support the allele. Rows that differ in a ``pass_*`` flag are kept apart.

    Use ``pp.call_alleles(collapse_umis=True)`` to collapse while reading the
    BAM, before the full table is built. Collapsed UMIs can no longer be
    merged if :func:`cassiopeia.pp.correct_intbcs` later finds that two of
    them are the same molecule with a sequencing error in the intBC; this
    does not happen with BAMs from :func:`cassiopeia.pp.align`, which merges
    such reads itself, and adds about 1% to UMI counts with STARsolo BAMs.

    Args:
        df: Allele table.
        read_cap: UMIs with at least this many reads are merged regardless of
            their read count.

    Returns:
        The collapsed allele table (without the ``UMI`` column).
    """
    out = _collapse(df, read_cap)
    logger.info(f"collapse_umis: {len(df):,} rows -> {len(out):,} alleles")
    return out


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
    collapse_umis: bool = False,
    read_cap: int = 5,
    n_threads: int | None = None,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Call per-molecule alleles from an aligned target-site BAM.

    Works with BAMs from :func:`cassiopeia.pp.align`, STARsolo and Cell Ranger
    (the latter two are recommended only for insertion and substitution
    recorders; see :func:`cassiopeia.pp.build_reference`).
    Reads are streamed, so BAMs of any size can be processed; with an index,
    only the construct contigs are read. For each read the intBC is read from
    the bases aligned to the masked intBC (or from the contig name) and every
    edit site is called from the alignment. Reads are then grouped into
    molecules (cell, intBC, UMI) and each site is called by majority vote.
    With ``collapse_umis=True``, UMIs with the same cell, intBC and allele are
    then collapsed into one row (see :func:`cassiopeia.pp.collapse_umis`)
    before the table is built, which keeps it small for large experiments.

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
        collapse_umis: Collapse UMIs with the same cell, intBC, allele and
            read count into one row with ``n_umis`` (see
            :func:`cassiopeia.pp.collapse_umis`).
        read_cap: (``collapse_umis``) UMIs with at least this many reads are
            collapsed regardless of their read count.
        n_threads: Threads (default: ``SLURM_CPUS_PER_TASK`` or up to 8).
        plot: Plot what happened to the reads and the alleles called at each
            site (unedited, the most common edits, missing).
        save: Save the plot to this path.

    Returns:
        Allele table with categorical ``cellBC``, ``intBC``, ``UMI`` and one
        column per edit site (NaN = missing), plus ``n_reads`` and
        ``allele_frac`` (lowest fraction of reads supporting a site call).
        With ``collapse_umis``, ``UMI`` is replaced by ``n_umis``.
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

    n_reads, allele_frac = res["n_reads"], res["allele_frac"]
    rows = slice(None)
    n_umis = None
    if collapse_umis:
        keys = {"cellBC": res["cellBC"][0], "intBC": res["intBC"][0]}
        keys.update({f"site{j}": s[0] for j, s in enumerate(res["sites"])})
        keys["_reads"] = np.minimum(n_reads, read_cap)
        rows, n_umis, n_reads, allele_frac = _group_rows(
            keys, np.ones(len(n_reads), dtype=np.int64), n_reads, allele_frac
        )
    data = {}
    codes, cats = res["cellBC"]
    codes = codes[rows]
    if cb_tag:
        data["cellBC"] = cat(codes, cats)
    else:
        data["cellBC"] = pd.Categorical([sample or "bulk"] * len(codes))
    data["intBC"] = cat(res["intBC"][0][rows], res["intBC"][1])
    if not collapse_umis:
        if umi_tag:
            data["UMI"] = cat(*res["UMI"])
        else:
            data["UMI"] = pd.Categorical([None] * len(codes), categories=pd.Index([], dtype=object))
    for site, (scodes, scats) in zip(c.site_names, res["sites"], strict=True):
        data[site] = cat(scodes[rows], scats)
    if n_umis is not None:
        data["n_umis"] = n_umis.astype(np.int32)
    data["n_reads"] = np.asarray(n_reads).astype(np.int32)
    data["allele_frac"] = np.asarray(allele_frac).astype(np.float32)
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
    df.attrs["has_umis"] = bool(umi_tag)
    record_params(df, umi_max_dist=umi_max_dist)
    if collapse_umis:
        record_params(df, read_cap=read_cap)
    elapsed = time.time() - t0
    used = stats.get("reads_used", 0)
    logger.info(
        f"call_alleles: {stats.get('records', 0):,} records read, {used:,} reads used, "
        f"{stats.get('molecules', len(df)):,} molecules"
        + (f" -> {len(df):,} alleles" if collapse_umis else "")
        + f" in {df['cellBC'].nunique():,} cells ({elapsed:.1f}s)"
    )
    if stats.get("records_on_target", 0) and used / max(stats["records_on_target"], 1) < 0.5:
        logger.warning(
            "Fewer than half of the on-target reads were usable; check the construct "
            f"(read statistics: {stats})."
        )
    if plot or save:
        from ._plotting import plot_call_alleles

        finish_plot(plot_call_alleles(df), plot, save)
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
