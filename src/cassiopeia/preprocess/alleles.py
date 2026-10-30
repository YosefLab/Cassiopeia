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
    allele_fractions,
    as_list,
    check_choice,
    copy_attrs,
    default_threads,
    finish_plot,
    is_collapsed,
    reads_per_umi_hist,
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


_GEM_SUFFIX = re.compile(r"-\d+$")


def _barcode_variants(cells: list[str]) -> list[str]:
    """``cells`` plus each barcode without its ``-1``-style suffix, or with ``-1`` added.

    Cell Ranger and STARsolo barcodes end in ``-1`` while :func:`cassiopeia.pp.align`
    writes bare barcodes, so both forms are matched. Barcodes that differ only in
    the suffix (several GEM wells) are matched exactly.
    """
    bases = [_GEM_SUFFIX.sub("", c) for c in cells]
    if len(set(bases)) < len(bases):
        return cells
    added = {b + "-1" for b, c in zip(bases, cells, strict=True) if b == c}
    return sorted(set(cells) | set(bases) | added)


def _peek_tag(bam: str, tag: str, n: int = 1000) -> list[str]:
    """Values of ``tag`` in the first ``n`` records of a BAM."""
    import pysam

    out = []
    with pysam.AlignmentFile(bam, check_sq=False) as f:
        for i, r in enumerate(f.fetch(until_eof=True)):
            if i >= n:
                break
            if r.has_tag(tag):
                out.append(str(r.get_tag(tag)))
    return out


def _warn_read_loss(
    stats: dict,
    n_rows: int,
    bam: str,
    *,
    cb_tag: str | None,
    umi_tag: str | None,
    cells: list[str] | None,
    min_mapq: int,
    min_reads_per_umi: int,
) -> None:
    """Warn when no alleles are called or most on-target records are dropped, naming why."""
    on_target = stats.get("records_on_target", 0)
    used = stats.get("records_used", 0)
    if stats.get("records", 0) == 0:
        logger.warning("call_alleles: the BAM has no records on the selected contigs.")
        return
    # barcodes outside `cells` (e.g. empty droplets) are expected unless nothing is left
    expected_loss = stats.get("not_in_cells", 0) if n_rows else 0
    if used >= 0.5 * (on_target - expected_loss):
        if n_rows == 0 and stats.get("low_read_umis", 0):
            logger.warning(
                f"call_alleles: no alleles were called: all {stats['low_read_umis']:,} UMIs have "
                f"fewer than {min_reads_per_umi} reads. Lower `min_reads_per_umi`."
            )
        return
    losses = {
        "filtered_flag": ("unmapped, secondary or supplementary", None),
        "filtered_mapq": (f"MAPQ < {min_mapq}", "lower `min_mapq`"),
        "missing_cell": (
            f"no {cb_tag!r} tag",
            "check `cb_tag` (pp.align, STARsolo and Cell Ranger write 'CB'), or pass "
            "`cb_tag=None` for bulk data",
        ),
        "not_in_cells": ("cell barcode not in `cells`", None),
        "missing_umi": (
            f"no {umi_tag!r} tag",
            "check `umi_tag` (pp.align, STARsolo and Cell Ranger write 'UB'), or pass "
            "`umi_tag=None` for data without UMIs",
        ),
        "unaligned": (
            "not aligned",
            "check that the BAM was aligned to the construct reference",
        ),
        "no_intbc": ("intBC not read", "check the construct's `intbc` position"),
    }
    counts = {k: stats.get(k, 0) for k in losses if stats.get(k, 0)}
    if n_rows:
        counts.pop("not_in_cells", None)
    parts = [f"{n:,} {losses[k][0]}" for k, n in sorted(counts.items(), key=lambda x: -x[1])]
    head = (
        "no alleles were called"
        if n_rows == 0
        else f"only {used:,}/{on_target - expected_loss:,} on-target records were used"
    )
    msg = f"call_alleles: {head}"
    if parts:
        msg += f"; records dropped: {', '.join(parts)}"
    if counts:
        top = max(counts, key=counts.get)
        hint = losses[top][1]
        if top == "not_in_cells" and cells and cb_tag:
            seen = _peek_tag(bam, cb_tag, 1000)
            if seen:
                hint = (
                    f"`cells` must hold the BAM's {cb_tag!r} barcodes (e.g. {seen[0]!r}; "
                    f"`cells` has e.g. {cells[0]!r})"
                )
        if hint:
            msg += f". {hint[0].upper()}{hint[1:]}"
    logger.warning(msg + ".")


def _group_rows(
    keys: dict[str, np.ndarray], umis: np.ndarray, reads: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows with identical integer keys.

    Returns the first row of each group (in order of appearance) and the
    summed UMIs and reads of each group.
    """
    if len(umis) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    gid = pd.DataFrame(keys, copy=False).groupby(list(keys), sort=False).ngroup().to_numpy()
    n = int(gid.max()) + 1
    _, first = np.unique(gid, return_index=True)
    n_umis = np.bincount(gid, weights=np.asarray(umis, dtype=np.float64), minlength=n)
    n_reads = np.bincount(gid, weights=np.asarray(reads, dtype=np.float64), minlength=n)
    return first, n_umis.astype(np.int64), n_reads.astype(np.int64)


def _collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows with the same cell, intBC, allele and pass flags."""
    sites = site_columns(df)
    names = [c for c in ("sample", "cellBC", "intBC") if c in df.columns] + sites
    names += [c for c in df.columns if c == "pass" or c.startswith("pass_")]
    keys = {
        c: df[c].cat.codes.to_numpy()
        if isinstance(df[c].dtype, pd.CategoricalDtype)
        else df[c].to_numpy()
        for c in names
    }
    first, n_umis, n_reads = _group_rows(keys, umi_counts(df), df["n_reads"].to_numpy())
    out = df.iloc[first].drop(columns=["UMI", "n_umis", "allele_frac"], errors="ignore")
    out = copy_attrs(df, out.reset_index(drop=True))
    out["n_umis"] = n_umis.astype(np.int32)
    out["n_reads"] = n_reads.astype(np.int32)
    out["allele_frac"] = allele_fractions(out)
    lead = [c for c in names if not c.startswith("pass")] + ["n_umis", "n_reads", "allele_frac"]
    return out[lead + [c for c in out.columns if c not in lead]]


def collapse_umis(
    df: pd.DataFrame,
    min_reads_per_umi: int = 2,
    *,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Collapse UMIs with the same cell, intBC and allele into one row.

    UMIs with fewer than ``min_reads_per_umi`` reads are dropped first. Gives
    one row per (cell, intBC, allele) with ``n_umis`` and ``n_reads`` summed
    over its UMIs and ``allele_frac``, the fraction of the (cell, intBC)'s
    UMIs that carry the allele. Every later step works on either form.

    :func:`cassiopeia.pp.call_alleles` collapses UMIs by default, while
    reading the BAM; use this function on tables from
    ``call_alleles(collapse_umis=False)``. Collapsed UMIs can no longer be
    merged if :func:`cassiopeia.pp.correct_intbcs` later finds that two of
    them are the same molecule with a sequencing error in the intBC; this
    does not happen with BAMs from :func:`cassiopeia.pp.align`, which merges
    such reads itself, and adds about 1% to UMI counts with STARsolo BAMs.

    Args:
        df: Allele table with one row per UMI.
        min_reads_per_umi: Drop UMIs with fewer reads.
        plot: Plot the reads per UMI, with ``min_reads_per_umi``.
        save: Save the plot to this path.

    Returns:
        The collapsed allele table (without the ``UMI`` column).
    """
    if is_collapsed(df):
        raise PreprocessError("The allele table is already collapsed.")
    if not df.attrs.get("has_umis", True):
        raise PreprocessError("The allele table has no UMIs; each row is already an allele.")
    reads = df["n_reads"].to_numpy()
    keep = reads >= min_reads_per_umi
    out = _collapse(copy_attrs(df, df[keep]))
    record_params(out, min_reads_per_umi=min_reads_per_umi)
    logger.info(
        f"collapse_umis: {int(keep.sum()):,}/{len(df):,} UMIs with >= {min_reads_per_umi} reads "
        f"-> {len(out):,} alleles"
    )
    if plot or save:
        from ._plotting import _plt, plot_reads_per_umi

        fig, ax = _plt().subplots(figsize=(5.2, 3.4), constrained_layout=True)
        plot_reads_per_umi(reads_per_umi_hist(reads), min_reads_per_umi, ax=ax)
        finish_plot(fig, plot, save)
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
    normalize_indels: bool = True,
    clip_missing: bool = True,
    min_reads_per_umi: int = 2,
    collapse_umis: bool = True,
    n_threads: int | None = None,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Call alleles from an aligned target-site BAM.

    Works with BAMs from :func:`cassiopeia.pp.align`, STARsolo and Cell Ranger
    (the latter two are recommended only for insertion and substitution
    recorders).
    Reads are streamed, so BAMs of any size can be processed; with an index,
    only the construct contigs are read. For each read the intBC is read from
    the bases aligned to the masked intBC (or from the contig name) and every
    edit site is called from the alignment. Reads are then grouped into
    molecules (cell, intBC, UMI) and each site is called by majority vote.
    UMIs with fewer than ``min_reads_per_umi`` reads are dropped. By default,
    UMIs with the same cell, intBC and allele are then collapsed into one row
    (see :func:`cassiopeia.pp.collapse_umis`) before the table is built,
    which keeps it small for large experiments.

    Process one sample per allele table. The later steps identify cells by
    ``cellBC`` alone, so the same barcode in two samples would be merged, and
    they need the categorical columns and ``df.attrs`` (thresholds, read
    statistics) that ``pd.concat`` drops. To analyze several samples
    together, process each one separately and combine the character
    matrices: use ``mapping="allele"`` or a shared dict in
    :func:`cassiopeia.pp.alleles_to_characters` so that states mean the same
    in every sample, and make the cell names unique (e.g. prefix them with
    the sample name).

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
            ``barcodes.tsv.gz``). Barcodes match with or without the ``-1``
            suffix that Cell Ranger and STARsolo add and
            :func:`cassiopeia.pp.align` does not; ``cellBC`` keeps the BAM's
            barcodes.
        umi_max_dist: Merge UMIs within this Hamming distance inside each
            (cell, intBC), as the pre-3.0 pipeline did. 0 disables (STARsolo
            and Cell Ranger already correct UMIs).
        sample: Optional sample name, stored in a ``sample`` column. Do not
            concatenate the tables of several samples (see above).
        normalize_indels: Left-align indels before calling (makes calls
            independent of the aligner).
        clip_missing: Call a site missing (instead of unedited) when the
            alignment does not cover its window, e.g. because of soft clipping.
        min_reads_per_umi: Drop UMIs with fewer reads. Depends on sequencing
            depth: lower it for shallow libraries; the plot shows where it
            falls. Not used without UMIs.
        collapse_umis: Collapse UMIs with the same cell, intBC and allele into
            one row (see :func:`cassiopeia.pp.collapse_umis`). ``False``
            returns one row per UMI. Without UMIs (``umi_tag=None``) each row
            is already a distinct allele and nothing is collapsed.
        n_threads: Threads (default: ``SLURM_CPUS_PER_TASK`` or up to 8).
        plot: Plot the reads per UMI, what happened to the reads, and the
            alleles called at each site (unedited, the most common edits,
            missing).
        save: Save the plot to this path.

    Returns:
        Allele table with categorical ``cellBC``, ``intBC`` and one column per
        edit site (NaN = missing). Collapsed, each row is one (cell, intBC,
        allele) with ``n_umis`` and ``n_reads``; otherwise each row is one
        UMI, with ``UMI`` and ``n_reads``. ``allele_frac`` is the fraction of
        the (cell, intBC)'s UMIs (reads without UMIs) that carry the allele.
        ``df.attrs`` holds the construct, site names and read statistics,
        including the number of UMIs with each read count
        (``read_stats['reads_per_umi']``).
    """
    t0 = time.time()
    check_choice("intbc_from", intbc_from, ("auto", "sequence", "contig"))
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
        "cells": _barcode_variants(cell_list) if cell_list else [],
        "umi_max_dist": umi_max_dist if umi_tag else 0,
        "min_reads": min_reads_per_umi if umi_tag else 0,
        "intbc_from_contig": from_contig,
        "default_intbc": c.name,
    }
    native = c._native(normalize_indels=normalize_indels, clip_missing=clip_missing)
    res = _native.call_alleles(native, options)

    def cat(codes, cats):
        return pd.Categorical.from_codes(codes, categories=pd.Index(cats, dtype=object))

    n_reads = res["n_reads"]
    collapse = collapse_umis and bool(umi_tag)
    rows = slice(None)
    n_umis = None
    if collapse:
        keys = {"cellBC": res["cellBC"][0], "intBC": res["intBC"][0]}
        keys.update({f"site{j}": s[0] for j, s in enumerate(res["sites"])})
        rows, n_umis, n_reads = _group_rows(keys, np.ones(len(n_reads), dtype=np.int64), n_reads)
    data = {}
    codes, cats = res["cellBC"]
    codes = codes[rows]
    if cb_tag:
        data["cellBC"] = cat(codes, cats)
    else:
        data["cellBC"] = pd.Categorical([sample or "bulk"] * len(codes))
    data["intBC"] = cat(res["intBC"][0][rows], res["intBC"][1])
    if not collapse:
        if umi_tag:
            data["UMI"] = cat(*res["UMI"])
        else:
            data["UMI"] = pd.Categorical([None] * len(codes), categories=pd.Index([], dtype=object))
    for site, (scodes, scats) in zip(c.site_names, res["sites"], strict=True):
        data[site] = cat(scodes[rows], scats)
    if n_umis is not None:
        data["n_umis"] = n_umis.astype(np.int32)
    data["n_reads"] = np.asarray(n_reads).astype(np.int32)
    df = pd.DataFrame(data)
    if sample is not None:
        df.insert(0, "sample", pd.Categorical([sample] * len(df)))
    stats = dict(res["stats"])
    stats["site_missing_reads"] = dict(zip(c.site_names, res["site_missing"].tolist(), strict=True))
    stats["site_clipped_reads"] = dict(zip(c.site_names, res["site_clipped"].tolist(), strict=True))
    if umi_tag:
        stats["reads_per_umi"] = {
            "reads": res["hist_reads"].tolist(),
            "umis": res["hist_umis"].tolist(),
        }
    else:
        for k in ("low_read_umis", "low_read_umi_records", "low_read_umi_reads"):
            stats.pop(k, None)
    df.attrs["construct"] = c.name
    df.attrs["sites"] = c.site_names
    df.attrs["edit_models"] = {s.name: s.edit_model for s in c.sites}
    df.attrs["site_states"] = {s.name: s.edited_states for s in c.sites if s.lookup}
    df.attrs["read_stats"] = stats
    df.attrs["has_umis"] = bool(umi_tag)
    df["allele_frac"] = allele_fractions(df)
    record_params(df, umi_max_dist=umi_max_dist)
    if umi_tag:
        record_params(df, min_reads_per_umi=min_reads_per_umi)
    elapsed = time.time() - t0
    used = stats.get("reads_used", 0)
    low = stats.get("low_read_umis", 0)
    logger.info(
        f"call_alleles: {stats.get('records', 0):,} records read, {used:,} reads used, "
        f"{stats.get('molecules', len(df)):,} molecules"
        + (f" ({low:,} UMIs with < {min_reads_per_umi} reads dropped)" if umi_tag else "")
        + (f" -> {len(df):,} alleles" if collapse else "")
        + f" in {df['cellBC'].nunique():,} cells ({elapsed:.1f}s)"
    )
    if cell_list and cb_tag and len(df):
        n_suffix = len(set(df["cellBC"].astype(str)) - set(cell_list))
        if n_suffix:
            logger.info(
                f"call_alleles: {n_suffix:,} cells matched `cells` only after adding or removing "
                "the '-1' barcode suffix; cellBC keeps the BAM's barcodes."
            )
    _warn_read_loss(
        stats,
        len(df),
        bam,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        cells=cell_list,
        min_mapq=min_mapq,
        min_reads_per_umi=min_reads_per_umi,
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
) -> dict | None:
    """Call the intBC and site alleles of a single aligned read (for inspection).

    Args:
        construct: Construct, preset name or YAML path.
        ref_pos: 0-based alignment start on the construct reference.
        cigar: CIGAR as ``(op, length)`` pairs using BAM op codes (0=M, 1=I,
            2=D, 3=N, 4=S, 5=H, 6=P, 7==, 8=X).
        seq: Read sequence.
        qual: Base qualities (Phred), one per base of ``seq``.

    Returns:
        ``None`` if the read cannot be called (e.g. it does not reach the
        edit sites), otherwise a dict with ``intBC`` (the bases aligned to the
        intBC; empty if not covered), ``tokens`` (the allele called at each
        site, in construct order; empty if missing) and ``clipped`` (per
        site, whether soft clipping hides its window).

    Raises:
        ValueError: If the CIGAR does not match the sequence or runs past
            the reference.
    """
    c = load_construct(construct)
    cigar = [(int(op), int(n)) for op, n in cigar]
    bad = [(op, n) for op, n in cigar if not 0 <= op <= 8 or n < 0]
    if bad:
        raise ValueError(f"Invalid CIGAR operations {bad}: use BAM op codes 0-8.")
    query = sum(n for op, n in cigar if op in (0, 1, 4, 7, 8))
    ref = sum(n for op, n in cigar if op in (0, 2, 3, 7, 8))
    if query != len(seq):
        raise ValueError(f"The CIGAR covers {query} read bases but `seq` has {len(seq)}.")
    if qual is not None:
        qual = list(qual)
        if len(qual) != len(seq):
            raise ValueError(f"`qual` has {len(qual)} values but `seq` has {len(seq)} bases.")
    if ref_pos < 0 or ref_pos + ref > len(c.reference):
        raise ValueError(
            f"The alignment spans reference positions {ref_pos}-{ref_pos + ref} but the "
            f"construct reference has {len(c.reference)} bases."
        )
    q = None if qual is None else bytes(np.asarray(qual, dtype=np.uint8))
    return _native.call_read_py(c._native(), ref_pos, cigar, seq, q)
