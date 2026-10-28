"""Allele and cell filters. Filters annotate ``pass_*`` columns; nothing is dropped."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from ._utils import (
    finish_plot,
    record_params,
    row_passing,
    set_pass,
    site_columns,
    umi_counts,
)


def _codes(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: df[c].cat.codes.to_numpy() for c in names}, copy=False)


def filter_alleles(
    df: pd.DataFrame,
    min_allele_frac: float = 0.5,
    min_umi_per_allele: int = 2,
    min_avg_reads_per_umi: float = 0,
    allow_allele_conflicts: bool = False,
    *,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Flag unreliable alleles in the allele table.

    Each allele (cell, intBC and site alleles) is judged as a whole. UMIs with
    too few reads were already dropped (``min_reads_per_umi`` of
    :func:`cassiopeia.pp.call_alleles`). Adds four filters:

    * ``pass_umis_per_allele``: the allele has at least
      ``min_umi_per_allele`` UMIs. Alleles seen in a single UMI are much less
      reliable; this also removes most ambient and erroneous intBCs.
    * ``pass_avg_reads_per_umi``: the allele's UMIs have at least
      ``min_avg_reads_per_umi`` reads on average. Molecules from ambient RNA
      tend to have fewer reads than the cell's own.
    * ``pass_allele_frac``: the allele holds at least ``min_allele_frac`` of
      the UMIs of its (cell, intBC) (``allele_frac``).
    * ``pass_allele_conflict``: with ``allow_allele_conflicts=False``, only
      the allele with the most UMIs (then reads) among those passing the
      filters above is kept for each (cell, intBC). With ``True``, every such
      allele is kept, and :func:`cassiopeia.pp.alleles_to_characters` needs
      ``allow_ambiguous=True``.

    Alleles removed only by ``pass_allele_frac`` or ``pass_allele_conflict``
    still count where the presence of an intBC or of several alleles
    matters: UMIs per cell (:func:`cassiopeia.pp.filter_cells`), clone
    calling, and the conflict rates of
    :func:`cassiopeia.pp.filter_integrations` and
    :func:`cassiopeia.pp.filter_doublets`.

    The UMI and read thresholds depend on sequencing depth: lower them for
    shallowly sequenced libraries and raise them for deep ones. The plot shows
    where the thresholds fall.

    Args:
        df: Allele table.
        min_allele_frac: Minimum fraction of the (cell, intBC)'s UMIs carrying
            the allele.
        min_umi_per_allele: Minimum UMIs per allele.
        min_avg_reads_per_umi: Minimum mean reads per UMI of the allele.
        allow_allele_conflicts: Keep every allele passing the filters instead
            of only the most abundant one per (cell, intBC).
        plot: Plot UMIs vs. mean reads per UMI per allele, colored by allele
            fraction.
        save: Save the plot to this path.

    Returns:
        The allele table with ``pass_*`` columns.
    """
    df = df.copy()
    pair_cols = [c for c in ("sample", "cellBC", "intBC") if c in df.columns]
    codes = _codes(df, pair_cols + site_columns(df))
    pair = codes.groupby(pair_cols, sort=False).ngroup().to_numpy()
    allele = codes.groupby(list(codes.columns), sort=True).ngroup().to_numpy()
    umis = umi_counts(df)
    allele_umis = np.bincount(allele, weights=umis)
    allele_reads = np.bincount(allele, weights=df["n_reads"].to_numpy().astype(np.float64))
    avg_reads = allele_reads / np.maximum(allele_umis, 1)
    set_pass(df, "umis_per_allele", allele_umis[allele] >= min_umi_per_allele)
    set_pass(df, "avg_reads_per_umi", avg_reads[allele] >= min_avg_reads_per_umi)
    set_pass(df, "allele_frac", df["allele_frac"].to_numpy() >= min_allele_frac)
    candidate = (
        df["pass_umis_per_allele"].to_numpy()
        & df["pass_avg_reads_per_umi"].to_numpy()
        & df["pass_allele_frac"].to_numpy()
    )
    if allow_allele_conflicts:
        minor = np.zeros(len(df), dtype=bool)
    else:
        # the most abundant candidate allele of each (cell, intBC)
        _, first = np.unique(allele, return_index=True)
        alleles = pd.DataFrame(
            {"pair": pair[first], "umis": allele_umis, "reads": allele_reads},
        )[candidate[first]]
        top = alleles.sort_values(
            ["pair", "umis", "reads"], ascending=[True, False, False], kind="stable"
        ).drop_duplicates("pair")
        is_top = np.zeros(len(allele_umis), dtype=bool)
        is_top[top.index.to_numpy()] = True
        minor = candidate & ~is_top[allele]
    set_pass(df, "allele_conflict", ~minor)
    record_params(
        df,
        min_allele_frac=min_allele_frac,
        min_umi_per_allele=min_umi_per_allele,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
        allow_allele_conflicts=allow_allele_conflicts,
    )
    passed = df["pass"].to_numpy()
    logger.info(
        f"filter_alleles: {int(umis[passed].sum()):,}/{int(umis.sum()):,} UMIs pass "
        f"(min_allele_frac={min_allele_frac}, min_umi_per_allele={min_umi_per_allele}, "
        f"min_avg_reads_per_umi={min_avg_reads_per_umi})"
        + ("" if allow_allele_conflicts else f"; {int(umis[minor].sum()):,} UMIs of minor alleles")
    )
    if plot or save:
        from ._plotting import plot_filter_alleles

        finish_plot(plot_filter_alleles(df), plot, save)
    return df


def _cell_umis(df: pd.DataFrame) -> pd.Series:
    d = row_passing(df)
    return pd.Series(umi_counts(d), index=d.index).groupby(d["cellBC"], observed=False).sum()


def filter_cells(
    df: pd.DataFrame,
    min_umi: int | None = None,
    min_intbc_frac: float | None = None,
    *,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Flag low-quality cells.

    Can be called more than once: typically with ``min_umi`` before clone
    calling (removes empty or ambient barcodes) and, optionally, with
    ``min_intbc_frac`` after :func:`cassiopeia.pp.filter_integrations`.

    Args:
        df: Allele table.
        min_umi: Minimum UMIs per cell, counting UMIs of alleles that pass
            the UMI and read filters of :func:`cassiopeia.pp.filter_alleles`
            (minor alleles included).
        min_intbc_frac: Minimum fraction of the intBCs expected for the
            cell's clone (its kept integrations; all kept integrations
            without clones) that are detected in the cell.
        plot: Plot UMIs vs. intBCs detected per cell (and the fraction of
            expected intBCs detected, with ``min_intbc_frac``).
        save: Save the plot to this path.

    Returns:
        The allele table with ``pass_min_umi`` / ``pass_intbc_frac`` columns.
    """
    from .qc import _intbc_fraction

    df = df.copy()
    if min_umi is not None:
        umis = _cell_umis(df)
        keep = umis[umis >= min_umi].index
        set_pass(df, "min_umi", df["cellBC"].isin(keep).to_numpy())
        record_params(df, min_umi=min_umi)
        logger.info(
            f"filter_cells: {len(keep):,}/{df['cellBC'].nunique():,} cells with >= {min_umi} UMIs"
        )
    if min_intbc_frac is not None:
        frac = _intbc_fraction(df)
        keep = frac[frac >= min_intbc_frac].index
        set_pass(df, "intbc_frac", df["cellBC"].astype(str).isin(keep).to_numpy())
        record_params(df, min_intbc_frac=min_intbc_frac)
        logger.info(
            f"filter_cells: {len(keep):,}/{int(frac.notna().sum()):,} cells detect "
            f">= {min_intbc_frac:.0%} of their expected intBCs"
        )
    if plot or save:
        from ._plotting import plot_filter_cells

        finish_plot(plot_filter_cells(df), plot, save)
    return df
