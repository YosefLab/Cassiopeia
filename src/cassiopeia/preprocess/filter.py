"""Allele and cell filters. Filters annotate ``pass_*`` columns; nothing is dropped."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from ._utils import (
    finish_plot,
    is_collapsed,
    record_params,
    row_passing,
    set_pass,
    umi_counts,
)


def filter_alleles(
    df: pd.DataFrame,
    min_reads_per_umi: float = 3,
    min_allele_frac: float = 0.5,
    min_umi_per_intbc: int = 2,
    *,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Flag low-quality rows of the allele table.

    Adds ``pass_reads_per_umi`` (UMIs with at least ``min_reads_per_umi``
    reads; exact on tables from :func:`cassiopeia.pp.collapse_umis` for
    thresholds up to its ``read_cap``), ``pass_allele_frac`` (at least
    ``min_allele_frac`` of the reads agree with the allele at every site) and
    ``pass_umis_per_intbc`` (the cell has at least ``min_umi_per_intbc``
    passing UMIs for the intBC; removes most ambient and erroneous intBCs).

    Both read and UMI thresholds depend on sequencing depth: lower them for
    shallowly sequenced libraries and raise them for deep ones. The plot shows
    where the thresholds fall.

    Args:
        df: Allele table.
        min_reads_per_umi: Minimum reads per UMI.
        min_allele_frac: Minimum fraction of reads supporting the allele at
            each site.
        min_umi_per_intbc: Minimum passing UMIs per (cell, intBC).
        plot: Plot the reads per UMI, and the passing UMIs per (cell, intBC).
        save: Save the plot to this path.

    Returns:
        The allele table with ``pass_*`` columns.
    """
    df = df.copy()
    umis = umi_counts(df)
    read_cap = df.attrs.get("qc_params", {}).get("read_cap")
    if is_collapsed(df) and read_cap is not None and min_reads_per_umi > read_cap:
        logger.warning(
            f"UMIs were collapsed with read_cap={read_cap}; UMIs with {read_cap} or more reads "
            f"are filtered by their mean reads per UMI, not exactly at min_reads_per_umi="
            f"{min_reads_per_umi}. Collapse with a higher read_cap to filter exactly."
        )
    reads_per_umi = df["n_reads"].to_numpy() / np.maximum(umis, 1)
    set_pass(df, "reads_per_umi", reads_per_umi >= min_reads_per_umi)
    set_pass(df, "allele_frac", df["allele_frac"].to_numpy() >= min_allele_frac)
    ok = df["pass_reads_per_umi"].to_numpy() & df["pass_allele_frac"].to_numpy()
    grp = pd.DataFrame(
        {"cell": df["cellBC"].cat.codes, "intbc": df["intBC"].cat.codes, "umis": umis * ok}
    )
    n_ok = grp.groupby(["cell", "intbc"])["umis"].transform("sum").to_numpy()
    set_pass(df, "umis_per_intbc", n_ok >= min_umi_per_intbc)
    record_params(
        df,
        min_reads_per_umi=min_reads_per_umi,
        min_allele_frac=min_allele_frac,
        min_umi_per_intbc=min_umi_per_intbc,
    )
    passed = df["pass"].to_numpy()
    logger.info(
        f"filter_alleles: {int(umis[passed].sum()):,}/{int(umis.sum()):,} UMIs pass "
        f"(min_reads_per_umi={min_reads_per_umi}, min_allele_frac={min_allele_frac}, "
        f"min_umi_per_intbc={min_umi_per_intbc})"
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
        min_umi: Minimum UMIs per cell, counting UMIs that pass
            :func:`cassiopeia.pp.filter_alleles`.
        min_intbc_frac: Minimum fraction of the intBCs expected for the
            cell's clone (its kept integrations; all kept integrations
            without clones) that are detected in the cell.
        plot: Plot UMIs vs. intBCs detected per cell (and the fraction of
            expected intBCs detected, with ``min_intbc_frac``).
        save: Save the plot to this path.

    Returns:
        The allele table with ``pass_min_umi`` / ``pass_intbc_frac`` columns.
    """
    from .qc import intbc_fraction

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
        frac = intbc_fraction(df)
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
