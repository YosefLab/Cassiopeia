"""Molecule and cell filters. Filters annotate ``pass_*`` columns; nothing is dropped."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from ._utils import finish_plot, molecule_passing, passing, record_params, set_pass


def filter_molecules(
    df: pd.DataFrame,
    min_reads: int = 2,
    min_allele_frac: float = 0.5,
    min_umi_per_intbc: int = 1,
    *,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Flag low-quality molecules.

    Adds ``pass_min_reads`` (UMIs with at least ``min_reads`` reads),
    ``pass_allele_frac`` (at least ``min_allele_frac`` of a molecule's reads
    agree at every site) and ``pass_umis_per_intbc`` (the (cell, intBC) has at
    least ``min_umi_per_intbc`` passing UMIs; removes most ambient and
    erroneous intBCs).

    Args:
        df: Molecule table.
        min_reads: Minimum reads per UMI.
        min_allele_frac: Minimum fraction of reads supporting each site call.
        min_umi_per_intbc: Minimum passing UMIs per (cell, intBC).
        plot: Plot UMIs vs. mean reads per UMI per (cell, intBC).
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        The molecule table with ``pass_*`` columns.
    """
    df = df.copy()
    set_pass(df, "min_reads", df["n_reads"].to_numpy() >= min_reads)
    set_pass(df, "allele_frac", df["allele_frac"].to_numpy() >= min_allele_frac)
    ok = df["pass_min_reads"].to_numpy() & df["pass_allele_frac"].to_numpy()
    grp = pd.DataFrame({"cell": df["cellBC"].cat.codes, "intbc": df["intBC"].cat.codes, "ok": ok})
    n_ok = grp.groupby(["cell", "intbc"])["ok"].transform("sum").to_numpy()
    set_pass(df, "umis_per_intbc", n_ok >= min_umi_per_intbc)
    record_params(
        df,
        min_reads=min_reads,
        min_allele_frac=min_allele_frac,
        min_umi_per_intbc=min_umi_per_intbc,
    )
    logger.info(
        f"filter_molecules: {int(df['pass'].sum()):,}/{len(df):,} molecules pass "
        f"(min_reads={min_reads}, min_allele_frac={min_allele_frac}, min_umi_per_intbc={min_umi_per_intbc})"
    )
    if plot or save:
        from .plotting import plot_umi_reads

        ax = plot_umi_reads(df, level="intbc")
        finish_plot(ax.figure, show and bool(plot), save)
    return df


def _cell_umis(df: pd.DataFrame) -> pd.Series:
    d = molecule_passing(df)
    return d.groupby("cellBC", observed=False).size()


def intbc_detection(df: pd.DataFrame) -> pd.Series:
    """Per cell, fraction of its clone's kept integrations that are detected."""
    d = passing(df)
    kept = df.attrs.get("integrations") or df.attrs.get("clone_intbcs")
    if not kept:
        raise PreprocessError(
            "intBC detection needs pp.call_clones (and pp.filter_integrations) first."
        )
    if "clone" not in d.columns:
        raise PreprocessError("intBC detection needs a `clone` column (run pp.call_clones).")
    kept_sets = {str(c): set(v) for c, v in kept.items()}
    pairs = d[["cellBC", "intBC", "clone"]].drop_duplicates()
    pairs = pairs[pairs["clone"].notna()]
    clone = pairs["clone"].astype(object).to_numpy()
    intbc = pairs["intBC"].astype(object).to_numpy()
    in_sig = np.fromiter(
        (str(i) in kept_sets.get(str(c), ()) for i, c in zip(intbc, clone, strict=True)),
        dtype=bool,
        count=len(pairs),
    )
    det = (
        pd.Series(in_sig.astype(float), index=pairs["cellBC"].astype(object).to_numpy())
        .groupby(level=0)
        .sum()
    )
    cc = d.drop_duplicates("cellBC")
    cells = cc["cellBC"].astype(object).to_numpy()
    sizes = np.array(
        [len(kept_sets.get(str(c), ())) for c in cc["clone"].astype(object)], dtype=float
    )
    sizes[sizes == 0] = np.nan
    return pd.Series(det.reindex(cells).fillna(0).to_numpy() / sizes, index=cells)


def filter_cells(
    df: pd.DataFrame,
    min_umi: int | None = None,
    min_intbc_frac: float | None = None,
    *,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Flag low-quality cells.

    Can be called more than once: typically with ``min_umi`` before clone
    calling (removes empty or ambient barcodes) and with ``min_intbc_frac``
    after :func:`cassiopeia.pp.filter_integrations`.

    Args:
        df: Molecule table.
        min_umi: Minimum UMIs per cell, counting molecules that pass
            :func:`cassiopeia.pp.filter_molecules`.
        min_intbc_frac: Minimum fraction of the cell's clone's kept
            integrations detected in the cell.
        plot: Plot UMIs vs. intBCs detected per cell (and the fraction of
            clone integrations detected, with ``min_intbc_frac``).
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        The molecule table with ``pass_min_umi`` / ``pass_intbc_frac`` columns.
    """
    df = df.copy()
    if min_umi is not None:
        umis = _cell_umis(df)
        keep = umis[umis >= min_umi].index
        set_pass(df, "min_umi", df["cellBC"].isin(keep).to_numpy())
        record_params(df, min_umi=min_umi)
        logger.info(
            f"filter_cells: {len(keep):,}/{int((umis > 0).sum()):,} cells with >= {min_umi} UMIs"
        )
    if min_intbc_frac is not None:
        frac = intbc_detection(df)
        keep = frac[frac >= min_intbc_frac].index
        set_pass(df, "intbc_frac", df["cellBC"].isin(keep).to_numpy())
        record_params(df, min_intbc_frac=min_intbc_frac)
        logger.info(
            f"filter_cells: {len(keep):,}/{len(frac):,} cells detect >= {min_intbc_frac:.0%} of their integrations"
        )
    if plot or save:
        from .plotting import plot_filter_cells

        fig = plot_filter_cells(df)
        finish_plot(fig, show and bool(plot), save)
    return df
