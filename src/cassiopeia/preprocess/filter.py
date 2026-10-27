"""Molecule and cell filters. Filters annotate ``pass_*`` columns; nothing is dropped."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from ._utils import finish_plot, passing, record_params, set_pass


def gmm_threshold(values: np.ndarray, log: bool = True, n_iter: int = 200) -> float | None:
    """Threshold separating the low and high component of a 2-component GMM.

    Fits a two-component Gaussian mixture (in log10 space by default) and
    returns the value where the posterior of the high component reaches 0.5.
    Returns None if the data is not clearly bimodal.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)] if log else x[np.isfinite(x)]
    if len(x) < 20:
        return None
    if log:
        x = np.log10(x)
    lo, hi = np.percentile(x, [10, 90])
    if hi - lo < 1e-6:
        return None
    mu = np.array([lo, hi])
    sd = np.array([max(x.std() / 2, 1e-3)] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        dens = np.stack(
            [w[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / sd[k] for k in range(2)]
        )
        tot = dens.sum(0) + 1e-300
        r = dens / tot
        nk = r.sum(1) + 1e-12
        new_mu = (r * x).sum(1) / nk
        new_sd = np.sqrt((r * (x - new_mu[:, None]) ** 2).sum(1) / nk) + 1e-6
        w = nk / nk.sum()
        if np.allclose(new_mu, mu, atol=1e-6) and np.allclose(new_sd, sd, atol=1e-6):
            mu, sd = new_mu, new_sd
            break
        mu, sd = new_mu, new_sd
    if mu[0] > mu[1]:
        mu, sd, w = mu[::-1], sd[::-1], w[::-1]
    # require clearly separated components (Ashman's D >= 3) and, in log
    # space, modes >= 4x apart: empty barcodes vs cells, not cells vs doublets
    ashman_d = np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    min_sep = np.log10(4) if log else 0.0
    if w.min() < 0.02 or ashman_d < 3 or (mu[1] - mu[0]) < min_sep:
        return None
    grid = np.linspace(mu[0], mu[1], 2000)
    d0 = w[0] * np.exp(-0.5 * ((grid - mu[0]) / sd[0]) ** 2) / sd[0]
    d1 = w[1] * np.exp(-0.5 * ((grid - mu[1]) / sd[1]) ** 2) / sd[1]
    idx = np.flatnonzero(d1 >= d0)
    t = grid[idx[0]] if len(idx) else mu[1] - 2 * sd[1]
    return float(10**t) if log else float(t)


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


def _cell_umis(df: pd.DataFrame, use_pass: bool = True) -> pd.Series:
    d = passing(df) if use_pass else df
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
    min_umi: int | Literal["gmm"] | None = None,
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
        min_umi: Minimum passing UMIs per cell, or ``"gmm"`` to fit a
            two-component mixture to log UMI counts and use the boundary.
        min_intbc_frac: Minimum fraction of the cell's clone's kept
            integrations detected in the cell.
        plot: Plot UMIs vs. reads per UMI per cell (and intBC detection).
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        The molecule table with ``pass_min_umi`` / ``pass_intbc_frac`` columns.
    """
    df = df.copy()
    if min_umi is not None:
        umis = _cell_umis(df)
        if min_umi == "gmm":
            thr = gmm_threshold(umis[umis > 0].to_numpy())
            if thr is None:
                logger.warning(
                    "filter_cells: UMI counts do not separate cells from empty barcodes; "
                    "no UMI threshold applied."
                )
        else:
            thr = float(min_umi)
        keep = umis[umis >= (thr or 0)].index
        set_pass(df, "min_umi", df["cellBC"].isin(keep).to_numpy())
        record_params(df, min_umi=thr, min_umi_method="gmm" if min_umi == "gmm" else "fixed")
        if thr is not None:
            logger.info(
                f"filter_cells: {len(keep):,}/{int((umis > 0).sum()):,} cells with >= {thr:.1f} UMIs"
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
