"""Integration (clone x intBC) quality filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from ._resolve import resolve_alleles
from ._utils import finish_plot, passing, record_params, set_pass


def integration_metrics(
    df: pd.DataFrame,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
) -> pd.DataFrame:
    """Per (clone, intBC) detection and allele-conflict rates.

    Args:
        df: Molecule table (with a ``clone`` column; otherwise all cells form
            one clone).
        min_umi: UMIs a minor allele needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.

    Returns:
        DataFrame with ``clone``, ``intBC``, ``n_cells`` (cells with the intBC),
        ``clone_cells``, ``detection`` (n_cells / clone_cells), ``conflict``
        (fraction of cells with conflicting alleles) and ``mean_umis``.
    """
    d = passing(
        df.drop(columns=["pass_integration", "pass_doublet"], errors="ignore").assign(
            **{"pass": _pass_without(df, ["pass_integration", "pass_doublet"])}
        )
    )
    res = resolve_alleles(d, min_umi=min_umi, min_minor_frac=min_minor_frac, use_pass=False)
    if "clone" in df.columns:
        cc = d.drop_duplicates("cellBC")
        cell_clone = pd.Series(
            cc["clone"].astype(object).to_numpy(), index=cc["cellBC"].astype(str).to_numpy()
        )
    else:
        cell_clone = pd.Series("0", index=d["cellBC"].astype(str).unique(), dtype=object)
    res["clone"] = res["cellBC"].astype(str).map(cell_clone).astype(object)
    res = res[res["clone"].notna() & (res["clone"] != "multiplet")]
    sizes = cell_clone[cell_clone.notna() & (cell_clone != "multiplet")].value_counts()
    g = res.groupby(["clone", "intBC"], observed=True)
    out = pd.DataFrame(
        {
            "n_cells": g.size(),
            "n_conflict": g["conflict"].sum(),
            "mean_umis": g["n_umis"].mean(),
        }
    ).reset_index()
    out["clone"] = out["clone"].astype(str)
    out["intBC"] = out["intBC"].astype(str)
    out["clone_cells"] = out["clone"].map(sizes).astype(int)
    out["detection"] = out["n_cells"] / out["clone_cells"]
    out["conflict"] = out["n_conflict"] / out["n_cells"]
    return out.sort_values(["clone", "detection"], ascending=[True, False]).reset_index(drop=True)


def _pass_without(df: pd.DataFrame, exclude: list[str]) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("pass_") and c not in exclude]
    if not cols:
        return np.ones(len(df), dtype=bool)
    return df[cols].all(axis=1).to_numpy()


def filter_integrations(
    df: pd.DataFrame,
    min_detection: float = 0.4,
    max_conflict: float = 0.2,
    *,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Keep reliable integrations (clone x intBC pairs).

    For each clone and intBC, ``detection`` is the fraction of the clone's
    cells with the intBC and ``conflict`` the fraction of those cells holding
    two incompatible alleles. Integrations are kept if
    ``detection >= min_detection`` and ``conflict <= max_conflict``, which
    removes silenced or poorly captured integrations, duplicated
    integrations sharing an intBC, and foreign intBCs (ambient RNA, doublets,
    errors). Doublets are a minority of cells, so they barely change these
    rates; doublet scoring then only uses the kept integrations.

    With a single clone (``pp.call_clones(method="single")``) this is the
    population intBC whitelist plus duplicate detection.

    Args:
        df: Molecule table with ``clone`` (all cells are one clone otherwise).
        min_detection: Minimum detection rate within the clone.
        max_conflict: Maximum fraction of cells with conflicting alleles.
        min_umi: UMIs a minor allele needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.
        plot: Plot detection vs. conflict per integration.
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        The molecule table with ``pass_integration``; kept integrations per
        clone are stored in ``df.attrs['integrations']``.
    """
    df = df.copy()
    m = integration_metrics(df, min_umi=min_umi, min_minor_frac=min_minor_frac)
    m["kept"] = (m["detection"] >= min_detection) & (m["conflict"] <= max_conflict)
    kept = m[m["kept"]].groupby("clone")["intBC"].apply(lambda s: sorted(s)).to_dict()
    if "clone" in df.columns:
        clone = df["clone"].astype(object).to_numpy()
    else:
        clone = np.full(len(df), "0", dtype=object)
    keys = set(zip(m.loc[m["kept"], "clone"], m.loc[m["kept"], "intBC"], strict=True))
    intbc = df["intBC"].astype(str).to_numpy()
    ok = np.fromiter(
        (((str(c), i) in keys) for c, i in zip(clone, intbc, strict=True)),
        dtype=bool,
        count=len(df),
    )
    set_pass(df, "integration", ok)
    df.attrs["integrations"] = {str(k): v for k, v in kept.items()}
    df.attrs["integration_table"] = {
        "clone": m["clone"].tolist(),
        "intBC": m["intBC"].tolist(),
        "detection": m["detection"].round(4).tolist(),
        "conflict": m["conflict"].round(4).tolist(),
        "kept": m["kept"].tolist(),
    }
    record_params(
        df,
        integration_min_detection=min_detection,
        integration_max_conflict=max_conflict,
        conflict_min_umi=min_umi,
        conflict_min_minor_frac=min_minor_frac,
    )
    n_dup = int(((m["detection"] >= min_detection) & (m["conflict"] > max_conflict)).sum())
    logger.info(
        f"filter_integrations: kept {int(m['kept'].sum()):,}/{len(m):,} integrations across "
        f"{m['clone'].nunique()} clones ({n_dup} with high allele conflict, e.g. duplicated integrations)"
    )
    if plot or save:
        from .plotting import plot_integrations

        ax = plot_integrations(df)
        finish_plot(ax.figure, show and bool(plot), save)
    return df
