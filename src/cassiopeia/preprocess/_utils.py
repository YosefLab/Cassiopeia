"""Shared helpers for the preprocessing steps."""

from __future__ import annotations

import gzip
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger


def default_threads(n_threads: int | None) -> int:
    """Number of threads: explicit value, else SLURM allocation, else up to 8."""
    if n_threads is not None and n_threads > 0:
        return int(n_threads)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return max(1, min(8, os.cpu_count() or 1))


def read_lines(path: str | Path) -> list[str]:
    """Read non-empty lines (first column) from a plain or gzipped text file."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split()[0].split(",")[0])
    return out


def as_list(values: str | Path | Iterable[str] | None) -> list[str] | None:
    """Barcodes given as a file path or an iterable."""
    if values is None:
        return None
    if isinstance(values, (str, Path)):
        return read_lines(values)
    return [str(v) for v in values]


def record_params(df: pd.DataFrame, **params: Any) -> None:
    """Store resolved thresholds in ``df.attrs['qc_params']``."""
    qc = dict(df.attrs.get("qc_params", {}))
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        qc[k] = v
    df.attrs["qc_params"] = qc


def set_pass(df: pd.DataFrame, name: str, values: np.ndarray | pd.Series) -> None:
    """Set a ``pass_<name>`` column and update the combined ``pass`` column."""
    col = f"pass_{name}"
    df[col] = np.asarray(values, dtype=bool)
    pass_cols = [c for c in df.columns if c.startswith("pass_")]
    df["pass"] = df[pass_cols].all(axis=1).to_numpy()


def passing(df: pd.DataFrame, ignore: Iterable[str] = ()) -> pd.DataFrame:
    """Rows passing all filters so far, except the ``ignore`` columns."""
    ignore = set(ignore)
    if ignore:
        cols = [c for c in df.columns if c.startswith("pass_") and c not in ignore]
        return df[df[cols].all(axis=1).to_numpy()] if cols else df
    if "pass" in df.columns:
        return df[df["pass"].to_numpy()]
    return df


#: Clone label used when the table has no ``clone`` column (all cells form one clone).
SINGLE_CLONE = "0"

#: Cell-level ``pass_*`` columns; the others are row-level.
CELL_PASS = ("pass_min_umi", "pass_intbc_frac", "pass_clone", "pass_doublet")

#: ``pass_*`` columns that choose among the alleles of a (cell, intBC). Cell,
#: intBC and conflict statistics (UMI counts, clones, integrations, doublets)
#: ignore them, so minor alleles still count there.
ALLELE_PASS = ("pass_allele_frac", "pass_allele_conflict")


def row_passing(df: pd.DataFrame) -> pd.DataFrame:
    """Rows passing the row-level quality filters (cell flags and allele choice ignored)."""
    cols = [
        c
        for c in df.columns
        if c.startswith("pass_") and c not in CELL_PASS and c not in ALLELE_PASS
    ]
    if cols:
        return df[df[cols].all(axis=1).to_numpy()]
    return df


def umi_counts(df: pd.DataFrame) -> np.ndarray:
    """UMIs each row stands for: ``n_umis`` after :func:`cassiopeia.pp.collapse_umis`, else 1."""
    if "n_umis" in df.columns:
        return df["n_umis"].to_numpy().astype(np.int64)
    return np.ones(len(df), dtype=np.int64)


def is_collapsed(df: pd.DataFrame) -> bool:
    """Whether rows are alleles (UMIs collapsed) rather than single UMIs."""
    return "n_umis" in df.columns


def reads_per_umi_hist(n_reads: np.ndarray) -> dict[str, list[int]]:
    """Number of UMIs (``umis``) with each read count (``reads``)."""
    reads, umis = np.unique(np.asarray(n_reads, dtype=np.int64), return_counts=True)
    return {"reads": reads.tolist(), "umis": umis.tolist()}


def allele_fractions(df: pd.DataFrame) -> np.ndarray:
    """Fraction of the UMIs of each row's (cell, intBC) that carry the row's allele.

    The fractions of the alleles of a (cell, intBC) sum to 1. Without UMIs
    (bulk data) reads are counted instead.
    """
    if len(df) == 0:
        return np.zeros(0, dtype=np.float32)
    pair_cols = [c for c in ("sample", "cellBC", "intBC") if c in df.columns]
    names = pair_cols + site_columns(df)
    codes = pd.DataFrame(
        {
            c: df[c].cat.codes.to_numpy()
            if isinstance(df[c].dtype, pd.CategoricalDtype)
            else pd.factorize(df[c], use_na_sentinel=False)[0]
            for c in names
        },
        copy=False,
    )
    if df.attrs.get("has_umis", True) or "n_umis" in df.columns:
        w = umi_counts(df).astype(np.float64)
    else:
        w = df["n_reads"].to_numpy().astype(np.float64)
    pair = codes.groupby(pair_cols, sort=False).ngroup().to_numpy()
    allele = codes.groupby(names, sort=False).ngroup().to_numpy()
    pair_total = np.bincount(pair, weights=w)[pair]
    allele_total = np.bincount(allele, weights=w)[allele]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(pair_total > 0, allele_total / pair_total, 0.0)
    return frac.astype(np.float32)


def site_columns(df: pd.DataFrame) -> list[str]:
    """Edit-site columns of an allele table."""
    sites = df.attrs.get("sites")
    if sites:
        return [s for s in sites if s in df.columns]
    reserved = {
        "sample",
        "cellBC",
        "intBC",
        "UMI",
        "clone",
        "n_reads",
        "n_umis",
        "allele_frac",
        "pass",
    }
    inferred = [
        c
        for c in df.columns
        if c not in reserved
        and not c.startswith(("pass_", "doublet"))
        and isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    if inferred:
        return inferred
    raise ValueError(
        "Cannot determine the edit-site columns; expected df.attrs['sites'] "
        "(set by pp.call_alleles)."
    )


def finish_plot(fig, plot: bool, save: str | Path | None):
    """Save and/or show a figure created by a step, then close it."""
    import matplotlib.pyplot as plt

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
        logger.info(f"Saved plot to {save}")
    if plot:
        plt.show()
    plt.close(fig)


def copy_attrs(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    """Copy attrs (pandas drops them on many operations)."""
    dst.attrs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in src.attrs.items()}
    return dst
