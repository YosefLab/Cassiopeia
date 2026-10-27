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


def passing(df: pd.DataFrame) -> pd.DataFrame:
    """Rows passing all filters so far."""
    if "pass" in df.columns:
        return df[df["pass"].to_numpy()]
    return df


def site_columns(df: pd.DataFrame) -> list[str]:
    """Edit-site columns of a molecule table."""
    sites = df.attrs.get("sites")
    if sites:
        return [s for s in sites if s in df.columns]
    reserved = {"sample", "cellBC", "intBC", "UMI", "clone", "n_reads", "allele_frac", "pass"}
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


def finish_plot(fig, show: bool, save: str | Path | None):
    """Save and/or show a figure created by a step."""
    import matplotlib.pyplot as plt

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
        logger.info(f"Saved plot to {save}")
    if show:
        plt.show()
    elif save:
        plt.close(fig)


def copy_attrs(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    """Copy attrs (pandas drops them on many operations)."""
    dst.attrs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in src.attrs.items()}
    return dst
