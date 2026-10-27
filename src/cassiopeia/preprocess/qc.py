"""QC metrics computed from the molecule table."""

from __future__ import annotations

import pandas as pd

from ._utils import passing
from .integrations import integration_metrics

_CELL_PASS = ("pass_min_umi", "pass_intbc_frac", "pass_clone", "pass_doublet")


def cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell QC metrics.

    Returns a DataFrame indexed by cellBC with ``n_molecules`` / ``n_reads``
    / ``reads_per_umi`` over all molecules, ``n_umis`` / ``n_intbcs`` over
    passing molecules, ``clone``, doublet scores and cell-level ``pass_*``
    flags (``passed_qc`` combines them).
    """
    cell = df["cellBC"].astype(str)
    g = df.groupby(cell, sort=True)
    out = pd.DataFrame({"n_molecules": g.size(), "n_reads": g["n_reads"].sum()})
    out["reads_per_umi"] = out["n_reads"] / out["n_molecules"]
    d = passing(df)
    if len(d) < len(df):
        gp = d.groupby(d["cellBC"].astype(str))
        out["n_umis"] = gp.size().reindex(out.index).fillna(0).astype(int)
        out["n_intbcs"] = gp["intBC"].nunique().reindex(out.index).fillna(0).astype(int)
    else:
        out["n_umis"] = out["n_molecules"]
        out["n_intbcs"] = g["intBC"].nunique()
    first = df.drop_duplicates("cellBC")
    first = first.set_index(first["cellBC"].astype(str))
    extra = [
        c
        for c in first.columns
        if c == "clone" or c.startswith("doublet") or c == "n_informative" or c in _CELL_PASS
    ]
    if "sample" in first.columns:
        extra = ["sample"] + extra
    for c in extra:
        out[c] = first[c].reindex(out.index).to_numpy()
    if "clone" in out.columns:
        out["clone"] = out["clone"].astype(object)
    flags = [c for c in _CELL_PASS if c in out.columns]
    out["passed_qc"] = out[flags].all(axis=1) if flags else True
    out.index.name = "cellBC"
    return out


__all__ = ["cell_metrics", "integration_metrics", "molecule_metrics"]


def molecule_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, intBC) UMIs, reads and pass status (used by the UMI/reads plot)."""
    g = df.groupby([df["cellBC"].astype(str), df["intBC"].astype(str)], sort=False)
    out = pd.DataFrame({"n_umis": g.size(), "n_reads": g["n_reads"].sum()})
    out["reads_per_umi"] = out["n_reads"] / out["n_umis"]
    if "pass" in df.columns:
        out["pass"] = g["pass"].any()
    else:
        out["pass"] = True
    out.index.names = ["cellBC", "intBC"]
    return out.reset_index()
