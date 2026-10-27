"""QC metrics computed from the allele table."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError

from ._utils import CELL_PASS as _CELL_PASS
from ._utils import SINGLE_CLONE, passing, row_passing, site_columns, umi_counts
from .integrations import integration_metrics


def _expected_intbcs(df: pd.DataFrame) -> dict[str, set[str]] | None:
    """Expected intBCs per clone: kept integrations, else clone signatures."""
    kept = df.attrs.get("integrations") or df.attrs.get("clone_intbcs")
    if not kept:
        return None
    return {str(c): {str(i) for i in v} for c, v in kept.items()}


def _cell_clones(df: pd.DataFrame) -> pd.Series:
    """Clone of each cellBC (str index), None when unassigned.

    Without a ``clone`` column all cells form one clone, except cells failing
    ``pass_min_umi``, which are unassigned as they would be by
    :func:`cassiopeia.pp.call_clones`.
    """
    first = df.drop_duplicates("cellBC")
    cells = first["cellBC"].astype(str).to_numpy()
    if "clone" not in df.columns:
        clone = pd.Series(SINGLE_CLONE, index=cells, dtype=object)
        if "pass_min_umi" in first.columns:
            clone[~first["pass_min_umi"].to_numpy(dtype=bool)] = None
        return clone
    clone = first["clone"].astype(object).to_numpy()
    return pd.Series(clone, index=cells, dtype=object).where(pd.notna(clone), None)


def intbc_fraction(df: pd.DataFrame) -> pd.Series:
    """Per cell, the fraction of its clone's expected intBCs that are detected.

    Expected intBCs are the clone's kept integrations
    (:func:`cassiopeia.pp.filter_integrations`), or its signature from
    :func:`cassiopeia.pp.call_clones`. Without a ``clone`` column all cells
    share the kept integrations. intBCs are detected on rows passing the
    allele filters. NaN for cells without a clone (unassigned, multiplets;
    without a ``clone`` column, cells failing ``pass_min_umi``).
    """
    expected = _expected_intbcs(df)
    if expected is None:
        raise PreprocessError(
            "The intBC fraction needs the expected intBCs: run pp.filter_integrations first."
        )
    clones = _cell_clones(df)
    if "clone" not in df.columns:
        expected = {SINGLE_CLONE: set().union(*expected.values())}
    d = row_passing(df)
    pairs = d[["cellBC", "intBC"]].drop_duplicates()
    cell = pairs["cellBC"].astype(str).to_numpy()
    intbc = pairs["intBC"].astype(str).to_numpy()
    clone = clones.reindex(cell).to_numpy()
    hit = np.fromiter(
        (i in expected.get(str(c), ()) for i, c in zip(intbc, clone, strict=True)),
        dtype=bool,
        count=len(pairs),
    )
    detected = pd.Series(hit.astype(float), index=cell).groupby(level=0).sum()
    size = np.array([len(expected.get(str(c), ())) for c in clones.to_numpy()], dtype=float)
    size[size == 0] = np.nan
    return pd.Series(detected.reindex(clones.index).fillna(0).to_numpy() / size, index=clones.index)


def cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell QC metrics.

    One row per cell barcode in the allele table, including cells that fail
    filters, so the metrics can be used to choose thresholds. UMIs, reads and
    intBCs are counted over the rows that pass the row-level filters run so
    far (:func:`~cassiopeia.pp.correct_intbcs` with a whitelist,
    :func:`~cassiopeia.pp.filter_alleles`,
    :func:`~cassiopeia.pp.filter_integrations`); cell-level filters do not
    change them. Columns appear once the step that computes them has run:

    * ``n_umis``: UMIs (molecules) in the cell.
    * ``n_reads``: reads supporting those UMIs.
    * ``reads_per_umi``: ``n_reads / n_umis``. Low values mean the library
      was sequenced shallowly and single-read UMIs are common.
    * ``n_intbcs``: distinct intBCs detected.
    * ``intbc_frac``: fraction of the intBCs expected for the cell's clone
      that are detected. The expected intBCs are the clone's kept
      integrations (:func:`~cassiopeia.pp.filter_integrations`; all kept
      integrations without clones) or, before that, its NovaClone signature.
      NaN for cells without a clone (unassigned or ``"multiplet"``).
    * ``clone``: clone from :func:`~cassiopeia.pp.call_clones`.
    * ``doublet_<method>``: doublet scores from
      :func:`~cassiopeia.pp.filter_doublets` (see
      :func:`~cassiopeia.pp.doublet_scores`), and ``n_informative``, the
      integrations used for the ``allele_conflict`` score.
    * ``pass_min_umi``, ``pass_intbc_frac``, ``pass_clone``,
      ``pass_doublet``: results of the cell-level filters
      (:func:`~cassiopeia.pp.filter_cells`, :func:`~cassiopeia.pp.call_clones`,
      :func:`~cassiopeia.pp.filter_doublets`).
    * ``passed_qc``: True when the cell passes every cell-level filter.

    Args:
        df: Allele table.

    Returns:
        DataFrame indexed by cellBC.
    """
    cells = pd.Index(df["cellBC"].astype(str).unique()).sort_values()
    d = row_passing(df)
    dcell = d["cellBC"].astype(str)
    out = pd.DataFrame(
        {
            "n_umis": pd.Series(umi_counts(d), index=d.index).groupby(dcell).sum(),
            "n_reads": d.groupby(dcell)["n_reads"].sum(),
        }
    ).reindex(cells)
    out = out.fillna(0).astype(int)
    out["reads_per_umi"] = out["n_reads"] / out["n_umis"].replace(0, np.nan)
    out["n_intbcs"] = d.groupby(dcell)["intBC"].nunique().reindex(cells).fillna(0).astype(int)
    if _expected_intbcs(df) is not None:
        out["intbc_frac"] = intbc_fraction(df).reindex(cells)
    first = df.drop_duplicates("cellBC")
    first = first.set_index(first["cellBC"].astype(str))
    extra = [
        c
        for c in first.columns
        if c == "clone" or c.startswith("doublet_") or c == "n_informative" or c in _CELL_PASS
    ]
    if "sample" in first.columns:
        extra = ["sample"] + extra
    for c in extra:
        out[c] = first[c].reindex(cells).to_numpy()
    if "clone" in out.columns:
        out["clone"] = out["clone"].astype(object)
    flags = [c for c in _CELL_PASS if c in out.columns]
    out["passed_qc"] = out[flags].all(axis=1) if flags else True
    out.index.name = "cellBC"
    return out


def clone_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-clone QC metrics.

    One row per clone from :func:`~cassiopeia.pp.call_clones`, plus
    ``"multiplet"`` for cells NovaClone assigned to several clones; cells
    without a clone are left out. Without a ``clone`` column all cells
    passing :func:`~cassiopeia.pp.filter_cells` ``min_umi`` form one clone,
    ``"0"``. Columns appear once the step that computes them has run:

    * ``n_cells``: cells assigned to the clone.
    * ``n_cells_passed``: cells passing every cell-level filter so far
      (``passed_qc`` in :func:`~cassiopeia.pp.qc.cell_metrics`).
    * ``n_doublets``: cells flagged by :func:`~cassiopeia.pp.filter_doublets`.
    * ``n_integrations``: number of intBCs that make up the clone: its kept
      integrations after :func:`~cassiopeia.pp.filter_integrations`,
      otherwise its NovaClone signature.
    * ``intBCs``: those intBCs, as a sorted list.
    * ``median_umis``: median UMIs per cell (``n_umis`` in
      :func:`~cassiopeia.pp.qc.cell_metrics`).
    * ``mean_intbc_frac``: mean fraction of the clone's intBCs detected per
      cell (``intbc_frac``).

    Args:
        df: Allele table.

    Returns:
        DataFrame indexed by clone, largest clones first.
    """
    cells = cell_metrics(df)
    clone = (
        cells["clone"].astype(object)
        if "clone" in cells.columns
        else _cell_clones(df).reindex(cells.index)
    )
    cells = cells[clone.notna().to_numpy()]
    clone = clone[clone.notna()].astype(str)
    g = cells.groupby(clone, sort=False)
    out = pd.DataFrame({"n_cells": g.size()})
    if any(c in cells.columns for c in _CELL_PASS):
        out["n_cells_passed"] = g["passed_qc"].sum().astype(int)
    if "pass_doublet" in cells.columns:
        out["n_doublets"] = (~cells["pass_doublet"].astype(bool)).groupby(clone).sum()
    expected = _expected_intbcs(df) or {}
    out["n_integrations"] = [len(expected.get(c, ())) for c in out.index]
    out["intBCs"] = [sorted(expected.get(c, ())) for c in out.index]
    out["median_umis"] = g["n_umis"].median()
    if "intbc_frac" in cells.columns:
        out["mean_intbc_frac"] = g["intbc_frac"].mean()
    out.index.name = "clone"
    order = sorted(out.index, key=lambda c: (c == "multiplet", -out.at[c, "n_cells"], c))
    return out.loc[order]


def _intbc_umis(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, intBC) UMIs, reads and pass status (for the allele filter plot).

    ``umis_ok`` counts UMIs passing the read filters of
    :func:`cassiopeia.pp.filter_alleles` (all UMIs before it has run).
    """
    keys = [df["cellBC"].astype(str), df["intBC"].astype(str)]
    umis = pd.Series(umi_counts(df), index=df.index)
    ok = np.ones(len(df), dtype=bool)
    for c in ("pass_reads_per_umi", "pass_allele_frac"):
        if c in df.columns:
            ok &= df[c].to_numpy()
    out = pd.DataFrame(
        {
            "n_umis": umis.groupby(keys).sum(),
            "umis_ok": (umis * ok).groupby(keys).sum(),
            "n_reads": df.groupby(keys)["n_reads"].sum(),
        }
    )
    out["reads_per_umi"] = out["n_reads"] / out["n_umis"]
    out["pass"] = df.groupby(keys)["pass"].any() if "pass" in df.columns else True
    out.index.names = ["cellBC", "intBC"]
    return out.reset_index()


__all__ = [
    "cell_metrics",
    "clone_metrics",
    "compute_empirical_indel_priors",
    "integration_metrics",
    "intbc_fraction",
]


def compute_empirical_indel_priors(
    df: pd.DataFrame,
    grouping_variables: list[str] | None = None,
    cut_sites: list[str] | None = None,
) -> pd.DataFrame:
    """Empirical indel frequencies, for use as priors.

    Counts the independent groups in which each indel occurs and divides by
    the number of groups. By default each intBC is a group, which holds for a
    clonal population; group by more columns (e.g. ``["clone", "intBC"]``)
    when the same intBC occurs in several clones.

    Args:
        df: Allele table (rows passing all filters are used), or a legacy
            allele table with one column per cut site.
        grouping_variables: Columns defining independent groups (default
            ``["intBC"]``).
        cut_sites: Site columns. Default: the construct's sites, or columns
            named ``r1``, ``r2``, ... of a legacy allele table.

    Returns:
        DataFrame indexed by indel with ``count`` (groups containing it) and
        ``freq`` (fraction of groups).
    """
    grouping_variables = list(grouping_variables or ["intBC"])
    if cut_sites is None:
        if df.attrs.get("sites"):
            cut_sites = site_columns(df)
        else:
            cut_sites = [c for c in df.columns if re.fullmatch(r"r\d+", str(c))]
    groups = passing(df).groupby(grouping_variables, observed=True)
    counts: dict[str, int] = {}
    for _, g in groups:
        for allele in pd.unique(g[cut_sites].astype(object).to_numpy().ravel()):
            if isinstance(allele, str) and "none" not in allele.lower():
                counts[allele] = counts.get(allele, 0) + 1
    out = pd.DataFrame({"count": pd.Series(counts, dtype=int)})
    out["freq"] = out["count"] / max(groups.ngroups, 1)
    out.index.name = "indel"
    return out
