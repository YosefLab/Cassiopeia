"""QC metrics computed from the allele table."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError
from cassiopeia.utils import _format_priors

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


def _intbc_fraction(df: pd.DataFrame) -> pd.Series:
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
      was sequenced shallowly.
    * ``n_intbcs``: distinct intBCs detected.
    * ``intbc_frac``: fraction of the intBCs expected for the cell's clone
      that are detected. The expected intBCs are the clone's kept
      integrations (:func:`~cassiopeia.pp.filter_integrations`; all kept
      integrations without clones) or, before that, its NovaClone signature.
      NaN for cells without a clone (unassigned or ``"multiplet"``).
    * ``clone``: clone from :func:`~cassiopeia.pp.call_clones`.
    * ``doublet_<method>``: doublet scores from
      :func:`~cassiopeia.pp.filter_doublets` (see
      :func:`~cassiopeia.pp.doublet_scores`).
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
        out["intbc_frac"] = _intbc_fraction(df).reindex(cells)
    first = df.drop_duplicates("cellBC")
    first = first.set_index(first["cellBC"].astype(str))
    extra = [
        c for c in first.columns if c == "clone" or c.startswith("doublet_") or c in _CELL_PASS
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
      (``passed_qc`` in :func:`~cassiopeia.pp.cell_metrics`).
    * ``n_doublets``: cells flagged by :func:`~cassiopeia.pp.filter_doublets`.
    * ``n_integrations``: number of intBCs that make up the clone: its kept
      integrations after :func:`~cassiopeia.pp.filter_integrations`,
      otherwise its NovaClone signature.
    * ``intBCs``: those intBCs, as a sorted list.
    * ``median_umis``: median UMIs per cell (``n_umis`` in
      :func:`~cassiopeia.pp.cell_metrics`).
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


def _allele_umis(df: pd.DataFrame) -> pd.DataFrame:
    """Per allele (cell, intBC, site alleles) summary for the plots.

    UMIs, reads, mean reads per UMI, ``allele_frac``, and whether the allele
    passes every filter of :func:`cassiopeia.pp.filter_alleles` (``kept``).
    """
    names = [c for c in ("sample", "cellBC", "intBC") if c in df.columns] + site_columns(df)
    codes = pd.DataFrame({c: df[c].cat.codes.to_numpy() for c in names}, copy=False)
    allele = codes.groupby(names, sort=False).ngroup().to_numpy()
    _, first = np.unique(allele, return_index=True)
    n_umis = np.bincount(allele, weights=umi_counts(df))
    n_reads = np.bincount(allele, weights=df["n_reads"].to_numpy().astype(np.float64))

    bad = np.zeros(len(df), dtype=bool)
    for c in (
        "pass_umis_per_allele",
        "pass_avg_reads_per_umi",
        "pass_allele_frac",
        "pass_allele_conflict",
    ):
        if c in df.columns:
            bad |= ~df[c].to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            "n_umis": n_umis.astype(np.int64),
            "n_reads": n_reads.astype(np.int64),
            "reads_per_umi": n_reads / np.maximum(n_umis, 1),
            "allele_frac": df["allele_frac"].to_numpy()[first],
            "kept": np.bincount(allele, weights=bad) == 0,
        }
    )


__all__ = [
    "cell_metrics",
    "clone_metrics",
    "compute_indel_priors",
    "integration_metrics",
]


def _character_intbc(name: str) -> str:
    """The intBC of a character named ``{intBC}-{site}``."""
    return name.rsplit("-", 1)[0] if "-" in name else name


def _allele_lookup(meta) -> dict[tuple[str, str], str]:
    """``(character, state) -> allele`` from ``character_states`` (empty if absent)."""
    states = pd.DataFrame(meta.get("character_states", {}))
    if not len(states):
        return {}
    keys = zip(states["character"].astype(str), states["state"].astype(str), strict=True)
    return dict(zip(keys, states["allele"].astype(str), strict=True))


def _state_priors(tdata, characters_key: str, freq: pd.Series) -> dict[int, dict[str, float]]:
    """Priors ``{character index: {state: probability}}``; a state's is its allele's frequency."""
    chars = tdata.obsm[characters_key]
    skip = {str(tdata.uns.get("missing_state", -1)), str(tdata.uns.get("unmodified_state", 0))}
    allele_of = _allele_lookup(tdata.uns)
    freq = freq.to_dict()
    priors = {}
    for i, name in enumerate(chars.columns):
        states = {}
        for v in pd.unique(chars[name].to_numpy()):
            for s in v if isinstance(v, tuple) else (v,):
                s = str(s)
                p = None if s in skip else freq.get(allele_of.get((str(name), s), s))
                if p is not None:
                    states[s] = float(p)
        if states:
            priors[i] = states
    return priors


def _priors_from_characters(
    data, grouping_variables: list[str], characters_key: str
) -> tuple[dict[str, int], int]:
    """Indel occurrences per group in a character matrix (TreeData or DataFrame)."""
    if hasattr(data, "obsm"):
        chars, meta, obs = data.obsm[characters_key], data.uns, data.obs
    else:
        chars, meta, obs = data, data.attrs, None
    missing = str(meta.get("missing_state", -1))
    unedited = str(meta.get("unmodified_state", 0))
    allele_of = _allele_lookup(meta)
    by = [g for g in grouping_variables if g != "intBC"]
    if by:
        if obs is None or any(g not in obs.columns for g in by):
            raise ValueError(f"Grouping variables {by} must be columns of tdata.obs.")
        cell_group = obs[by].astype(str).agg("|".join, axis=1).to_numpy()
    else:
        cell_group = np.zeros(len(chars), dtype=int)
    columns: dict[str, list[str]] = {}
    for name in chars.columns:
        columns.setdefault(_character_intbc(str(name)), []).append(name)
    counts: dict[str, int] = {}
    n_groups = 0
    for group in pd.unique(cell_group):
        rows = cell_group == group
        for names in columns.values():
            seen: set[str] = set()
            observed = False
            for name in names:
                for v in pd.unique(chars.loc[rows, name].to_numpy()):
                    for s in v if isinstance(v, tuple) else (v,):
                        s = str(s)
                        if s == missing:
                            continue
                        observed = True
                        if s != unedited:
                            seen.add(allele_of.get((str(name), s), s))
            n_groups += observed
            for allele in seen:
                counts[allele] = counts.get(allele, 0) + 1
    return counts, n_groups


def compute_indel_priors(
    data,
    grouping_variables: list[str] | None = None,
    cut_sites: list[str] | None = None,
    characters_key: str = "characters",
) -> pd.DataFrame:
    """Empirical indel frequencies, for use as priors.

    Counts the independent groups in which each indel occurs and divides by
    the number of groups. By default each intBC is a group, which holds for a
    clonal population; group by more columns (e.g. ``["clone", "intBC"]``)
    when the same intBC occurs in several clones.

    Accepts the allele table, a legacy allele table with one column per cut
    site, or a character matrix from
    :func:`cassiopeia.pp.alleles_to_characters` (a TreeData, or the DataFrame
    from ``return_type="dataframe"``). For a character matrix, edits are
    mapped back to their alleles with ``character_states``, the intBC of a
    character is taken from its name (``{intBC}-{site}``), and grouping
    variables other than ``"intBC"`` are columns of ``tdata.obs``.

    For a TreeData, the priors are also stored in ``tdata.uns["priors"]`` as
    ``{character index: {state: probability}}``, the probability of a state
    being the frequency of its indel. The solvers (e.g.
    :func:`cassiopeia.solver.greedy`) use them by default.

    Args:
        data: Allele table (rows passing all filters are used), legacy allele
            table, or character matrix (TreeData or DataFrame).
        grouping_variables: Variables defining independent groups (default
            ``["intBC"]``).
        cut_sites: (allele tables) Site columns. Default: the construct's
            sites, or columns named ``r1``, ``r2``, ... of a legacy allele table.
        characters_key: (TreeData) Key of the character matrix in ``obsm``.

    Returns:
        DataFrame indexed by indel with ``count`` (groups containing it) and
        ``freq`` (fraction of groups).
    """
    grouping_variables = list(grouping_variables or ["intBC"])
    is_matrix = hasattr(data, "obsm") or (
        isinstance(data, pd.DataFrame)
        and "intBC" not in data.columns
        and "missing_state" in data.attrs
    )
    if is_matrix:
        counts, n_groups = _priors_from_characters(data, grouping_variables, characters_key)
    else:
        df = data
        if cut_sites is None:
            if df.attrs.get("sites"):
                cut_sites = site_columns(df)
            else:
                cut_sites = [c for c in df.columns if re.fullmatch(r"r\d+", str(c))]
        groups = passing(df).groupby(grouping_variables, observed=True)
        counts = {}
        for _, g in groups:
            for allele in pd.unique(g[cut_sites].astype(object).to_numpy().ravel()):
                if isinstance(allele, str) and "none" not in allele.lower():
                    counts[allele] = counts.get(allele, 0) + 1
        n_groups = groups.ngroups
    out = pd.DataFrame({"count": pd.Series(counts, dtype=int)})
    out["freq"] = out["count"] / max(n_groups, 1)
    out.index.name = "indel"
    if hasattr(data, "obsm"):
        data.uns["priors"] = _format_priors(_state_priors(data, characters_key, out["freq"]))
    return out
