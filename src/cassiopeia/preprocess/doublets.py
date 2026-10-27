"""Explicit doublet detection from lineage tracing data."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from ._resolve import resolve_alleles
from ._utils import finish_plot, passing, record_params, set_pass, umi_counts

Method = Literal["allele_conflict", "clone_mixture", "umi_outlier"]


def _pass_without(df: pd.DataFrame, exclude: Iterable[str]) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("pass_") and c not in set(exclude)]
    if not cols:
        return np.ones(len(df), dtype=bool)
    return df[cols].all(axis=1).to_numpy()


def doublet_scores(
    df: pd.DataFrame,
    methods: Iterable[Method] | Literal["auto"] = "auto",
    *,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    min_informative: int = 2,
    hooks: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Per-cell doublet scores (no thresholds applied).

    Scores:
        * ``allele_conflict``: fraction of the cell's informative integrations
          (kept by :func:`cassiopeia.pp.filter_integrations`, with at least
          ``min_umi`` UMIs) holding two incompatible alleles. Two cells from
          different lineages carry different alleles at the same intBC, so
          conflicts nominate doublets. NaN when fewer than ``min_informative``
          integrations are observed.
        * ``clone_mixture``: fraction of the cell's UMIs on intBCs that belong
          to another clone's signature but not its own (1 for NovaClone
          multiplets). Only with more than one clone.
        * ``umi_outlier``: robust z-score of log UMIs within the clone.
        * ``hooks``: name -> function returning a per-cell score Series.

    Returns:
        DataFrame indexed by cellBC with one ``<method>`` column per score.
    """
    have_clones = "clone" in df.columns
    n_clones = len(df.attrs.get("clone_intbcs", {})) if have_clones else 1
    if methods == "auto":
        methods = ["allele_conflict"] + (["clone_mixture"] if n_clones > 1 else [])
    methods = list(methods)
    base = df.assign(**{"pass": _pass_without(df, ["pass_doublet"])})
    # Score every cell with usable molecules, including cells that fail
    # clone / integration based filters (e.g. NovaClone multiplets).
    cell_level = ["pass_doublet", "pass_integration", "pass_intbc_frac", "pass_clone"]
    mol = df.assign(**{"pass": _pass_without(df, cell_level)})
    cells = passing(mol)["cellBC"].astype(str).unique()
    out = pd.DataFrame(index=pd.Index(cells, name="cellBC"))

    if "allele_conflict" in methods:
        d = passing(base)
        res = resolve_alleles(d, min_umi=min_umi, min_minor_frac=min_minor_frac, use_pass=False)
        res = res[res["n_umis"] >= min_umi]
        res["cellBC"] = res["cellBC"].astype(str)
        g = res.groupby("cellBC")
        n_inf = g.size()
        n_conf = g["conflict"].sum()
        score = (n_conf / n_inf).where(n_inf >= min_informative)
        out["allele_conflict"] = score.reindex(out.index)
        out["n_informative"] = n_inf.reindex(out.index).fillna(0).astype(int)

    if "clone_mixture" in methods:
        if not have_clones:
            raise PreprocessError("clone_mixture needs pp.call_clones first.")
        sig = df.attrs.get("clone_intbcs", {})
        owner: dict[str, set[str]] = {}
        for clone, intbcs in sig.items():
            for i in intbcs:
                owner.setdefault(i, set()).add(str(clone))
        # molecules passing molecule-level filters (before integration filtering)
        d = passing(mol)
        t = (
            pd.Series(umi_counts(d), index=d.index, name="umis")
            .groupby([d["cellBC"], d["intBC"]], observed=True)
            .sum()
            .reset_index()
        )
        t["cellBC"] = t["cellBC"].astype(str)
        t["intBC"] = t["intBC"].astype(str)
        cc = df.drop_duplicates("cellBC")
        cell_clone = pd.Series(
            cc["clone"].astype(object).to_numpy(), index=cc["cellBC"].astype(str).to_numpy()
        )
        t["clone"] = t["cellBC"].map(cell_clone).astype(object)
        foreign = np.array(
            [
                (i in owner) and (str(c) not in owner[i])
                for i, c in zip(t["intBC"].to_numpy(), t["clone"].to_numpy(), strict=True)
            ],
            dtype=bool,
        )
        t["foreign"] = t["umis"] * foreign
        g = t.groupby("cellBC")
        score = g["foreign"].sum() / g["umis"].sum()
        score = score.reindex(out.index)
        clone_of = cell_clone.reindex(out.index)
        score[clone_of.isna().to_numpy()] = np.nan  # unassigned cells: no signature to compare to
        score[(clone_of == "multiplet").to_numpy()] = 1.0
        out["clone_mixture"] = score

    if "umi_outlier" in methods:
        d = passing(base)
        umis = pd.Series(umi_counts(d), index=d.index).groupby(d["cellBC"], observed=True).sum()
        umis.index = umis.index.astype(str)
        if have_clones:
            cc = df.drop_duplicates("cellBC")
            clone = pd.Series(
                cc["clone"].astype(object).to_numpy(), index=cc["cellBC"].astype(str).to_numpy()
            )
            clone = clone.reindex(umis.index).fillna("none")
        else:
            clone = pd.Series("0", index=umis.index)
        logu = np.log10(umis.astype(float))
        med = logu.groupby(clone).transform("median")
        mad = (logu - med).abs().groupby(clone).transform("median") * 1.4826
        out["umi_outlier"] = ((logu - med) / mad.replace(0, np.nan)).reindex(out.index)

    for name, fn in (hooks or {}).items():
        s = fn(df)
        s.index = s.index.astype(str)
        out[name] = s.reindex(out.index)
    return out


def filter_doublets(
    df: pd.DataFrame,
    methods: Iterable[Method] | Literal["auto"] = "auto",
    *,
    conflict_threshold: float = 0.2,
    mixture_threshold: float = 0.2,
    umi_outlier_threshold: float = 3.0,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    min_informative: int = 2,
    hooks: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
    hook_thresholds: dict[str, float] | None = None,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Score and flag doublets.

    Computes :func:`doublet_scores` and flags a cell as a doublet if any score
    exceeds its threshold (``allele_conflict > conflict_threshold``,
    ``clone_mixture > mixture_threshold``, ``umi_outlier >
    umi_outlier_threshold``, hooks above ``hook_thresholds`` (default 0.5)).
    With the default ``methods="auto"``, allele conflicts are always used and
    clone mixtures when there is more than one clone, so the same call works
    for data with many clones and for data where all cells share intBCs.

    Run after :func:`cassiopeia.pp.filter_integrations` so that duplicated or
    unreliable integrations do not create false conflicts.

    Args:
        df: Allele table.
        methods: Scores to use, or ``"auto"``.
        conflict_threshold: Threshold for ``allele_conflict``.
        mixture_threshold: Threshold for ``clone_mixture``.
        umi_outlier_threshold: Threshold for ``umi_outlier``.
        min_umi: UMIs a minor allele needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.
        min_informative: Integrations needed to score ``allele_conflict``.
        hooks: Extra scores: name -> function(df) returning a per-cell Series.
        hook_thresholds: Thresholds for hook scores (default 0.5).
        plot: Plot score distributions with thresholds.
        save: Save the plot to this path.

    Returns:
        The allele table with ``doublet_<method>`` score columns and
        ``pass_doublet`` (False for cells flagged by any method).
    """
    df = df.copy()
    scores = doublet_scores(
        df,
        methods,
        min_umi=min_umi,
        min_minor_frac=min_minor_frac,
        min_informative=min_informative,
        hooks=hooks,
    )
    thresholds = {
        "allele_conflict": conflict_threshold,
        "clone_mixture": mixture_threshold,
        "umi_outlier": umi_outlier_threshold,
    }
    for name in hooks or {}:
        thresholds[name] = (hook_thresholds or {}).get(name, 0.5)
    flag = pd.Series(False, index=scores.index)
    for name, thr in thresholds.items():
        if name in scores.columns:
            s = scores[name]
            if s.dtype == bool:
                flag |= s.fillna(False)
            else:
                flag |= (s > thr).fillna(False)
    cells = df["cellBC"].astype(str)
    for col in scores.columns:
        name = col if col == "n_informative" else f"doublet_{col}"
        df[name] = cells.map(scores[col]).to_numpy()
    set_pass(df, "doublet", ~cells.map(flag).fillna(False).astype(bool).to_numpy())
    used = [m for m in thresholds if m in scores.columns]
    record_params(df, **{f"doublet_{m}_threshold": thresholds[m] for m in used})
    record_params(df, doublet_methods=used)
    n = int(flag.sum())
    logger.info(f"filter_doublets: {n:,}/{len(flag):,} cells flagged ({', '.join(used)})")
    if plot or save:
        from ._plotting import plot_doublet_scores

        finish_plot(plot_doublet_scores(df), plot, save)
    return df
