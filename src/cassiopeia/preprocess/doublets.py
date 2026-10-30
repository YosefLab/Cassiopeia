"""Explicit doublet detection from lineage tracing data."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from ._resolve import resolve_alleles
from ._utils import ALLELE_PASS, finish_plot, passing, record_params, set_pass, umi_counts

Method = Literal["allele_conflict", "clone_mixture"]


def _pass_without(df: pd.DataFrame, exclude: Iterable[str]) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("pass_") and c not in set(exclude)]
    if not cols:
        return np.ones(len(df), dtype=bool)
    return df[cols].all(axis=1).to_numpy()


def _drop_conflicting_intbcs(
    df: pd.DataFrame, res: pd.DataFrame, max_conflict: float
) -> tuple[pd.DataFrame, int]:
    """Drop (cell, intBC) rows of intBCs that conflict in most of their clone's cells.

    Duplicated integrations hold two alleles in most cells of their clone,
    while doublets add conflicts to only a few cells per intBC.
    """
    from .qc import _cell_clones

    clone = res["cellBC"].map(_cell_clones(df)).astype(object)
    intbc = res["intBC"].astype(str)
    rate = res["conflict"].astype(float).groupby([clone, intbc]).mean()
    noisy = set(rate.index[(rate > max_conflict).to_numpy()])
    drop = np.fromiter(
        ((c, i) in noisy for c, i in zip(clone, intbc, strict=True)), dtype=bool, count=len(res)
    )
    return res[~drop], len(noisy)


def doublet_scores(
    df: pd.DataFrame,
    methods: Iterable[Method] | Literal["auto"] = "auto",
    *,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    min_informative: int = 2,
    max_intbc_conflict: float | None = 0.2,
    hooks: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Per-cell doublet scores (no thresholds applied).

    Scores:
        * ``allele_conflict``: fraction of the cell's informative integrations
          (intBCs with at least ``min_umi`` UMIs) holding two incompatible
          alleles. Two cells from different lineages carry different alleles
          at the same intBC, so conflicts nominate doublets. NaN when fewer
          than ``min_informative`` integrations are observed. Minor alleles
          removed by :func:`cassiopeia.pp.filter_alleles`
          (``min_allele_frac``, ``allow_allele_conflicts``) are included.
          After :func:`cassiopeia.pp.filter_integrations`, only its kept
          integrations are informative; before it, intBCs that conflict in
          more than ``max_intbc_conflict`` of their clone's cells, such as
          duplicated integrations, are not.
        * ``clone_mixture``: fraction of the cell's UMIs on intBCs that belong
          to another clone's signature but not its own (1 for NovaClone
          multiplets). Only with more than one clone.
        * ``hooks``: name -> function returning a per-cell score Series.

    Args:
        df: Allele table.
        methods: Scores to compute (``"allele_conflict"``,
            ``"clone_mixture"``), or ``"auto"``: allele conflicts, plus clone
            mixtures when there is more than one clone.
        min_umi: UMIs an intBC needs to be informative, and a minor allele
            needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.
        min_informative: Informative integrations needed to score
            ``allele_conflict``.
        max_intbc_conflict: Before :func:`cassiopeia.pp.filter_integrations`,
            ignore intBCs whose alleles conflict in more than this fraction
            of their clone's cells. ``None`` uses every intBC.
        hooks: Extra scores: name -> function(df) returning a per-cell Series
            indexed by cellBC.

    Returns:
        DataFrame indexed by cellBC with one ``<method>`` column per score.
        ``attrs['ignored_intbcs']`` counts the intBCs ignored for
        ``allele_conflict`` because of ``max_intbc_conflict``.
    """
    have_clones = "clone" in df.columns
    n_clones = len(df.attrs.get("clone_intbcs", {})) if have_clones else 1
    if methods == "auto":
        methods = ["allele_conflict"] + (["clone_mixture"] if n_clones > 1 else [])
    methods = list(methods)
    # minor alleles (removed only by the allele choice) are conflict evidence
    base = df.assign(**{"pass": _pass_without(df, ["pass_doublet", *ALLELE_PASS])})
    # Score every cell with usable molecules, including cells that fail
    # clone / integration based filters (e.g. NovaClone multiplets).
    cell_level = ["pass_doublet", "pass_integration", "pass_intbc_frac", "pass_clone", *ALLELE_PASS]
    mol = df.assign(**{"pass": _pass_without(df, cell_level)})
    cells = passing(mol)["cellBC"].astype(str).unique()
    out = pd.DataFrame(index=pd.Index(cells, name="cellBC"))

    if "allele_conflict" in methods:
        d = passing(base)
        res = resolve_alleles(d, min_umi=min_umi, min_minor_frac=min_minor_frac, use_pass=False)
        res = res[res["n_umis"] >= min_umi]
        res["cellBC"] = res["cellBC"].astype(str)
        if "pass_integration" not in df.columns and max_intbc_conflict is not None:
            res, n_ignored = _drop_conflicting_intbcs(df, res, max_intbc_conflict)
            out.attrs["ignored_intbcs"] = n_ignored
        g = res.groupby("cellBC")
        n_inf = g.size()
        n_conf = g["conflict"].sum()
        score = (n_conf / n_inf).where(n_inf >= min_informative)
        out["allele_conflict"] = score.reindex(out.index)

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
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    min_informative: int = 2,
    max_intbc_conflict: float | None = 0.2,
    hooks: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
    hook_thresholds: dict[str, float] | None = None,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Score and flag doublets.

    Computes :func:`doublet_scores` and flags a cell as a doublet if any score
    exceeds its threshold (``allele_conflict > conflict_threshold``,
    ``clone_mixture > mixture_threshold``, hooks above ``hook_thresholds``
    (default 0.5)).
    With the default ``methods="auto"``, allele conflicts are always used and
    clone mixtures when there is more than one clone, so the same call works
    for data with many clones and for data where all cells share intBCs.

    Can run before or after :func:`cassiopeia.pp.filter_integrations`. After
    it, only the kept integrations are used for ``allele_conflict``. Before
    it, intBCs that conflict in more than ``max_intbc_conflict`` of their
    clone's cells, such as duplicated integrations, are not used, so they do
    not flag the whole clone; :func:`cassiopeia.pp.filter_integrations` then
    leaves the flagged cells out of its conflict rates.

    Args:
        df: Allele table.
        methods: Scores to use, or ``"auto"`` (see :func:`doublet_scores`).
        conflict_threshold: Threshold for ``allele_conflict``.
        mixture_threshold: Threshold for ``clone_mixture``.
        min_umi: UMIs an intBC needs to be informative, and a minor allele
            needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.
        min_informative: Integrations needed to score ``allele_conflict``.
        max_intbc_conflict: Before :func:`cassiopeia.pp.filter_integrations`,
            ignore intBCs whose alleles conflict in more than this fraction
            of their clone's cells. ``None`` uses every intBC.
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
        max_intbc_conflict=max_intbc_conflict,
        hooks=hooks,
    )
    thresholds = {"allele_conflict": conflict_threshold, "clone_mixture": mixture_threshold}
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
        df[f"doublet_{col}"] = cells.map(scores[col]).to_numpy()
    set_pass(df, "doublet", ~cells.map(flag).fillna(False).astype(bool).to_numpy())
    used = [m for m in thresholds if m in scores.columns]
    record_params(df, **{f"doublet_{m}_threshold": thresholds[m] for m in used})
    record_params(df, doublet_methods=used)
    ignored = scores.attrs.get("ignored_intbcs")
    if ignored is not None:
        record_params(df, doublet_max_intbc_conflict=max_intbc_conflict)
    n = int(flag.sum())
    logger.info(
        f"filter_doublets: {n:,}/{len(flag):,} cells flagged ({', '.join(used)})"
        + (
            f"; allele_conflict ignores {ignored} intBCs that conflict in more than "
            f"{max_intbc_conflict:.0%} of their clone's cells"
            if ignored
            else ""
        )
    )
    if plot or save:
        from ._plotting import plot_doublet_scores

        finish_plot(plot_doublet_scores(df), plot, save)
    return df
