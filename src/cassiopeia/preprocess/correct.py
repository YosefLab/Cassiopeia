"""intBC error correction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from . import _native
from ._utils import (
    allele_fractions,
    as_list,
    copy_attrs,
    default_threads,
    finish_plot,
    is_collapsed,
    record_params,
    set_pass,
    site_columns,
    umi_counts,
)


def _merge_duplicate_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows sharing (cell, intBC, UMI), or (cell, intBC, allele), after correction."""
    if is_collapsed(df):
        from .alleles import _collapse

        return _collapse(df)
    keys = ["cellBC", "intBC", "UMI"] if df["UMI"].notna().any() else None
    if keys is None:
        return df
    if "sample" in df.columns:
        keys = ["sample"] + keys
    dup = df.duplicated(keys, keep=False).to_numpy()
    if not dup.any():
        return df
    d = df[dup]
    order = d.sort_values("n_reads", ascending=False, kind="stable")
    best = order.drop_duplicates(keys, keep="first")
    total = d.groupby(keys, observed=True)["n_reads"].sum()
    best = best.set_index(keys)
    best["n_reads"] = total.reindex(best.index).to_numpy().astype(np.int32)
    best = best.reset_index()[df.columns]
    out = pd.concat([df[~dup], best], ignore_index=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) and not isinstance(
            out[c].dtype, pd.CategoricalDtype
        ):
            out[c] = pd.Categorical(out[c], categories=df[c].cat.categories)
    return copy_attrs(df, out)


def correct_intbcs(
    df: pd.DataFrame,
    whitelist: str | Path | Iterable[str] | None = None,
    *,
    max_dist: int = 2,
    min_ratio: float = 5.0,
    n_threads: int | None = None,
    plot: bool = True,
    save: str | Path | None = None,
) -> pd.DataFrame:
    """Correct intBC sequencing errors.

    With a ``whitelist``, each intBC is mapped to the unique closest
    whitelist entry within ``max_dist`` edits; intBCs that cannot be placed
    fail ``pass_intbc``. Without a whitelist, each intBC is merged into the
    closest intBC within ``max_dist`` edits that has at least ``min_ratio``
    times as many UMIs (error variants are much rarer than real intBCs).
    Rows that end up with the same (cell, intBC, UMI), or the same (cell,
    intBC, allele) after :func:`cassiopeia.pp.collapse_umis`, are merged,
    and ``allele_frac`` is recomputed for the corrected (cell, intBC) pairs.

    Args:
        df: Allele table from :func:`cassiopeia.pp.call_alleles`.
        whitelist: Known intBCs (list or file).
        max_dist: Maximum Levenshtein distance for a correction.
        min_ratio: (no whitelist) Minimum UMI ratio of the target intBC.
        n_threads: Threads.
        plot: Plot intBC abundance with corrected intBCs highlighted.
        save: Save the plot to this path.

    Returns:
        Allele table with corrected ``intBC``.
    """
    wl = as_list(whitelist)
    counts = pd.Series(umi_counts(df), index=df.index).groupby(df["intBC"], observed=True).sum()
    seqs = [str(s) for s in counts.index]
    mapping = _native.correct_intbcs(
        seqs,
        counts.to_numpy().astype(np.int64),
        wl,
        max_dist,
        min_ratio,
        default_threads(n_threads),
    )
    if wl:
        target = np.array([wl[i] if i >= 0 else None for i in mapping], dtype=object)
    else:
        target = np.array([seqs[i] for i in mapping], dtype=object)
    remap = dict(zip(seqs, target, strict=True))
    n_changed = int(sum(1 for s, t in remap.items() if t is not None and t != s))
    n_unplaced = int(sum(1 for t in remap.values() if t is None))
    new = df["intBC"].astype(object).map(remap)
    placed = new.notna().to_numpy()
    new = new.where(new.notna(), df["intBC"].astype(object))
    out = df.copy()
    cats = sorted(set(new.dropna().unique()))
    out["intBC"] = pd.Categorical(new, categories=cats)
    out = copy_attrs(df, out)
    if wl:
        set_pass(out, "intbc", placed)
    out = _merge_duplicate_molecules(out)
    out["allele_frac"] = allele_fractions(out)
    record_params(out, intbc_max_dist=max_dist, intbc_min_ratio=None if wl else min_ratio)
    logger.info(
        f"correct_intbcs: {len(seqs):,} intBCs, {n_changed:,} corrected"
        + (f", {n_unplaced:,} not in whitelist" if wl else "")
        + f" -> {out['intBC'].nunique():,} intBCs"
    )
    if plot or save:
        from ._plotting import plot_intbc_correction

        finish_plot(plot_intbc_correction(counts, remap), plot, save)
    site_columns(out)  # validate
    return out
