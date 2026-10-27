"""Molecule table -> character matrix (TreeData) and legacy allele tables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from ._resolve import resolve_alleles
from ._utils import finish_plot, passing, site_columns


def _is_unedited(token: str) -> bool:
    return token == "None" or "[None]" in token


def _character_order(df: pd.DataFrame, cells_df: pd.DataFrame) -> list[tuple[str, str]]:
    """(clone, intBC) pairs in output order."""
    kept = df.attrs.get("integrations")
    if kept is None:
        kept = df.attrs.get("clone_intbcs")
    pairs = []
    if kept:
        for clone in sorted(kept, key=lambda x: (len(str(x)), str(x))):
            for intbc in kept[clone]:
                pairs.append((str(clone), str(intbc)))
    else:
        for intbc in sorted(cells_df["intBC"].astype(str).unique()):
            pairs.append(("0", intbc))
    return pairs


def alleles_to_characters(
    df: pd.DataFrame,
    *,
    min_allele_frac: float = 0.5,
    min_umi: int = 1,
    unknown: Literal["missing", "keep"] = "missing",
    compute_priors: bool = True,
    cell_meta: pd.DataFrame | None = None,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
):
    """Build the character matrix from the molecule table.

    Uses molecules passing all filters. For each (cell, intBC) the alleles of
    its molecules are merged (molecules that differ only where a site is
    missing are combined) and the major allele is used if it holds at least
    ``min_allele_frac`` of the UMIs; otherwise the intBC's sites are missing.

    Each character is one site of one kept integration (``"{intBC}-{site}"``).
    States: ``0`` unedited, ``-1`` missing, ``1..k`` edits ordered by
    frequency. With several clones, cells get ``-1`` for characters of
    integrations outside their clone; ``uns['clone_characters']`` lists each
    clone's characters.

    Args:
        df: Filtered molecule table.
        min_allele_frac: UMI fraction the major allele needs.
        min_umi: Minimum UMIs supporting a (cell, intBC).
        unknown: For insertion_lookup sites, treat insertions that are not in
            the lookup as ``"missing"`` (default) or ``"keep"`` them as states.
        compute_priors: Store empirical state frequencies in ``uns['priors']``.
        cell_meta: Extra per-cell columns (indexed by cellBC) for ``obs``.
        plot: Plot the fraction of edited / unedited / missing per character.
        show: Show the plot.
        save: Save the plot to this path.

    Returns:
        :class:`treedata.TreeData` with ``obsm['characters']`` (integer states),
        ``obs`` (clone, QC metrics, doublet scores), ``uns['character_states']``
        (character, state, allele), ``uns['priors']`` and ``uns['preprocess']``
        (construct, thresholds, read statistics and per-cell QC for all cells).
    """
    import treedata as td

    from .qc import cell_metrics

    sites = site_columns(df)
    d = passing(df)
    res = resolve_alleles(d, sites, use_pass=False)
    res = res[(res["n_umis"] >= min_umi)]
    res = res.assign(cellBC=res["cellBC"].astype(str), intBC=res["intBC"].astype(str))
    ok = (res["major_umis"] / res["n_umis"]) >= min_allele_frac
    known = df.attrs.get("site_states", {})

    cells = sorted(res["cellBC"].unique())
    cell_index = pd.Index(cells, name="cellBC")
    pairs = _character_order(df, res)
    intbcs_needed = sorted({i for _, i in pairs})
    res = res[res["intBC"].isin(intbcs_needed)]
    if "clone" in df.columns:
        cc = d.drop_duplicates("cellBC")
        cell_clone = pd.Series(
            cc["clone"].astype(object).to_numpy(), index=cc["cellBC"].astype(str).to_numpy()
        )
    else:
        cell_clone = pd.Series("0", index=cells, dtype=object)

    columns: dict[str, np.ndarray] = {}
    state_rows = []
    priors: dict[str, dict[str, float]] = {}
    clone_chars: dict[str, list[str]] = {}
    seen = set()
    row_pos = cell_index.get_indexer(res["cellBC"])
    for clone, intbc in pairs:
        sub_mask = (res["intBC"] == intbc).to_numpy()
        for site in sites:
            name = f"{intbc}-{site}"
            clone_chars.setdefault(clone, []).append(name)
            if name in seen:
                continue
            seen.add(name)
            tokens = np.full(len(cells), None, dtype=object)
            vals = res.loc[sub_mask, site].astype(object).to_numpy()
            good = ok.to_numpy()[sub_mask]
            tokens[row_pos[sub_mask]] = np.where(good, vals, None)
            states = np.full(len(cells), -1, dtype=np.int32)
            present = pd.notna(tokens)
            uniq, counts = (
                np.unique(tokens[present].astype(str), return_counts=True)
                if present.any()
                else ([], [])
            )
            edits = []
            for tok, cnt in zip(uniq, counts, strict=False):
                if _is_unedited(tok):
                    continue
                if unknown == "missing" and site in known and tok not in known[site]:
                    continue
                edits.append((tok, cnt))
            edits.sort(key=lambda x: (-x[1], x[0]))
            mapping = {tok: i + 1 for i, (tok, _) in enumerate(edits)}
            tok_str = np.array([str(t) if t is not None else "" for t in tokens], dtype=object)
            unedited = np.array([_is_unedited(t) for t in tok_str], dtype=bool) & present
            states[unedited] = 0
            for tok, s in mapping.items():
                states[tok_str == tok] = s
                state_rows.append((name, s, tok))
            if unedited.any():
                state_rows.append((name, 0, "None"))
            idx = len(columns)
            columns[name] = states
            if compute_priors and mapping:
                # string keys so the priors survive h5ad/h5td round trips
                n_edit = sum(c for _, c in edits)
                priors[str(idx)] = {str(mapping[t]): float(c) / n_edit for t, c in edits}
    chars = pd.DataFrame(columns, index=cell_index)
    obs = cell_metrics(df).reindex(cells)
    if "clone" not in obs.columns:
        obs["clone"] = cell_clone.reindex(cells).to_numpy()
    if cell_meta is not None:
        obs = obs.join(cell_meta, how="left", rsuffix="_meta")
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype(str).replace({"nan": None, "None": None})
    tdata = td.TreeData(obs=obs, obsm={"characters": chars})
    tdata.uns["character_states"] = pd.DataFrame(
        state_rows, columns=["character", "state", "allele"]
    )
    if compute_priors:
        tdata.uns["priors"] = priors
    tdata.uns["clone_characters"] = clone_chars
    all_cells = cell_metrics(df)
    all_cells.index = all_cells.index.astype(str)
    tdata.uns["preprocess"] = {
        "construct": str(df.attrs.get("construct", "")),
        "sites": list(sites),
        "qc_params": {
            k: (v if not isinstance(v, (list, tuple)) else list(map(str, v)))
            for k, v in df.attrs.get("qc_params", {}).items()
        },
        "read_stats": {
            k: v for k, v in df.attrs.get("read_stats", {}).items() if not isinstance(v, dict)
        },
        "cell_qc": _h5_safe(all_cells),
    }
    n_missing = float((chars.to_numpy() == -1).mean()) if chars.size else 0.0
    logger.info(
        f"alleles_to_characters: {len(cells):,} cells x {chars.shape[1]:,} characters "
        f"({n_missing:.0%} missing)"
    )
    if plot or save:
        from .plotting import plot_edit_frequency

        ax = plot_edit_frequency(tdata)
        finish_plot(ax.figure, show and bool(plot), save)
    return tdata


def _h5_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype) or out[c].dtype == object:
            out[c] = out[c].astype(str)
        elif out[c].dtype == bool:
            out[c] = out[c].astype(bool)
    return out


def to_allele_table(df: pd.DataFrame, use_pass: bool = True) -> pd.DataFrame:
    """Legacy Cassiopeia allele table (one row per cell, intBC and allele).

    The result works with :func:`cassiopeia.pp.convert_alleletable_to_character_matrix`
    and the other legacy converters: columns ``cellBC``, ``intBC``,
    ``allele``, ``r1..rN`` (one per site), ``lineageGrp`` (clone), ``UMI``
    (number of UMIs) and ``readCount``.
    """
    sites = site_columns(df)
    d = passing(df) if use_pass else df
    keys = ["cellBC", "intBC"] + sites
    agg = d.groupby(keys, observed=True, dropna=False).agg(
        UMI=("n_reads", "size"), readCount=("n_reads", "sum")
    )
    out = agg.reset_index()
    for i, s in enumerate(sites):
        out[f"r{i + 1}"] = out[s].astype(object)
    out["allele"] = out[[f"r{i + 1}" for i in range(len(sites))]].astype(str).agg("".join, axis=1)
    if "clone" in d.columns:
        cc = d.drop_duplicates("cellBC")
        cell_clone = pd.Series(
            cc["clone"].astype(object).to_numpy(), index=cc["cellBC"].astype(str).to_numpy()
        )
        out["lineageGrp"] = out["cellBC"].astype(str).map(cell_clone)
    else:
        out["lineageGrp"] = 0
    cols = (
        ["cellBC", "intBC", "allele"]
        + [f"r{i + 1}" for i in range(len(sites))]
        + ["lineageGrp", "UMI", "readCount"]
    )
    out = out[cols]
    out["cellBC"] = out["cellBC"].astype(str)
    out["intBC"] = out["intBC"].astype(str)
    return out
