"""Allele table -> character matrix."""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd

from cassiopeia.mixins import logger

from ._resolve import resolve_alleles
from ._utils import SINGLE_CLONE, passing, site_columns


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
            pairs.append((SINGLE_CLONE, intbc))
    return pairs


def _site_tokens(
    res: pd.DataFrame,
    clusters: pd.DataFrame | None,
    sites: list[str],
    min_allele_frac: float,
) -> dict[str, np.ndarray]:
    """Per site, the allele of each (cell, intBC) row of ``res``.

    Entries are None (missing), an allele, or with ``clusters`` a tuple of
    alleles when several alleles hold at least ``min_allele_frac`` of the UMIs.
    """
    frac = (res["major_umis"] / res["n_umis"]).to_numpy()
    if clusters is None:
        ok = frac >= min_allele_frac
        return {
            s: np.where(ok, res[s].astype(object).to_numpy(), None).astype(object) for s in sites
        }
    row = pd.Series(
        np.arange(len(res)),
        index=pd.MultiIndex.from_arrays([res["cellBC"].astype(str), res["intBC"].astype(str)]),
    )
    pos = row.reindex(
        pd.MultiIndex.from_arrays([clusters["cellBC"].astype(str), clusters["intBC"].astype(str)])
    ).to_numpy()
    keep = ~np.isnan(pos)
    cl = clusters[keep]
    pos = pos[keep].astype(np.int64)
    ok = cl["umis"].to_numpy() >= min_allele_frac * res["n_umis"].to_numpy()[pos]
    cl, pos = cl[ok], pos[ok]
    out = {}
    for s in sites:
        vals = np.full(len(res), None, dtype=object)
        a = cl[s].astype(object).to_numpy()
        called = pd.notna(a)
        seen: dict[int, list] = {}
        for p, v in zip(pos[called], a[called], strict=True):
            seen.setdefault(int(p), []).append(v)
        for p, v in seen.items():
            uniq = sorted(set(v))
            vals[p] = uniq[0] if len(uniq) == 1 else tuple(uniq)
        out[s] = vals
    return out


def alleles_to_characters(
    df: pd.DataFrame,
    *,
    mapping: Literal["str", "int", "allele"] | dict = "str",
    min_allele_frac: float = 0.5,
    min_umi: int = 1,
    ambiguous: bool = False,
    unknown: Literal["missing", "keep"] = "missing",
    return_type: Literal["treedata", "dataframe"] = "treedata",
):
    """Build the character matrix from the allele table.

    Uses rows passing all filters. For each (cell, intBC) the alleles are
    merged (alleles that differ only where a site is missing are combined)
    and the major allele is used if it holds at least ``min_allele_frac`` of
    the UMIs; otherwise the intBC's sites are missing. With
    ``ambiguous=True``, every allele holding at least ``min_allele_frac`` of
    the UMIs is kept, and a site where they differ gets an ambiguous state:
    a tuple of states such as ``("1", "2")`` (lower ``min_allele_frac``, e.g.
    to 0.2, to keep minor alleles).

    Each character is one site of one kept integration (``"{intBC}-{site}"``).
    Edits are numbered by frequency within each character, and ``mapping``
    sets how states are written:

    * ``"str"``: ``"*"`` unedited, ``"-"`` missing, ``"1"``, ``"2"``, ... edits.
    * ``"int"``: ``0`` unedited, ``-1`` missing, ``1``, ``2``, ... edits.
    * ``"allele"``: ``"*"`` unedited, ``"-"`` missing and the allele itself
      (as in the allele table) for edits.
    * a dict from allele to state, e.g. to use the same states across
      samples. Unedited and missing are ``0`` and ``-1`` when all states are
      integers, ``"*"`` and ``"-"`` otherwise; alleles not in the dict are
      missing.

    The unedited and missing states are recorded (``uns`` or ``attrs``
    ``'unmodified_state'`` / ``'missing_state'``) so the solvers pick them up.
    With several clones, cells are missing for characters of integrations
    outside their clone, and ``'clone_characters'`` lists each clone's
    characters.

    Args:
        df: Filtered allele table.
        mapping: How states are written in the character matrix (see above).
        min_allele_frac: UMI fraction an allele needs.
        min_umi: Minimum UMIs supporting a (cell, intBC).
        ambiguous: Write sites where a cell holds several alleles as tuples of
            states instead of missing.
        unknown: For insertion_lookup sites, treat insertions that are not in
            the lookup as ``"missing"`` (default) or ``"keep"`` them as states.
        return_type: ``"treedata"`` or ``"dataframe"`` (the character matrix
            only, with the states and clone characters in ``attrs``).

    Returns:
        :class:`treedata.TreeData` with ``obsm['characters']``, ``obs`` (QC
        metrics, clone, doublet scores), ``uns['character_states']``
        (character, state, allele) and ``uns['preprocess']`` (construct,
        thresholds, read statistics and per-cell QC for all cells); or the
        character matrix as a DataFrame.
    """
    from .qc import cell_metrics

    if isinstance(mapping, dict):
        mapping_dict = dict(mapping)
        ints = all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in mapping_dict.values()
        )
        missing_state, unmodified_state = (-1, 0) if ints else ("-", "*")
    elif mapping in ("str", "int", "allele"):
        mapping_dict = None
        missing_state, unmodified_state = (-1, 0) if mapping == "int" else ("-", "*")
    else:
        raise ValueError(f"mapping must be 'str', 'int', 'allele' or a dict, not {mapping!r}.")
    if return_type not in ("treedata", "dataframe"):
        raise ValueError(f"return_type must be 'treedata' or 'dataframe', not {return_type!r}.")
    sites = site_columns(df)
    d = passing(df)
    res = resolve_alleles(d, sites, use_pass=False, return_clusters=ambiguous)
    res, clusters = res if ambiguous else (res, None)
    res = res[res["n_umis"] >= min_umi].reset_index(drop=True)
    res = res.assign(cellBC=res["cellBC"].astype(str), intBC=res["intBC"].astype(str))
    known = df.attrs.get("site_states", {})

    cells = sorted(res["cellBC"].unique())
    cell_index = pd.Index(cells, name="cellBC")
    pairs = _character_order(df, res)
    res = res[res["intBC"].isin({i for _, i in pairs})].reset_index(drop=True)
    tokens_by_site = _site_tokens(res, clusters, sites, min_allele_frac)
    row_pos = cell_index.get_indexer(res["cellBC"])

    columns: dict[str, np.ndarray] = {}
    state_rows = []
    clone_chars: dict[str, list[str]] = {}
    n_unmapped = 0
    for clone, intbc in pairs:
        sub = (res["intBC"] == intbc).to_numpy()
        for site in sites:
            name = f"{intbc}-{site}"
            clone_chars.setdefault(clone, []).append(name)
            if name in columns:
                continue
            lookup = known.get(site) if unknown == "missing" else None

            def usable(tok, lookup=lookup):
                return _is_unedited(tok) or lookup is None or tok in lookup

            tokens = np.full(len(cells), None, dtype=object)
            tokens[row_pos[sub]] = tokens_by_site[site][sub]
            is_tuple = np.fromiter(
                (isinstance(tok, tuple) for tok in tokens), dtype=bool, count=len(tokens)
            )
            for i in np.flatnonzero(is_tuple):
                parts = [x for x in tokens[i] if usable(x)]
                tokens[i] = None if not parts else parts[0] if len(parts) == 1 else tuple(parts)
                is_tuple[i] = len(parts) > 1
            scalar = pd.notna(tokens) & ~is_tuple
            counts = pd.Series(tokens[scalar], dtype=object).value_counts()
            if lookup is not None:
                bad = [x for x in counts.index if not usable(x)]
                if bad:
                    tokens[scalar & pd.Series(tokens, dtype=object).isin(bad).to_numpy()] = None
                    scalar = pd.notna(tokens) & ~is_tuple
                    counts = counts.drop(bad)
            counts = Counter(counts.to_dict())
            for tok in tokens[is_tuple]:
                counts.update(tok)
            edits = sorted(
                (x for x in counts if not _is_unedited(x)), key=lambda x: (-counts[x], x)
            )
            rank = {x: i + 1 for i, x in enumerate(edits)}
            if mapping_dict is not None:
                value = {x: mapping_dict.get(x, missing_state) for x in edits}
                n_unmapped += sum(counts[x] for x in edits if x not in mapping_dict)
            elif mapping == "int":
                value = rank
            elif mapping == "str":
                value = {x: str(s) for x, s in rank.items()}
            else:
                value = {x: x for x in edits}
            lut = {x: unmodified_state if _is_unedited(x) else value[x] for x in counts}
            col = np.full(len(cells), missing_state, dtype=object)
            col[scalar] = pd.Series(tokens[scalar], dtype=object).map(lut).to_numpy()
            for i in np.flatnonzero(is_tuple):
                parts = sorted(tokens[i], key=lambda x: 0 if _is_unedited(x) else rank[x])
                vals = []
                for v in (lut[x] for x in parts):
                    if v != missing_state and v not in vals:
                        vals.append(v)
                col[i] = missing_state if not vals else vals[0] if len(vals) == 1 else tuple(vals)
            if isinstance(missing_state, int) and not is_tuple.any():
                col = col.astype(np.int64)
            has_unedited = any(_is_unedited(x) for x in counts)
            columns[name] = col
            state_rows += [(name, value[t], t) for t in edits if value[t] != missing_state]
            if has_unedited:
                state_rows.append((name, unmodified_state, "None"))
    if n_unmapped:
        logger.warning(
            f"alleles_to_characters: {n_unmapped:,} cell-character entries have alleles that "
            "are not in `mapping`; they are missing."
        )
    chars = pd.DataFrame(columns, index=cell_index)
    states = pd.DataFrame(state_rows, columns=["character", "state", "allele"])
    has_clones = "clone" in df.columns
    values = chars.to_numpy()
    n_missing = float((values == missing_state).mean()) if chars.size else 0.0
    n_ambiguous = (
        int(sum(isinstance(v, tuple) for v in values.ravel())) if values.dtype == object else 0
    )
    logger.info(
        f"alleles_to_characters: {len(cells):,} cells x {chars.shape[1]:,} characters "
        f"({n_missing:.0%} missing"
        + (f", {n_ambiguous:,} ambiguous entries" if ambiguous else "")
        + ")"
    )
    if return_type == "dataframe":
        chars.attrs["missing_state"] = missing_state
        chars.attrs["unmodified_state"] = unmodified_state
        chars.attrs["character_states"] = {c: states[c].tolist() for c in states.columns}
        if has_clones:
            chars.attrs["clone_characters"] = clone_chars
        return chars

    import treedata as td

    obs = cell_metrics(df).reindex(cells)
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype(str).replace({"nan": None, "None": None})
    tdata = td.TreeData(obs=obs, obsm={"characters": chars})
    tdata.uns["character_states"] = states
    tdata.uns["missing_state"] = missing_state
    tdata.uns["unmodified_state"] = unmodified_state
    if has_clones:
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
    return tdata


def _h5_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype) or out[c].dtype == object:
            out[c] = out[c].astype(str)
        elif out[c].dtype == bool:
            out[c] = out[c].astype(bool)
    return out
