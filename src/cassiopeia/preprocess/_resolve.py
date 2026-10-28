"""Resolve the alleles of each (cell, intBC): merge compatible alleles and flag conflicts."""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from ._utils import passing, site_columns, umi_counts


@numba.njit(cache=True)
def _resolve_groups(starts, codes, umis, min_umi, min_minor_frac):
    """Greedy clustering of alleles within each group.

    Alleles (rows of ``codes``, -1 = missing) within a group are sorted by
    UMI count (descending). Each allele joins the first cluster it is
    compatible with (equal at every site where both are called), filling in
    missing sites; otherwise it starts a new cluster. The first cluster is
    the major allele. A group is in conflict if another cluster has at least
    ``min_umi`` UMIs and a UMI fraction of at least ``min_minor_frac``.
    """
    n_groups = starts.shape[0] - 1
    n_sites = codes.shape[1]
    major = np.full((n_groups, n_sites), -1, dtype=np.int32)
    major_umis = np.zeros(n_groups, dtype=np.int64)
    total_umis = np.zeros(n_groups, dtype=np.int64)
    conflict_umis = np.zeros(n_groups, dtype=np.int64)
    conflict = np.zeros(n_groups, dtype=np.bool_)
    n_clusters = np.zeros(n_groups, dtype=np.int32)
    reps = np.empty((64, n_sites), dtype=np.int32)
    rep_umis = np.empty(64, dtype=np.int64)
    # every cluster: group, alleles and UMIs (at most one cluster per allele row)
    cl_group = np.empty(codes.shape[0], dtype=np.int64)
    cl_codes = np.empty((codes.shape[0], n_sites), dtype=np.int32)
    cl_umis = np.empty(codes.shape[0], dtype=np.int64)
    n_cl = 0
    for g in range(n_groups):
        s, e = starts[g], starts[g + 1]
        k = 0
        tot = 0
        for i in range(s, e):
            tot += umis[i]
            placed = False
            for c in range(k):
                ok = True
                for j in range(n_sites):
                    a = codes[i, j]
                    b = reps[c, j]
                    if a >= 0 and b >= 0 and a != b:
                        ok = False
                        break
                if ok:
                    for j in range(n_sites):
                        if reps[c, j] < 0:
                            reps[c, j] = codes[i, j]
                    rep_umis[c] += umis[i]
                    placed = True
                    break
            if not placed:
                if k < 64:
                    for j in range(n_sites):
                        reps[k, j] = codes[i, j]
                    rep_umis[k] = umis[i]
                    k += 1
                else:
                    rep_umis[k - 1] += umis[i]
        # major = largest cluster (clusters can grow out of order)
        best = 0
        for c in range(1, k):
            if rep_umis[c] > rep_umis[best]:
                best = c
        for j in range(n_sites):
            major[g, j] = reps[best, j]
        major_umis[g] = rep_umis[best]
        total_umis[g] = tot
        n_clusters[g] = k
        cu = 0
        flag = False
        for c in range(k):
            if c == best:
                continue
            cu += rep_umis[c]
            if rep_umis[c] >= min_umi and rep_umis[c] >= min_minor_frac * tot:
                flag = True
        conflict_umis[g] = cu
        conflict[g] = flag
        for c in range(k):
            cl_group[n_cl] = g
            for j in range(n_sites):
                cl_codes[n_cl, j] = reps[c, j]
            cl_umis[n_cl] = rep_umis[c]
            n_cl += 1
    return (
        major,
        major_umis,
        total_umis,
        conflict_umis,
        conflict,
        n_clusters,
        cl_group[:n_cl],
        cl_codes[:n_cl],
        cl_umis[:n_cl],
    )


def resolve_alleles(
    df: pd.DataFrame,
    sites: list[str] | None = None,
    min_umi: int = 2,
    min_minor_frac: float = 0.1,
    use_pass: bool = True,
    return_clusters: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Per (cell, intBC) resolved alleles and conflict flags.

    Args:
        df: Allele table.
        sites: Site columns (default: ``df.attrs['sites']``).
        min_umi: UMIs a minor allele needs to count as a conflict.
        min_minor_frac: UMI fraction a minor allele needs to count as a conflict.
        use_pass: Only use rows passing all filters so far.
        return_clusters: Also return every allele cluster of each (cellBC,
            intBC) with its ``umis`` and site alleles.

    Returns:
        DataFrame with one row per (cellBC, intBC): ``n_umis``, ``major_umis``,
        ``conflict_umis``, ``conflict``, ``n_alleles`` and the resolved site
        alleles (categoricals with the same categories as ``df``).
    """
    sites = sites or site_columns(df)
    d = passing(df) if use_pass else df
    cols = ["cellBC", "intBC"]
    if len(d) == 0:
        out = pd.DataFrame(
            columns=cols
            + ["n_umis", "major_umis", "conflict_umis", "conflict", "n_alleles"]
            + sites
        )
        if return_clusters:
            return out, pd.DataFrame(columns=["cellBC", "intBC", "umis"] + sites)
        return out
    allele = d.groupby(sites, observed=True, dropna=False, sort=False).ngroup().to_numpy()
    key = pd.DataFrame(
        {
            "cell": d["cellBC"].cat.codes.to_numpy(),
            "intbc": d["intBC"].cat.codes.to_numpy(),
            "allele": allele,
            "umis": umi_counts(d),
        }
    )
    counts = key.groupby(["cell", "intbc", "allele"], sort=False)["umis"].sum().reset_index()
    counts = counts.sort_values(
        ["cell", "intbc", "umis", "allele"], ascending=[True, True, False, True]
    )
    first = pd.Series(np.arange(len(d)), index=allele).groupby(level=0).first()
    rows = first.reindex(counts["allele"].to_numpy()).to_numpy()
    code_mat = (
        np.stack([d[s].cat.codes.to_numpy().astype(np.int32)[rows] for s in sites], axis=1)
        if sites
        else np.zeros((len(counts), 0), np.int32)
    )
    gkey = (
        counts["cell"].to_numpy().astype(np.int64) * (int(counts["intbc"].max()) + 1)
        + counts["intbc"].to_numpy()
    )
    change = np.flatnonzero(np.diff(gkey)) + 1
    starts = np.concatenate([[0], change, [len(gkey)]]).astype(np.int64)
    (
        major,
        major_umis,
        total,
        conflict_umis,
        conflict,
        n_clusters,
        cl_group,
        cl_codes,
        cl_umis,
    ) = _resolve_groups(
        starts,
        np.ascontiguousarray(code_mat),
        counts["umis"].to_numpy().astype(np.int64),
        int(min_umi),
        float(min_minor_frac),
    )
    g_cell = counts["cell"].to_numpy()[starts[:-1]]
    g_intbc = counts["intbc"].to_numpy()[starts[:-1]]
    out = pd.DataFrame(
        {
            "cellBC": pd.Categorical.from_codes(g_cell, categories=d["cellBC"].cat.categories),
            "intBC": pd.Categorical.from_codes(g_intbc, categories=d["intBC"].cat.categories),
            "n_umis": total,
            "major_umis": major_umis,
            "conflict_umis": conflict_umis,
            "conflict": conflict,
            "n_alleles": n_clusters,
        }
    )
    for j, s in enumerate(sites):
        out[s] = pd.Categorical.from_codes(major[:, j], categories=d[s].cat.categories)
    if not return_clusters:
        return out
    clusters = pd.DataFrame(
        {
            "cellBC": pd.Categorical.from_codes(
                g_cell[cl_group], categories=d["cellBC"].cat.categories
            ),
            "intBC": pd.Categorical.from_codes(
                g_intbc[cl_group], categories=d["intBC"].cat.categories
            ),
            "umis": cl_umis,
        }
    )
    for j, s in enumerate(sites):
        clusters[s] = pd.Categorical.from_codes(cl_codes[:, j], categories=d[s].cat.categories)
    return out, clusters
