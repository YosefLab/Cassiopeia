"""QC plots for the preprocessing steps.

Each step draws its figure when ``plot=True`` (or ``save=`` is given).
:func:`plot_edit_frequency` is public; the others are drawn by the steps.
Thresholds are read from ``df.attrs['qc_params']``.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

from ._utils import row_passing, site_columns, umi_counts
from .qc import _allele_umis, cell_metrics

# Colors: categorical slot 1 for kept data, a recessive neutral for filtered
# data, slot 2 for flagged cells, neutral ink for thresholds and text.
KEPT = "#2a78d6"
FILTERED = "#bdbcb6"
FLAGGED = "#eb6834"
THRESH = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8984"
OUTLINE = "#c3c2b7"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Edit states: one categorical hue per frequent edit (fixed order), then neutrals.
EDIT_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
OTHER_EDITS = "#8a8984"
AMBIGUOUS = "#52514e"
UNEDITED = "#dcdbd5"
MISSING = "#ffffff"


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color="#ecebe6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)


def _plain_log(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == "log":
            a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            a.set_major_formatter(fmt)
            a.set_minor_formatter(NullFormatter())


def _vline(ax, x, label):
    if x is None or (ax.get_xscale() == "log" and x <= 0):
        return
    ax.axvline(x, color=THRESH, linestyle="--", linewidth=1.2, zorder=3)
    ax.annotate(
        label,
        xy=(x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=7,
        color=THRESH,
    )


def _hline(ax, y, label):
    if y is None or (ax.get_yscale() == "log" and y <= 0):
        return
    ax.axhline(y, color=THRESH, linestyle="--", linewidth=1.2, zorder=3)
    ax.annotate(
        label,
        xy=(1, y),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 3),
        textcoords="offset points",
        fontsize=7,
        color=THRESH,
        ha="right",
    )


def _threshold(ax, value, label=None, axis="x"):
    """Dashed threshold line, labelled in the legend if ``label`` (set the axis scale first)."""
    scale = ax.get_xscale() if axis == "x" else ax.get_yscale()
    if value is None or (scale == "log" and value <= 0):
        return
    line = ax.axvline if axis == "x" else ax.axhline
    line(value, color=THRESH, linestyle="--", linewidth=1.2, zorder=3, label=label)


def _params(data) -> dict:
    if hasattr(data, "uns"):
        return dict(data.uns.get("preprocess", {}).get("qc_params", {}))
    return dict(getattr(data, "attrs", {}).get("qc_params", {}))


def _new_ax(ax, figsize=(4, 3.4)):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return ax


def _subsample(ax, n, max_points):
    """Indices of a random subset of ``max_points`` of ``n`` points (noted on ``ax``), or None."""
    if n <= max_points:
        return None
    ax.text(
        0.01,
        0.01,
        f"random {max_points:,} of {n:,} shown",
        transform=ax.transAxes,
        fontsize=6,
        color=MUTED,
    )
    return np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))


def _scatter_pass(ax, x, y, ok, labels=("kept", "filtered"), max_points=100_000):
    """Kept and filtered points.

    Above ``max_points`` a random subset is drawn; the legend counts all points.
    """
    x, y = np.asarray(x), np.asarray(y)
    ok = np.asarray(ok, dtype=bool)
    n_ok, n_bad = int(ok.sum()), int((~ok).sum())
    idx = _subsample(ax, len(x), max_points)
    if idx is not None:
        x, y, ok = x[idx], y[idx], ok[idx]
    raster = len(x) > 50000
    size = 6 if len(x) > 5000 else 12
    ax.scatter(
        x[~ok],
        y[~ok],
        s=size,
        color=FILTERED,
        linewidths=0,
        alpha=0.7,
        rasterized=raster,
        label=f"{labels[1]} (n={n_bad:,})",
        zorder=2,
    )
    ax.scatter(
        x[ok],
        y[ok],
        s=size,
        color=KEPT,
        linewidths=0,
        alpha=0.7,
        rasterized=raster,
        label=f"{labels[0]} (n={n_ok:,})",
        zorder=2,
    )
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="lower right")


# ------------------------------------------------------------------ reads


def plot_align(res: dict, min_score: int | None = None):
    """Figure for :func:`cassiopeia.pp.align`."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), constrained_layout=True)
    ax = axes[0]
    reads = np.sort(np.asarray(res.get("cell_reads", [])))[::-1]
    if len(reads):
        ax.plot(np.arange(1, len(reads) + 1), reads, color=KEPT, linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax)
    _style(ax, "Barcode rank (corrected barcodes)", "Rank", "Reads")
    h = np.asarray(res.get("reads_per_umi_hist", []))
    nz = np.flatnonzero(h)
    plot_reads_per_umi({"reads": nz.tolist(), "umis": h[nz].tolist()}, ax=axes[1])
    ax = axes[2]
    s = np.asarray(res.get("score_hist", []))
    if s.size and s.sum():
        nz = np.flatnonzero(s)
        lo, hi = int(nz[0]), int(nz[-1]) + 1
        edges = np.unique(np.linspace(lo, hi, min(40, hi - lo) + 1).astype(int))
        binned = np.add.reduceat(s[lo:hi], edges[:-1] - lo)
        ax.bar(
            edges[:-1],
            binned,
            width=np.diff(edges),
            align="edge",
            color=KEPT,
            edgecolor="white",
            linewidth=0.5,
        )
        if min_score and min_score > 1:
            _vline(ax, min_score, "min score")
    _style(ax, "Alignment score", "Score", "Records")
    st = res.get("stats", {})
    if st:
        fig.suptitle(
            f"{st.get('reads', 0):,} reads -> {st.get('molecules', 0):,} molecules; "
            f"{st.get('reads_cell_corrected', 0):,} reads with corrected barcodes, "
            f"{st.get('reads_cell_invalid', 0):,} invalid",
            fontsize=9,
            color=INK,
            x=0.01,
            ha="left",
        )
    return fig


_READ_FATE = [
    ("no_intbc", "no intBC"),
    ("unaligned", "not aligned"),
    ("filtered_flag", "unmapped, secondary or supplementary"),
    ("filtered_mapq", "low MAPQ"),
    ("missing_cell", "no cell barcode"),
    ("missing_umi", "no UMI"),
    ("not_in_cells", "not in `cells`"),
]


def _plot_read_fate(df: pd.DataFrame, ax):
    st = df.attrs.get("read_stats", {})
    low = int(st.get("low_read_umi_records", 0))
    min_reads = _params(df).get("min_reads_per_umi")
    rows = [("used", int(st.get("records_used", 0)) - low)]
    rows.append((f"UMI with < {min_reads} reads", low))
    rows += [(label, int(st.get(key, 0))) for key, label in _READ_FATE]
    off = int(st.get("records", 0)) - int(st.get("records_on_target", 0))
    rows.append(("other contigs", off))
    rows = [r for i, r in enumerate(rows) if i == 0 or r[1] > 0]
    total = max(int(st.get("records", 0)), 1)
    y = np.arange(len(rows))[::-1]
    counts = np.array([n for _, n in rows], dtype=float)
    colors = [KEPT] + [FILTERED] * (len(rows) - 1)
    ax.barh(y, counts, color=colors, height=0.7, zorder=2)
    ax.set_yticks(y, [label for label, _ in rows], fontsize=8)
    for yi, n in zip(y, counts, strict=True):
        ax.annotate(
            f"{int(n):,} ({n / total:.0%})",
            xy=(n, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.set_xlim(0, counts.max() * 1.45 if counts.max() > 0 else 1)
    _style(ax, f"Read fate ({total:,} BAM records)", "Records", None)
    ax.grid(False, axis="y")


def plot_call_alleles(df: pd.DataFrame):
    """Figure for :func:`cassiopeia.pp.call_alleles`: reads per UMI, read fate, alleles."""
    plt = _plt()
    hist = df.attrs.get("read_stats", {}).get("reads_per_umi")
    n_sites = len(site_columns(df))
    width = float(np.clip(3.2 + 0.32 * n_sites, 4.0, 13.0))
    ratios = ([5.2] if hist else []) + [4.6, width]
    fig, axes = plt.subplots(
        1,
        len(ratios),
        figsize=(sum(ratios), 3.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": ratios},
    )
    if hist:
        plot_reads_per_umi(hist, _params(df).get("min_reads_per_umi"), ax=axes[0])
    _plot_read_fate(df, axes[-2])
    plot_edit_frequency(df, color_by="edit", ax=axes[-1])
    return fig


def plot_intbc_correction(counts: pd.Series, remap: dict):
    """Abundance rank of intBCs, highlighting corrected / unplaced ones."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)
    c = counts.sort_values(ascending=False)
    rank = np.arange(1, len(c) + 1)
    status = np.array(
        [
            "unplaced"
            if remap.get(str(i)) is None
            else ("kept" if remap.get(str(i)) == str(i) else "corrected")
            for i in c.index
        ]
    )
    for lab, color in (("corrected", FLAGGED), ("unplaced", FILTERED), ("kept", KEPT)):
        m = status == lab
        if m.any():
            ax.scatter(
                rank[m],
                c.to_numpy()[m],
                s=8,
                color=color,
                linewidths=0,
                label=f"{lab} (n={int(m.sum()):,})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "intBC abundance", "Rank", "UMIs")
    ax.legend(frameon=False, fontsize=7)
    return fig


def plot_reads_per_umi(hist: dict, min_reads: int | None = None, ax=None):
    """Histogram of reads per UMI (log-spaced bins), with ``min_reads``.

    ``hist`` holds the read counts (``reads``) and the number of UMIs with
    each (``umis``), e.g. ``df.attrs['read_stats']['reads_per_umi']``.
    Without ``min_reads`` (or with 1), all UMIs are drawn as kept.
    """
    ax = _new_ax(ax)
    reads = np.maximum(np.asarray(hist.get("reads", []), dtype=np.int64), 1)
    umis = np.asarray(hist.get("umis", []), dtype=np.float64)
    thr = int(min_reads) if min_reads else 1
    if len(reads):
        hi = int(reads.max()) + 1
        edges = np.round(np.logspace(0, np.log10(hi), 25)).astype(np.int64)
        edges = np.unique(np.concatenate([edges, [1, hi], [thr] if 1 < thr < hi else []]))
        edges = edges.astype(np.int64)
        idx = np.searchsorted(edges, reads, side="right") - 1
        counts = np.bincount(idx, weights=umis, minlength=len(edges) - 1)[: len(edges) - 1]
        left, width = edges[:-1], np.diff(edges)
        ok = left >= thr
        n_ok = int(umis[reads >= thr].sum())
        n_bad = int(umis.sum()) - n_ok
        kept = f"kept (n={n_ok:,})" if thr > 1 else f"UMIs (n={n_ok:,})"
        for sel, color, label in (
            (~ok, FILTERED, f"filtered (n={n_bad:,})" if n_bad else "_nolegend_"),
            (ok, KEPT, kept),
        ):
            if sel.any():
                ax.bar(
                    left[sel],
                    counts[sel],
                    width=width[sel],
                    align="edge",
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                    label=label,
                )
        ax.set_xscale("log")
        _plain_log(ax)
        if thr > 1:
            _threshold(ax, thr)
        ax.legend(frameon=False, fontsize=7, loc="best")
    _style(ax, "Reads per UMI", "Reads per UMI", "UMIs")
    ax.grid(False, axis="x")
    return ax


def plot_umi_reads(df: pd.DataFrame, ax=None, max_points: int = 100_000):
    """UMIs vs. mean reads per UMI per allele, colored by allele fraction.

    Dashed lines mark the thresholds of :func:`cassiopeia.pp.filter_alleles`
    (``min_allele_frac`` on the colorbar). Alleles with low fractions are drawn
    on top. Above ``max_points`` a random subset is drawn.
    """
    from matplotlib.colors import LinearSegmentedColormap

    ax = _new_ax(ax)
    m = _allele_umis(df)
    x = np.clip(m["n_umis"].to_numpy(), 0.8, None)
    y = m["reads_per_umi"].to_numpy()
    frac = m["allele_frac"].to_numpy(dtype=float)
    n, n_kept = len(m), int(m["kept"].sum())
    idx = _subsample(ax, n, max_points)
    if idx is not None:
        x, y, frac = x[idx], y[idx], frac[idx]
    order = np.argsort(-frac, kind="stable")
    cmap = LinearSegmentedColormap.from_list("allele_frac", SEQ[2:])
    sc = ax.scatter(
        x[order],
        y[order],
        c=frac[order],
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=6 if len(x) > 5000 else 12,
        linewidths=0,
        rasterized=len(x) > 50000,
        zorder=2,
    )
    _style(ax, f"Alleles: {n_kept:,} of {n:,} kept", "UMIs", "Mean reads per UMI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax)
    params = _params(df)
    thr = params.get("min_umi_per_allele")
    if thr is not None:
        _threshold(ax, thr - 0.5)
    thr = params.get("min_avg_reads_per_umi")
    if thr:
        _threshold(ax, thr, axis="y")
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, aspect=30)
    cbar.set_label("Allele fraction", fontsize=9, color=INK)
    cbar.ax.tick_params(colors=INK, labelsize=8)
    cbar.outline.set_visible(False)
    thr = params.get("min_allele_frac")
    if thr is not None:
        cbar.ax.axhline(thr, color=INK, linestyle="--", linewidth=1.5)
    return ax


def plot_filter_alleles(df: pd.DataFrame):
    """Figure for :func:`cassiopeia.pp.filter_alleles`."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    plot_umi_reads(df, ax=ax)
    return fig


# ----------------------------------------------------------------- cells


def plot_umi_intbcs(df: pd.DataFrame, ax=None):
    """UMIs vs. unique intBCs detected per cell, for :func:`cassiopeia.pp.filter_cells`.

    Counts are over rows passing the allele filters, so filtered cells are
    drawn where they were before cell filtering. Points are jittered
    vertically so integer intBC counts don't overlap.
    """
    from matplotlib.ticker import MaxNLocator

    ax = _new_ax(ax)
    params = _params(df)
    c = cell_metrics(df)
    n_umis, n_intbcs = c["n_umis"].to_numpy(), c["n_intbcs"].to_numpy()
    ok = c["passed_qc"].to_numpy().astype(bool) & (n_umis > 0)
    x = np.clip(n_umis, 0.8, None)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(c))
    _scatter_pass(ax, x, n_intbcs + jitter, ok, labels=("kept cells", "filtered cells"))
    _style(ax, "Cells", "UMIs", "intBCs detected")
    _vline(ax, params.get("min_umi"), "min UMIs")
    ax.set_xscale("log")
    _plain_log(ax)
    ax.set_ylim(bottom=-0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def plot_intbc_fraction(df: pd.DataFrame, ax=None):
    """Fraction of the expected intBCs detected per cell, with ``min_intbc_frac``."""
    from .qc import _intbc_fraction

    ax = _new_ax(ax)
    frac = _intbc_fraction(df).dropna()
    ax.hist(frac, bins=np.linspace(0, 1, 26), color=KEPT, edgecolor="white", linewidth=1)
    _style(ax, "intBC detection per cell", "Fraction of expected intBCs detected", "Cells")
    _vline(ax, _params(df).get("min_intbc_frac"), "min fraction")
    return ax


def plot_filter_cells(df: pd.DataFrame):
    """Figure for :func:`cassiopeia.pp.filter_cells`."""
    plt = _plt()
    has_frac = "pass_intbc_frac" in df.columns
    fig, axes = plt.subplots(
        1,
        2 if has_frac else 1,
        figsize=(8.5 if has_frac else 4.3, 3.4),
        constrained_layout=True,
        squeeze=False,
    )
    plot_umi_intbcs(df, ax=axes[0, 0])
    if has_frac:
        plot_intbc_fraction(df, ax=axes[0, 1])
    return fig


def plot_clones(df: pd.DataFrame, max_clones: int = 30, max_intbcs: int = 60):
    """Clone sizes and intBC detection per clone, for :func:`cassiopeia.pp.call_clones`."""
    plt = _plt()
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.6]}
    )
    sizes = df.drop_duplicates("cellBC")["clone"].value_counts()
    sizes = sizes[sizes > 0]
    ax = axes[0]
    show = sizes.head(max_clones)
    colors = [FLAGGED if str(i) == "multiplet" else KEPT for i in show.index]
    ax.bar(np.arange(len(show)), show.to_numpy(), color=colors, width=0.8)
    ax.set_xticks(np.arange(len(show)), [str(i) for i in show.index], rotation=90, fontsize=7)
    n_clones = int(sum(1 for i in sizes.index if str(i) != "multiplet"))
    _style(ax, f"Clone sizes ({n_clones} clones)", "Clone", "Cells")
    plot_clone_intbc_heatmap(
        df, value="detection", max_clones=max_clones, max_intbcs=max_intbcs, ax=axes[1]
    )
    return fig


def plot_clone_intbc_heatmap(
    df: pd.DataFrame, value: str = "detection", max_clones: int = 30, max_intbcs: int = 60, ax=None
):
    """Clone x intBC heatmap of detection (or allele conflict) rates."""
    from matplotlib.colors import LinearSegmentedColormap

    from .integrations import integration_metrics

    ax = _new_ax(ax, figsize=(6, 4))
    m = integration_metrics(df)
    sizes = m.groupby("clone")["clone_cells"].first().sort_values(ascending=False)
    clones = list(sizes.index[:max_clones])
    m = m[m["clone"].isin(clones)]
    order = m.sort_values(["clone", value], ascending=[True, False])
    intbcs = list(dict.fromkeys(order.sort_values("n_cells", ascending=False)["intBC"]))[
        :max_intbcs
    ]
    mat = m.pivot_table(index="clone", columns="intBC", values=value, aggfunc="first").reindex(
        index=clones, columns=intbcs
    )
    # order intBCs by the clone they are most detected in
    if mat.shape[1]:
        owner = mat.fillna(0).to_numpy().argmax(axis=0)
        mat = mat.iloc[:, np.lexsort((-mat.fillna(0).to_numpy().max(axis=0), owner))]
    cmap = LinearSegmentedColormap.from_list("cas_blue", SEQ)
    im = ax.imshow(
        mat.fillna(0).to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_yticks(np.arange(len(mat.index)), [str(i) for i in mat.index], fontsize=7)
    ax.set_xticks([])
    kept = df.attrs.get("integrations")
    title = f"intBC {value} per clone"
    if kept and value == "conflict":
        # outline integrations detected well enough but removed for allele conflict
        det = m.pivot_table(
            index="clone", columns="intBC", values="detection", aggfunc="first"
        ).reindex(index=mat.index, columns=mat.columns)
        min_det = _params(df).get("integration_min_detection") or 0
        for yi, c in enumerate(mat.index):
            for xi, i in enumerate(mat.columns):
                d = det.iloc[yi, xi]
                if not np.isnan(d) and d >= min_det and i not in kept.get(str(c), []):
                    ax.add_patch(
                        _plt().Rectangle(
                            (xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor=FLAGGED, lw=1.5
                        )
                    )
        title += " (outlined: removed for conflict)"
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(f"intBCs (top {mat.shape[1]})", fontsize=9, color=INK)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    return ax


def plot_integrations(df: pd.DataFrame, ax=None):
    """Detection vs. allele-conflict rate per integration, for :func:`cassiopeia.pp.filter_integrations`."""
    ax = _new_ax(ax, figsize=(4.4, 3.6))
    t = pd.DataFrame(df.attrs.get("integration_table", {}))
    if t.empty:
        raise ValueError("Run pp.filter_integrations first.")
    params = _params(df)
    ok = t["kept"].to_numpy().astype(bool)
    _scatter_pass(ax, t["detection"].to_numpy(), t["conflict"].to_numpy(), ok)
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="center left")
    _style(
        ax,
        "Integrations (clone x intBC)" if "clone" in df.columns else "Integrations (intBCs)",
        "Detection (fraction of clone cells)"
        if "clone" in df.columns
        else "Detection (fraction of cells)",
        "Cells with conflicting alleles",
    )
    _vline(ax, params.get("integration_min_detection"), "min detection")
    _hline(ax, params.get("integration_max_conflict"), "max conflict")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax


def plot_doublet_scores(df: pd.DataFrame):
    """Distributions of the doublet scores with their thresholds, for :func:`cassiopeia.pp.filter_doublets`."""
    plt = _plt()
    params = _params(df)
    cols = [c for c in df.columns if c.startswith("doublet_")]
    cells = df.drop_duplicates("cellBC")
    n = max(1, len(cols))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.3), constrained_layout=True, squeeze=False)
    flagged = ~cells["pass_doublet"].to_numpy().astype(bool)
    for ax, col in zip(axes[0], cols, strict=False):
        method = col[len("doublet_") :]
        vals = cells[col].astype(float).to_numpy()
        bins = np.linspace(0, 1, 26) if method in ("allele_conflict", "clone_mixture") else 30
        ok = ~np.isnan(vals)
        ax.hist(
            [vals[ok & ~flagged], vals[ok & flagged]],
            bins=bins,
            stacked=True,
            color=[KEPT, FLAGGED],
            label=["kept", "flagged doublet (any method)"],
            edgecolor="white",
            linewidth=0.5,
        )
        thr = params.get(f"doublet_{method}_threshold")
        _vline(ax, thr, "threshold")
        n_method = int((vals[ok] > thr).sum()) if thr is not None else int(flagged.sum())
        _style(
            ax,
            f"{method.replace('_', ' ')}: {n_method:,} of {int(ok.sum()):,} scored cells above threshold",
            "Score",
            "Cells",
        )
        ax.legend(frameon=False, fontsize=7)
    return fig


# ------------------------------------------------------------ edit states


def _edit_label(token, width: int = 24) -> str:
    """Short label for an allele (the bracketed edits of an indel allele)."""
    s = str(token)
    edits = re.findall(r"\[([^\]]+)\]", s)
    if edits:
        s = ",".join(edits)
    return s if len(s) <= width else s[: width - 1] + "…"


def _is_unedited(token: str) -> bool:
    return token == "None" or "[None]" in token


def _state_counts(data) -> tuple[pd.DataFrame, str, str]:
    """Long table (unit, state, count) and the unit / denominator names.

    ``state`` is an allele, ``"unedited"``, ``"missing"`` or ``"ambiguous"``.
    """
    rows = []
    if isinstance(data, pd.DataFrame) and "cellBC" in data.columns:
        # allele table: one bar per site, counting UMIs (reads without UMIs)
        d = row_passing(data)
        has_umis = data.attrs.get("has_umis", True)
        w = (umi_counts(d) if has_umis else d["n_reads"].to_numpy()).astype(np.float64)
        for site in site_columns(data):
            col = d[site]
            if not isinstance(col.dtype, pd.CategoricalDtype):
                col = col.astype("category")
            cats = col.cat.categories
            counts = np.bincount(col.cat.codes.to_numpy() + 1, weights=w, minlength=len(cats) + 1)
            rows.append((site, "missing", int(counts[0])))
            for tok, n in zip(cats, counts[1:], strict=True):
                if n > 0:
                    tok = str(tok)
                    rows.append((site, "unedited" if _is_unedited(tok) else tok, int(n)))
        unit = "UMIs" if has_umis else "reads"
        return pd.DataFrame(rows, columns=["unit", "state", "count"]), "site", unit
    if hasattr(data, "obsm"):
        chars = data.obsm["characters"]
        meta = data.uns
        clone = data.obs["clone"].astype(str).to_numpy() if "clone" in data.obs.columns else None
    elif isinstance(data, pd.DataFrame):
        chars, meta, clone = data, data.attrs, None
    else:
        raise TypeError(
            "plot_edit_frequency expects a TreeData or character matrix from "
            "pp.alleles_to_characters, or an allele table."
        )
    missing = str(meta.get("missing_state", -1))
    unedited = str(meta.get("unmodified_state", 0))
    states = pd.DataFrame(meta.get("character_states", {}))
    allele_of = {}
    if len(states):
        allele_of = dict(
            zip(
                zip(states["character"].astype(str), states["state"].astype(str), strict=True),
                states["allele"],
                strict=True,
            )
        )
    owners: dict[str, set[str]] = {}
    for c, names in (meta.get("clone_characters") or {}).items():
        for name in names:
            owners.setdefault(name, set()).add(str(c))
    values = chars.to_numpy()
    for j, name in enumerate(chars.columns):
        col = values[:, j]
        if clone is not None and name in owners:
            col = col[np.isin(clone, list(owners[name]))]
        s = pd.Series(col, dtype=object)
        is_tuple = s.map(lambda v: isinstance(v, tuple)).to_numpy(dtype=bool)
        if is_tuple.any():
            rows.append((name, "ambiguous", int(is_tuple.sum())))
        for v, n in s[~is_tuple].astype(str).value_counts().items():
            if v == missing:
                state = "missing"
            elif v == unedited:
                state = "unedited"
            else:
                state = str(allele_of.get((str(name), v), v))
            rows.append((name, state, int(n)))
    denom = "clone cells" if clone is not None and owners else "cells"
    return pd.DataFrame(rows, columns=["unit", "state", "count"]), "character", denom


def plot_edit_frequency(
    data,
    by: Literal["character", "edit"] = "character",
    color_by: Literal["edited", "edit"] = "edited",
    n_edits: int = 8,
    max_characters: int = 80,
    ax=None,
):
    """Frequency of edits per character, or of each edit across characters.

    With ``by="character"``, one stacked bar per character shows the fraction
    of cells that are edited, unedited (light gray) or missing (white), and
    ambiguous if there are ambiguous states. ``color_by="edit"`` splits the
    edited part into the ``n_edits`` most frequent edits (one color each)
    and the remaining edits. With ``by="edit"``, one bar per edit (the
    ``n_edits`` most frequent) shows the fraction of observed (non-missing)
    entries, across all characters, that carry it.

    Edits are identified by their allele (through
    ``uns['character_states']``), so the same edit has the same color in
    every character. With several clones, each character is summarized over
    the cells of the clone(s) it belongs to. An allele table can also be
    passed, giving one bar per site over its UMIs.

    Args:
        data: TreeData or character matrix (DataFrame) from
            :func:`cassiopeia.pp.alleles_to_characters`, or an allele table.
        by: ``"character"`` (one bar per character) or ``"edit"`` (one bar
            per edit).
        color_by: ``"edited"`` (one color for all edits) or ``"edit"``
            (a color per frequent edit; at most 8).
        n_edits: Number of edits to show individually.
        max_characters: Show at most this many characters (those with the
            fewest missing values).
        ax: Matplotlib axes.

    Returns:
        The matplotlib axes.
    """
    if by not in ("character", "edit"):
        raise ValueError(f"by must be 'character' or 'edit', not {by!r}.")
    if color_by not in ("edited", "edit"):
        raise ValueError(f"color_by must be 'edited' or 'edit', not {color_by!r}.")
    long, unit, denom = _state_counts(data)
    special = {"missing", "unedited", "ambiguous"}
    edits = long[~long["state"].isin(special)]
    top = list(edits.groupby("state")["count"].sum().sort_values(ascending=False).index[:n_edits])
    color_of = {
        e: EDIT_COLORS[i] if i < len(EDIT_COLORS) else OTHER_EDITS for i, e in enumerate(top)
    }

    if by == "edit":
        observed = long.loc[long["state"] != "missing", "count"].sum()
        freq = edits.groupby("state")["count"].sum().reindex(top) / max(observed, 1)
        ax = _new_ax(ax, figsize=(max(3.5, 0.32 * len(top) + 1.6), 3.4))
        x = np.arange(len(top))
        colors = [color_of[e] if color_by == "edit" else KEPT for e in top]
        ax.bar(x, freq.to_numpy(), color=colors, width=0.8, zorder=2)
        ax.set_xticks(x, [_edit_label(e) for e in top], rotation=90, fontsize=7)
        _style(
            ax,
            f"Most frequent edits (across {long['unit'].nunique():,} {unit}s)",
            None,
            f"Fraction of observed {unit} entries"
            if unit == "character"
            else f"Fraction of {denom}",
        )
        return ax

    table = long.pivot_table(index="unit", columns="state", values="count", aggfunc="sum")
    table = table.reindex(pd.unique(long["unit"])).fillna(0)
    table = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    if len(table) > max_characters:
        miss = table["missing"] if "missing" in table.columns else pd.Series(0, index=table.index)
        table = table.loc[miss.sort_values(kind="stable").index[:max_characters]]
    edit_cols = [c for c in table.columns if c not in special]
    if color_by == "edit":
        named = [e for e in top if e in table.columns and color_of[e] != OTHER_EDITS]
        others = table[[c for c in edit_cols if c not in named]].sum(axis=1)
        layers = [(_edit_label(e), table[e], color_of[e]) for e in named]
        layers.append(("other edits", others, OTHER_EDITS))
    else:
        layers = [("edited", table[edit_cols].sum(axis=1), KEPT)]
    for name, color in (("ambiguous", AMBIGUOUS), ("unedited", UNEDITED), ("missing", MISSING)):
        if name in table.columns or name != "ambiguous":
            col = table[name] if name in table.columns else pd.Series(0.0, index=table.index)
            layers.append((name, col, color))

    n = len(table)
    ax = _new_ax(ax, figsize=(max(4.5, 0.14 * n + 3.5), 3.6))
    x = np.arange(n)
    from matplotlib.patches import Patch

    bottom = np.zeros(n)
    handles = []
    for name, vals, color in layers:
        v = vals.to_numpy(dtype=float)
        if not v.any() and name not in ("unedited", "missing"):
            continue
        ax.bar(x, v, bottom=bottom, color=color, width=0.85, linewidth=0, zorder=2)
        bottom += v
        edge = OUTLINE if color == MISSING else color
        handles.append(Patch(facecolor=color, edgecolor=edge, linewidth=0.6, label=name))
    # outline each bar so the white (missing) part stays visible
    ax.bar(x, np.ones(n), width=0.85, fill=False, edgecolor=OUTLINE, linewidth=0.6, zorder=3)
    if n <= 50:
        ax.set_xticks(x, [str(i) for i in table.index], rotation=90, fontsize=6 if n > 20 else 7)
    else:
        ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, n - 0.4)
    title = (
        f"Alleles per site ({int(long['count'].sum() / max(long['unit'].nunique(), 1)):,} {denom})"
        if unit == "site"
        else f"Character states ({n:,} characters)"
    )
    _style(ax, title, None, f"Fraction of {denom}")
    ax.grid(False, axis="x")
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        handlelength=1.0,
        borderaxespad=0,
    )
    return ax
