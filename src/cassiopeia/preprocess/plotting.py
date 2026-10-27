"""QC plots for the preprocessing steps.

Every QC step calls one of these when ``plot=True``; they can also be called
directly on a molecule table (or TreeData) to redraw a figure later.
Thresholds are read from ``df.attrs['qc_params']`` (or passed explicitly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import molecule_passing
from .qc import cell_metrics, molecule_metrics

# Colors: categorical slot 1 for kept data, a recessive neutral for filtered
# data, slot 2 for flagged cells, neutral ink for thresholds and text.
KEPT = "#2a78d6"
FILTERED = "#bdbcb6"
FLAGGED = "#eb6834"
THRESH = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8984"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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


def _params(data) -> dict:
    if hasattr(data, "uns"):
        return dict(data.uns.get("preprocess", {}).get("qc_params", {}))
    return dict(getattr(data, "attrs", {}).get("qc_params", {}))


def _new_ax(ax, figsize=(4, 3.4)):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return ax


def _scatter_pass(ax, x, y, ok, labels=("kept", "filtered")):
    n = len(x)
    raster = n > 50000
    size = 6 if n > 5000 else 12
    ok = np.asarray(ok, dtype=bool)
    ax.scatter(
        x[~ok],
        y[~ok],
        s=size,
        color=FILTERED,
        linewidths=0,
        alpha=0.7,
        rasterized=raster,
        label=f"{labels[1]} (n={int((~ok).sum()):,})",
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
        label=f"{labels[0]} (n={int(ok.sum()):,})",
        zorder=2,
    )
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="lower right")


# ------------------------------------------------------------- molecules


def plot_umi_reads(
    df: pd.DataFrame,
    level: str = "intbc",
    hue: str = "pass",
    thresholds: dict | bool | None = None,
    ax=None,
):
    """UMIs vs. mean reads per UMI, per (cell, intBC) or per cell.

    Points are colored by whether they pass filtering; dashed lines mark the
    thresholds used (``min_umi_per_intbc`` / ``min_reads`` per intBC, or
    ``min_umi`` per cell).

    Args:
        df: Molecule table.
        level: ``"intbc"`` (per cell and intBC) or ``"cell"``.
        hue: Boolean column (``pass`` or a ``pass_*`` column) used for coloring.
        thresholds: Dict overriding the recorded thresholds; ``False`` hides them.
        ax: Matplotlib axes.
    """
    ax = _new_ax(ax)
    params = _params(df) if thresholds is None or thresholds is True else (thresholds or {})
    if level == "intbc":
        d = df if hue == "pass" else df.assign(**{"pass": df[hue]}) if hue in df.columns else df
        m = molecule_metrics(d)
        x, y, ok = m["n_umis"].to_numpy(), m["reads_per_umi"].to_numpy(), m["pass"].to_numpy()
        _scatter_pass(ax, x, y, ok)
        _style(ax, "Molecules per (cell, intBC)", "UMIs", "Mean reads per UMI")
        if thresholds is not False:
            _vline(ax, params.get("min_umi_per_intbc"), "min UMIs")
            _hline(ax, params.get("min_reads"), "min reads")
    else:
        c = cell_metrics(df)
        if hue == "pass":
            ok = c["passed_qc"].to_numpy() & (c["n_umis"] > 0).to_numpy()
        elif hue in c.columns:
            ok = c[hue].fillna(False).to_numpy().astype(bool)
        else:
            ok = np.ones(len(c), dtype=bool)
        x = c["n_umis"].clip(lower=0.8).to_numpy()
        y = c["reads_per_umi"].to_numpy()
        _scatter_pass(ax, x, y, ok, labels=("kept cells", "filtered cells"))
        _style(ax, "Cells", "UMIs (passing molecules)", "Mean reads per UMI")
        if thresholds is not False:
            _vline(ax, params.get("min_umi"), "min UMIs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax)
    return ax


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
    ax = axes[1]
    h = np.asarray(res.get("reads_per_umi_hist", []))
    if h.size:
        k = np.arange(len(h))
        last = int(np.flatnonzero(h)[-1]) + 1 if h.any() else 2
        ax.bar(k[1:last], h[1:last], color=KEPT, width=1.0, linewidth=0)
        ax.set_yscale("log")
    _style(ax, "Reads per UMI", "Reads", "Molecules")
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


# ----------------------------------------------------------------- cells


def plot_umi_intbcs(
    df: pd.DataFrame,
    hue: str = "pass",
    thresholds: dict | bool | None = None,
    ax=None,
):
    """UMIs vs. unique intBCs detected, per cell.

    Counts are over molecules passing the molecule-level filters, so filtered
    cells are drawn where they were before cell filtering. Points are colored
    by whether the cell passes filtering and jittered vertically so integer
    intBC counts don't overlap; a dashed line marks ``min_umi``.

    Args:
        df: Molecule table.
        hue: ``pass`` (all cell filters) or a cell-level ``pass_*`` column.
        thresholds: Dict overriding the recorded thresholds; ``False`` hides them.
        ax: Matplotlib axes.
    """
    from matplotlib.ticker import MaxNLocator

    ax = _new_ax(ax)
    params = _params(df) if thresholds is None or thresholds is True else (thresholds or {})
    c = cell_metrics(df)
    d = molecule_passing(df)
    g = d.groupby(d["cellBC"].astype(str))
    n_umis = g.size().reindex(c.index, fill_value=0).to_numpy()
    n_intbcs = g["intBC"].nunique().reindex(c.index, fill_value=0).to_numpy()
    if hue == "pass":
        ok = c["passed_qc"].to_numpy() & (n_umis > 0)
    elif hue in c.columns:
        ok = c[hue].fillna(False).to_numpy().astype(bool)
    else:
        ok = np.ones(len(c), dtype=bool)
    x = np.clip(n_umis, 0.8, None)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(c))
    y = n_intbcs + jitter
    _scatter_pass(ax, x, y, ok, labels=("kept cells", "filtered cells"))
    _style(ax, "Cells", "UMIs", "intBCs detected")
    if thresholds is not False:
        _vline(ax, params.get("min_umi"), "min UMIs")
    ax.set_xscale("log")
    _plain_log(ax)
    ax.set_ylim(bottom=-0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
        plot_intbc_detection(df, by="cell", ax=axes[0, 1])
    return fig


def plot_intbc_detection(df: pd.DataFrame, by: str = "cell", ax=None):
    """Detection of intBCs: per cell (fraction of its clone's integrations) or per integration."""
    ax = _new_ax(ax)
    params = _params(df)
    if by == "cell":
        from .filter import intbc_detection

        frac = intbc_detection(df).dropna()
        ax.hist(frac, bins=np.linspace(0, 1, 26), color=KEPT, edgecolor="white", linewidth=1)
        _style(ax, "intBC detection per cell", "Fraction of clone integrations detected", "Cells")
        _vline(ax, params.get("min_intbc_frac"), "min fraction")
    else:
        t = pd.DataFrame(df.attrs.get("integration_table", {}))
        if t.empty:
            from .integrations import integration_metrics

            t = integration_metrics(df)
            t["kept"] = True
        t = t.sort_values("detection", ascending=False)
        x = np.arange(len(t))
        ax.bar(x, t["detection"], color=np.where(t["kept"], KEPT, FILTERED), width=0.8)
        _style(
            ax,
            "Detection per integration",
            "Integration (clone x intBC)",
            "Fraction of clone cells",
        )
        _hline(ax, params.get("integration_min_detection"), "min detection")
        ax.set_xticks([])
    return ax


def plot_clones(df: pd.DataFrame, max_clones: int = 30, max_intbcs: int = 60):
    """Clone sizes and intBC detection per clone."""
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
    if kept and value == "conflict":
        for yi, c in enumerate(mat.index):
            for xi, i in enumerate(mat.columns):
                if not np.isnan(mat.iloc[yi, xi]) and i not in kept.get(str(c), []):
                    ax.add_patch(
                        _plt().Rectangle(
                            (xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor=FLAGGED, lw=1
                        )
                    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"intBC {value} per clone", fontsize=10, color=INK, loc="left")
    ax.set_xlabel(f"intBCs (top {mat.shape[1]})", fontsize=9, color=INK)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    return ax


def plot_integrations(df: pd.DataFrame, ax=None):
    """Detection vs. allele-conflict rate per integration (clone x intBC)."""
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
        "Integrations (clone x intBC)",
        "Detection (fraction of clone cells)",
        "Cells with conflicting alleles",
    )
    _vline(ax, params.get("integration_min_detection"), "min detection")
    _hline(ax, params.get("integration_max_conflict"), "max conflict")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax


def plot_doublet_scores(df: pd.DataFrame):
    """Distributions of the doublet scores with their thresholds."""
    plt = _plt()
    params = _params(df)
    cols = [c for c in df.columns if c.startswith("doublet_") and c != "doublet"]
    cells = df.drop_duplicates("cellBC")
    n = max(1, len(cols))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.3), constrained_layout=True, squeeze=False)
    for ax, col in zip(axes[0], cols, strict=False):
        method = col[len("doublet_") :]
        v = cells[col].astype(float)
        flagged = cells["doublet"].to_numpy().astype(bool)
        bins = np.linspace(0, 1, 26) if method in ("allele_conflict", "clone_mixture") else 30
        vals = v.to_numpy()
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


# ------------------------------------------------------------ characters


def plot_edit_frequency(data, ax=None, max_characters: int = 80):
    """Fraction of cells edited, unedited and missing per character."""
    if hasattr(data, "obsm"):
        chars = data.obsm["characters"]
    else:
        raise ValueError("plot_edit_frequency expects a TreeData from pp.alleles_to_characters.")
    a = np.asarray(chars).astype(str)
    missing = np.asarray(data.uns.get("missing_state", -1), dtype=object).astype(str).ravel()
    unedited = np.asarray(data.uns.get("unmodified_state", 0), dtype=object).astype(str).ravel()
    # fraction among cells of the clone(s) that carry each character
    owners = {}
    for clone, names in (data.uns.get("clone_characters") or {}).items():
        for n in names:
            owners.setdefault(n, set()).add(str(clone))
    clone = data.obs["clone"].astype(str).to_numpy() if "clone" in data.obs.columns else None
    rows = []
    for j, name in enumerate(chars.columns):
        col = a[:, j]
        if clone is not None and name in owners:
            col = col[np.isin(clone, list(owners[name]))]
        n = max(len(col), 1)
        miss, uned = np.isin(col, missing), np.isin(col, unedited)
        rows.append(((~miss & ~uned).sum() / n, uned.sum() / n, miss.sum() / n))
    t = pd.DataFrame(rows, columns=["edited", "unedited", "missing"], index=chars.columns)
    if len(t) > max_characters:
        t = t.sort_values("missing").head(max_characters)
    ax = _new_ax(ax, figsize=(max(4, 0.12 * len(t) + 1.5), 3.4))
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for col, color in (("edited", KEPT), ("unedited", "#1baf7a"), ("missing", FILTERED)):
        ax.bar(x, t[col], bottom=bottom, color=color, width=0.85, label=col, linewidth=0)
        bottom += t[col].to_numpy()
    if len(t) <= 40:
        ax.set_xticks(x, t.index, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    ax.set_ylim(0, 1)
    _style(
        ax,
        f"Character states ({chars.shape[0]:,} cells x {chars.shape[1]:,} characters)",
        None,
        "Fraction of clone cells",
    )
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12 if len(t) > 40 else -0.35),
    )
    return ax
