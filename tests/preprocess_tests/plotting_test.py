"""Tests for the built-in QC plots and pp.plot_edit_frequency."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import cassiopeia as cas
from cassiopeia.preprocess import _plotting


@pytest.fixture(scope="module")
def processed(sim_factory):
    sim = sim_factory("weissman", "single")
    df = cas.pp.call_alleles(sim.bam, "weissman", plot=False)
    df = cas.pp.filter_alleles(df, min_umi_per_allele=2, min_avg_reads_per_umi=3, plot=False)
    df = cas.pp.filter_cells(df, min_umi=5, plot=False)
    df = cas.pp.filter_integrations(df, plot=False)
    df = cas.pp.filter_cells(df, min_intbc_frac=0.5, plot=False)
    df = cas.pp.filter_doublets(df, plot=False)
    return sim, df


def _vlines(ax):
    return [
        ln.get_xdata()[0]
        for ln in ax.get_lines()
        if ln.get_linestyle() == "--" and len(set(ln.get_xdata())) == 1
    ]


def _hlines(ax):
    return [
        ln.get_ydata()[0]
        for ln in ax.get_lines()
        if ln.get_linestyle() == "--" and len(set(ln.get_ydata())) == 1
    ]


def test_step_plot_thresholds(processed):
    _, df = processed
    hist = df.attrs["read_stats"]["reads_per_umi"]
    ax = _plotting.plot_reads_per_umi(hist, 2)
    assert _vlines(ax) == [2]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    filtered = sum(u for r, u in zip(hist["reads"], hist["umis"], strict=True) if r < 2)
    assert f"filtered (n={filtered:,})" in labels and len(labels) == 2
    fig = _plotting.plot_filter_alleles(df)
    ax, cbar = fig.axes
    assert _vlines(ax) == [1.5] and _hlines(ax) == [3] and ax.get_legend() is None
    assert _hlines(cbar) == [0.5] and cbar.get_ylabel() == "Allele fraction"
    flags = ["pass_umis_per_allele", "pass_avg_reads_per_umi", "pass_allele_frac"]
    kept = df[df[[*flags, "pass_allele_conflict"]].all(axis=1)]
    n_kept = len(kept.drop_duplicates(["cellBC", "intBC", *df.attrs["sites"]]))
    assert ax.get_title(loc="left").startswith(f"Alleles: {n_kept:,} of")
    ax = _plotting.plot_umi_intbcs(df)
    assert 5 in _vlines(ax)
    assert ax.get_ylabel() == "intBCs detected"
    ax = _plotting.plot_intbc_fraction(df)
    assert 0.5 in _vlines(ax)
    plt.close("all")


def test_step_figures_render(processed):
    _, df = processed
    assert _plotting.plot_filter_cells(df) is not None
    assert _plotting.plot_integrations(df) is not None
    assert _plotting.plot_clone_intbc_heatmap(df, value="conflict") is not None
    assert _plotting.plot_doublet_scores(df) is not None
    fig = _plotting.plot_call_alleles(df)
    assert fig.axes[0].get_title(loc="left") == "Reads per UMI"
    fate = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert fate[:2] == ["used", "UMI with < 2 reads"]
    plt.close("all")


def test_plot_edit_frequency(processed):
    _, df = processed
    tdata = cas.pp.alleles_to_characters(df)
    ax = cas.pp.plot_edit_frequency(tdata)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["edited", "unedited", "missing"]
    legend = ax.get_legend()
    colors = {
        t.get_text(): p.get_facecolor()
        for t, p in zip(legend.get_texts(), legend.get_patches(), strict=True)
    }
    assert colors["missing"][:3] == (1.0, 1.0, 1.0)  # missing is white
    ax = cas.pp.plot_edit_frequency(tdata, color_by="edit", n_edits=3)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3 + 3 and labels[-3:] == ["other edits", "unedited", "missing"]
    # edits are identified by allele, whatever the mapping
    ints = cas.pp.alleles_to_characters(df, mapping="int")
    ax2 = cas.pp.plot_edit_frequency(ints, color_by="edit", n_edits=3)
    assert [t.get_text() for t in ax2.get_legend().get_texts()] == labels
    ax = cas.pp.plot_edit_frequency(tdata, by="edit", n_edits=5)
    assert len(ax.patches) == 5
    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True) and 0 < sum(heights) <= 1
    chars = cas.pp.alleles_to_characters(df, return_type="dataframe")
    assert cas.pp.plot_edit_frequency(chars, by="edit") is not None
    ax = cas.pp.plot_edit_frequency(df)  # allele table: one bar per site
    assert [t.get_text() for t in ax.get_xticklabels()] == df.attrs["sites"]
    assert ax.get_ylabel() == "Fraction of UMIs"
    # without UMIs (bulk), sites are summarized by reads
    sim = processed[0]
    bulk = cas.pp.call_alleles(sim.bam, "weissman", umi_tag=None, plot=False)
    ax = cas.pp.plot_edit_frequency(bulk)
    assert ax.get_ylabel() == "Fraction of reads"
    assert f"{bulk['n_reads'].sum():,} reads" in ax.get_title(loc="left")
    plt.close("all")


def test_steps_plot_and_save(sim_factory, tmp_path):
    plt.close("all")
    sim = sim_factory("weissman", "clean")
    df = cas.pp.call_alleles(sim.bam, "weissman", save=tmp_path / "alleles.png", plot=False)
    assert (tmp_path / "alleles.png").exists()
    df = cas.pp.filter_alleles(df, save=tmp_path / "alleles_filter.png", plot=False)
    assert (tmp_path / "alleles_filter.png").exists()
    res = cas.pp.align(
        sim.fastqs, tmp_path / "a.bam", "weissman", save=tmp_path / "align.png", plot=False
    )
    assert (tmp_path / "align.png").exists() and res["stats"]["reads"] > 0
    ax = _plotting.plot_align(res).axes[1]
    assert ax.get_xscale() == "log" and ax.get_title(loc="left") == "Reads per UMI"
    assert ax.get_legend().get_texts()[0].get_text().startswith("UMIs (n=")
    plt.close(ax.figure)
    n_files = len(list(tmp_path.iterdir()))
    cas.pp.filter_cells(df, min_umi=5)  # plot=True shows, writes nothing
    assert len(list(tmp_path.iterdir())) == n_files
    assert plt.get_fignums() == []  # figures are closed


def test_plots_from_parquet(processed, tmp_path):
    _, df = processed
    df.to_parquet(tmp_path / "m.parquet")
    df2 = cas.pp.read_alleles(tmp_path / "m.parquet")
    assert df2.attrs["qc_params"]["min_reads_per_umi"] == 2
    assert _vlines(_plotting.plot_call_alleles(df2).axes[0]) == [2]
    assert _hlines(_plotting.plot_filter_alleles(df2).axes[1]) == [0.5]
    pd.testing.assert_series_equal(df2["pass"], df["pass"], check_names=False)
    plt.close("all")
