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
    df = cas.pp.filter_alleles(df, min_reads_per_umi=2, min_umi_per_intbc=2, plot=False)
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


def test_step_plot_thresholds(processed):
    _, df = processed
    ax = _plotting.plot_reads_per_umi(df)
    assert _vlines(ax) == [1.5]  # between 1 and 2 reads
    ax = _plotting.plot_umi_reads(df)
    assert _vlines(ax) == [1.5]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("kept") for t in labels) and any(
        t.startswith("filtered") for t in labels
    )
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
    fate = fig.axes[0]
    assert [t.get_text() for t in fate.get_yticklabels()][0] == "used"
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
    n_files = len(list(tmp_path.iterdir()))
    cas.pp.filter_cells(df, min_umi=5)  # plot=True shows, writes nothing
    assert len(list(tmp_path.iterdir())) == n_files
    assert plt.get_fignums() == []  # figures are closed


def test_plots_from_parquet(processed, tmp_path):
    _, df = processed
    df.to_parquet(tmp_path / "m.parquet")
    df2 = cas.pp.read_alleles(tmp_path / "m.parquet")
    assert df2.attrs["qc_params"]["min_reads_per_umi"] == 2
    ax = _plotting.plot_reads_per_umi(df2)
    assert _vlines(ax) == [1.5]
    pd.testing.assert_series_equal(df2["pass"], df["pass"], check_names=False)
    plt.close("all")
