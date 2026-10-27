"""Tests for the built-in QC plots."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import cassiopeia as cas


@pytest.fixture(scope="module")
def processed(sim_factory):
    sim = sim_factory("weissman", "single")
    df = cas.pp.call_alleles(sim.bam, "weissman")
    df = cas.pp.filter_molecules(df, min_reads=2, min_umi_per_intbc=2)
    df = cas.pp.filter_cells(df, min_umi=5)
    df = cas.pp.call_clones(df, method="single")
    df = cas.pp.filter_integrations(df)
    df = cas.pp.filter_cells(df, min_intbc_frac=0.5)
    df = cas.pp.filter_doublets(df)
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


def test_umi_reads_plot_thresholds(processed):
    _, df = processed
    ax = cas.pp.plot_umi_reads(df, level="intbc")
    assert 2 in _vlines(ax) and 2 in _hlines(ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("kept") for t in labels) and any(
        t.startswith("filtered") for t in labels
    )
    ax = cas.pp.plot_umi_reads(df, level="cell")
    assert 5 in _vlines(ax)
    ax = cas.pp.plot_umi_reads(df, level="intbc", thresholds=False)
    assert _vlines(ax) == []
    plt.close("all")


def test_all_plots_render(processed, tmp_path, monkeypatch):
    _, df = processed
    monkeypatch.chdir(tmp_path)
    assert cas.pp.plot_intbc_detection(df, by="cell") is not None
    assert cas.pp.plot_intbc_detection(df, by="intbc") is not None
    assert cas.pp.plot_integrations(df) is not None
    assert cas.pp.plot_clone_intbc_heatmap(df, value="conflict") is not None
    assert cas.pp.plot_doublet_scores(df) is not None
    tdata = cas.pp.alleles_to_characters(df)
    assert cas.pp.plot_edit_frequency(tdata) is not None
    assert list(tmp_path.iterdir()) == []  # nothing written
    plt.close("all")


def test_steps_plot_and_save(sim_factory, tmp_path):
    sim = sim_factory("weissman", "clean")
    df = cas.pp.call_alleles(sim.bam, "weissman", plot=True, show=False)
    df = cas.pp.filter_molecules(df, save=tmp_path / "mol.png", show=False)
    assert (tmp_path / "mol.png").exists()
    res = cas.pp.align(
        sim.fastqs, tmp_path / "a.bam", "weissman", save=tmp_path / "align.png", show=False
    )
    assert (tmp_path / "align.png").exists() and res["stats"]["reads"] > 0
    plt.close("all")


def test_plots_from_parquet(processed, tmp_path):
    _, df = processed
    df.to_parquet(tmp_path / "m.parquet")
    df2 = cas.pp.read_molecules(tmp_path / "m.parquet")
    assert df2.attrs["qc_params"]["min_reads"] == 2
    ax = cas.pp.plot_umi_reads(df2)
    assert 2 in _hlines(ax)
    pd.testing.assert_series_equal(df2["pass"], df["pass"], check_names=False)
    plt.close("all")
