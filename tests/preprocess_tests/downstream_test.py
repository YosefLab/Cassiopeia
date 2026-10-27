"""Tests for the allele-table steps: correction, filters, clones, doublets, characters."""

import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas

from .conftest import requires_nova


def _process(sim, clones=False, collapse=False, correct=True, **kw):
    df = cas.pp.call_alleles(sim.bam, sim.construct, n_threads=2, collapse_umis=collapse)
    if correct:
        df = cas.pp.correct_intbcs(df)
    df = cas.pp.filter_alleles(df, min_reads_per_umi=1, min_umi_per_intbc=2)
    df = cas.pp.filter_cells(df, min_umi=5)
    if clones:
        df = cas.pp.call_clones(df, **kw)
    df = cas.pp.filter_integrations(df, min_detection=0.4)
    df = cas.pp.filter_cells(df, min_intbc_frac=0.5)
    return cas.pp.filter_doublets(df)


def _truth_cells(sim, df):
    cells = cas.pp.qc.cell_metrics(df)
    truth = sim.cells.set_index("cellBC_10x")
    truth = truth.rename(columns={c: f"{c}_true" for c in truth.columns if c in cells.columns})
    return cells.join(truth, how="inner")


def test_correct_intbcs(sim_factory):
    sim = sim_factory("weissman", "multi")
    raw = cas.pp.call_alleles(sim.bam, "weissman")
    df = cas.pp.correct_intbcs(raw)
    true = set(sim.integrations["intBC"])
    assert set(df["intBC"].astype(str)) <= true | set(raw["intBC"].astype(str))
    assert df["intBC"].nunique() <= len(true) + 2
    assert not df.duplicated(["cellBC", "intBC", "UMI"]).any()
    assert df["n_reads"].sum() == raw["n_reads"].sum()
    # whitelist mode: unknown intBCs are flagged, not dropped
    wl = sorted(true)[:-1]
    df2 = cas.pp.correct_intbcs(raw, whitelist=wl)
    assert "pass_intbc" in df2.columns
    assert set(df2.loc[df2["pass_intbc"], "intBC"].astype(str)) <= set(wl)
    # collapsed tables are corrected the same way, but UMIs that only differed
    # by an intBC error can no longer be merged
    col = cas.pp.correct_intbcs(cas.pp.collapse_umis(raw))
    assert col["n_umis"].sum() == len(raw) and col["n_reads"].sum() == df["n_reads"].sum()
    assert set(col["intBC"].astype(str)) == set(df["intBC"].astype(str))


def test_collapse_umis(sim_factory):
    sim = sim_factory("weissman", "multi")
    raw = cas.pp.call_alleles(sim.bam, "weissman", plot=False)
    col = cas.pp.collapse_umis(raw, read_cap=3)
    sites = raw.attrs["sites"]
    assert "UMI" not in col.columns and len(col) < len(raw)
    assert col["n_umis"].sum() == len(raw) and col["n_reads"].sum() == raw["n_reads"].sum()
    assert col.attrs["sites"] == sites and col.attrs["qc_params"]["read_cap"] == 3
    # UMIs are kept apart by read count below the cap
    per_umi = col["n_reads"] / col["n_umis"]
    low = per_umi < 3
    assert (per_umi[low] == per_umi[low].round()).all()
    cls = np.minimum(per_umi, 3).astype(int).rename("cls")
    assert not pd.concat([col[["cellBC", "intBC", *sites]], cls], axis=1).duplicated().any()
    # collapsing while reading the BAM gives the same table
    native = cas.pp.call_alleles(sim.bam, "weissman", collapse_umis=True, read_cap=3, plot=False)
    key = ["cellBC", "intBC", *sites, "cls"]

    def keyed(x):
        x = x.assign(cls=np.minimum(x["n_reads"] // x["n_umis"], 3))
        return x.astype(dict.fromkeys(key, str)).set_index(key).sort_index()

    a, b = keyed(col), keyed(native)
    pd.testing.assert_frame_equal(a[["n_umis", "n_reads"]], b[["n_umis", "n_reads"]])
    np.testing.assert_allclose(a["allele_frac"], b["allele_frac"], rtol=1e-5)
    # the allele filters give the same result on both forms
    for min_reads in (1, 2, 3):
        f_umi = cas.pp.filter_alleles(raw, min_reads_per_umi=min_reads, plot=False)
        f_col = cas.pp.filter_alleles(native, min_reads_per_umi=min_reads, plot=False)
        assert f_col.loc[f_col["pass"], "n_umis"].sum() == f_umi["pass"].sum()


def test_filters_annotate(sim_factory):
    sim = sim_factory("weissman", "multi")
    df = cas.pp.call_alleles(sim.bam, "weissman")
    out = cas.pp.filter_alleles(df, min_reads_per_umi=3, min_umi_per_intbc=2)
    assert len(out) == len(df)
    assert {"pass_reads_per_umi", "pass_allele_frac", "pass_umis_per_intbc", "pass"} <= set(
        out.columns
    )
    assert (out.loc[out["pass"], "n_reads"] >= 3).all()
    assert out.attrs["qc_params"]["min_reads_per_umi"] == 3
    cells = cas.pp.filter_cells(out, min_umi=20)
    assert cells.attrs["qc_params"]["min_umi"] == 20
    # defaults: 3 reads per UMI and 2 UMIs per (cell, intBC)
    default = cas.pp.filter_alleles(df)
    assert default.attrs["qc_params"]["min_reads_per_umi"] == 3
    assert default.attrs["qc_params"]["min_umi_per_intbc"] == 2
    # on a collapsed table the UMI filter counts UMIs, not rows
    col = cas.pp.filter_alleles(cas.pp.collapse_umis(df), min_reads_per_umi=1)
    ok = col[col["pass"]]
    per_intbc = ok.groupby(["cellBC", "intBC"], observed=True)["n_umis"].sum()
    assert (per_intbc >= 2).all()
    with pytest.raises(Exception, match="filter_integrations"):
        cas.pp.filter_cells(out, min_intbc_frac=0.5)


def test_single_clone_pipeline(sim_factory):
    sim = sim_factory("weissman", "single")
    df = _process(sim)
    assert "clone" not in df.columns
    kept = df.attrs["integrations"]["0"]
    kinds = sim.integrations.set_index("intBC")["kind"]
    assert not any(kinds.get(i) in ("duplicated", "silenced") for i in kept)
    assert len(kept) == (kinds == "normal").sum()
    cells = _truth_cells(sim, df)
    assert "doublet" not in cas.pp.qc.cell_metrics(df).columns
    flagged, true = ~cells["pass_doublet"].astype(bool), cells["doublet"].astype(bool)
    assert (flagged & true).sum() >= 0.6 * true.sum()
    assert (flagged & ~true).sum() <= 2
    # intBC fraction is computed against the kept integrations, for cells passing min_umi
    assert cells["intbc_frac"].notna().to_numpy().tolist() == cells["pass_min_umi"].tolist()
    assert cells["intbc_frac"].dropna().between(0, 1).all()
    assert (cells.loc[cells["pass_intbc_frac"], "intbc_frac"] >= 0.5).all()
    metrics = cas.pp.clone_metrics(df)
    assert list(metrics.index) == ["0"]
    assert metrics.loc["0", "n_integrations"] == len(kept)
    assert metrics.loc["0", "intBCs"] == sorted(kept)
    all_cells = cas.pp.qc.cell_metrics(df)
    assert metrics.loc["0", "n_cells"] == all_cells["pass_min_umi"].sum()


def test_collapsed_pipeline_matches(sim_factory):
    sim = sim_factory("weissman", "single")
    # without intBC correction, which can merge UMIs only before collapsing
    a = cas.pp.qc.cell_metrics(_process(sim, correct=False))
    b = cas.pp.qc.cell_metrics(_process(sim, collapse=True, correct=False))
    cols = ["n_umis", "n_reads", "reads_per_umi", "n_intbcs", "intbc_frac", "pass_doublet"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


@requires_nova
def test_multi_clone_pipeline(sim_factory):
    from sklearn.metrics import adjusted_rand_score

    sim = sim_factory("weissman", "multi", cells_per_clone=40)
    df = _process(
        sim, clones=True, min_intbc_umi_count=2, min_intbc_cellbc_count=5, min_cells_assigned=10
    )
    cells = _truth_cells(sim, df)
    assert not cells.loc[cells["clone"] == "multiplet", "pass_clone"].astype(bool).any()
    single = cells[~cells["doublet"] & cells["clone"].notna() & (cells["clone"] != "multiplet")]
    assert adjusted_rand_score(single["clone_true"], single["clone"].astype(str)) > 0.95
    flagged, true = ~cells["pass_doublet"].astype(bool), cells["doublet"].astype(bool)
    assert (flagged & cells["heterotypic"]).sum() >= 0.9 * cells["heterotypic"].sum()
    assert (flagged & ~true).sum() <= 2


def test_user_clones(sim_factory):
    sim = sim_factory("weissman", "multi")
    clones = dict(zip(sim.cells["cellBC_10x"], sim.cells["clone"], strict=True))
    df = cas.pp.call_alleles(sim.bam, sim.construct, n_threads=2)
    df = cas.pp.filter_alleles(df, min_reads_per_umi=1)
    df = cas.pp.filter_cells(df, min_umi=5)
    df = cas.pp.call_clones(df, clones=clones)
    df = cas.pp.filter_integrations(df)
    df = cas.pp.filter_doublets(df)
    assert df.attrs["qc_params"]["clone_method"] == "user"
    assert len(df.attrs["clone_intbcs"]) >= 3
    assert "doublet_clone_mixture" in df.columns
    metrics = cas.pp.clone_metrics(df)
    assert {"n_cells", "n_cells_passed", "n_doublets", "n_integrations", "intBCs"} <= set(
        metrics.columns
    )
    kept = df.attrs["integrations"]
    for clone, row in metrics.iterrows():
        assert row["intBCs"] == sorted(kept.get(clone, []))
    with pytest.raises(Exception, match="Unknown clone calling method"):
        cas.pp.call_clones(df, method="single")


def test_alleles_to_characters(sim_factory, tmp_path):
    sim = sim_factory("petracer", "single")
    df = _process(sim)
    tdata = cas.pp.alleles_to_characters(df)
    chars = tdata.obsm["characters"]
    n_int = len(df.attrs["integrations"]["0"])
    assert chars.shape[1] == 3 * n_int
    assert set(np.unique(chars.to_numpy())) <= {"-", "*"} | {str(i) for i in range(1, 9)}
    assert (tdata.uns["missing_state"], tdata.uns["unmodified_state"]) == ("-", "*")
    assert tdata.obs["pass_doublet"].astype(bool).all()
    assert "clone" not in tdata.obs.columns and "priors" not in tdata.uns
    states = tdata.uns["character_states"]
    assert set(states.loc[states["state"] != "*", "allele"]) <= {str(i) for i in range(1, 9)}
    # the other mappings encode the same states
    ints = cas.pp.alleles_to_characters(df, mapping="int")
    alleles = cas.pp.alleles_to_characters(df, mapping="allele")
    s, i, a = (t.obsm["characters"].to_numpy() for t in (tdata, ints, alleles))
    assert ((s == "-") == (i == -1)).all() and ((a == "*") == (i == 0)).all()
    assert (s[i > 0] == i[i > 0].astype(str)).all()
    allele_of = dict(
        zip(zip(states["character"], states["state"], strict=False), states["allele"], strict=True)
    )
    names = chars.columns
    assert all(
        a[r, c] == allele_of[(names[c], s[r, c])] for r, c in zip(*np.nonzero(i > 0), strict=True)
    )
    # a dict maps alleles to the same states in every character
    lookup = {str(k): 10 + k for k in range(1, 9)}
    d = cas.pp.alleles_to_characters(df, mapping=lookup, return_type="dataframe")
    assert isinstance(d, pd.DataFrame) and d.attrs["missing_state"] == -1
    assert ((d.to_numpy() == -1) == (i == -1)).all()
    edited = i > 0
    assert (d.to_numpy()[edited] == np.vectorize(lambda x: lookup[x])(a[edited])).all()
    # h5td round trip keeps characters and QC
    tdata.write_h5td(tmp_path / "t.h5td")
    t2 = td.read_h5td(tmp_path / "t.h5td")
    pd.testing.assert_frame_equal(t2.obsm["characters"], chars, check_dtype=False)
    assert "cell_qc" in t2.uns["preprocess"]
    # a solver runs directly on the output
    cas.solver.nj(t2)
    assert "nj" in t2.obst


def test_ambiguous_characters(sim_factory):
    sim = sim_factory("weissman", "single")
    df = cas.pp.call_alleles(sim.bam, "weissman", plot=False)
    df = cas.pp.filter_alleles(df, min_reads_per_umi=1, plot=False)
    df = cas.pp.filter_integrations(df, max_conflict=1.0, plot=False)
    plain = cas.pp.alleles_to_characters(df, mapping="allele", min_allele_frac=0.2)
    amb = cas.pp.alleles_to_characters(df, mapping="allele", ambiguous=True, min_allele_frac=0.2)
    p, a = plain.obsm["characters"], amb.obsm["characters"]
    tuples = a.map(lambda v: isinstance(v, tuple)).to_numpy()
    assert tuples.any()
    # ambiguous entries hold distinct states, including the major allele
    for r, c in zip(*np.nonzero(tuples), strict=True):
        v = a.iat[r, c]
        assert len(v) >= 2 and len(set(v)) == len(v) and "-" not in v
        assert p.iat[r, c] in (*v, "-")
    # everything else is unchanged where the major allele was called
    same = ~tuples & (p.to_numpy() != "-")
    assert (p.to_numpy()[same] == a.to_numpy()[same]).all()
    ints = cas.pp.alleles_to_characters(df, mapping="int", ambiguous=True, min_allele_frac=0.2)
    v = ints.obsm["characters"].to_numpy()[tuples][0]
    assert isinstance(v, tuple) and all(isinstance(x, (int, np.integer)) for x in v)
    strs = cas.pp.alleles_to_characters(df, ambiguous=True, min_allele_frac=0.2).obsm["characters"]
    v = strs.to_numpy()[tuples][0]
    assert isinstance(v, tuple) and all(isinstance(x, str) for x in v)


def test_compute_empirical_indel_priors(sim_factory):
    sim = sim_factory("weissman", "single")
    df = _process(sim)
    priors = cas.pp.compute_empirical_indel_priors(df)
    assert len(priors) > 0 and priors["freq"].between(0, 1).all()
    assert not any("None" in i for i in priors.index)
    assert priors["count"].max() <= df.loc[df["pass"], "intBC"].nunique()
