"""Tests for the molecule-table steps: correction, filters, clones, doublets, characters."""

import numpy as np
import pandas as pd
import treedata as td

import cassiopeia as cas

from .conftest import requires_nova


def _process(sim, method, **kw):
    df = cas.pp.call_alleles(sim.bam, sim.construct, n_threads=2)
    df = cas.pp.correct_intbcs(df)
    df = cas.pp.filter_molecules(df, min_reads=1, min_umi_per_intbc=2)
    df = cas.pp.filter_cells(df, min_umi=5)
    df = cas.pp.call_clones(df, method=method, **kw)
    df = cas.pp.filter_integrations(df)
    df = cas.pp.filter_cells(df, min_intbc_frac=0.5)
    return cas.pp.filter_doublets(df)


def _truth_cells(sim, df):
    cells = cas.pp.qc.cell_metrics(df)
    return cells.join(sim.cells.set_index("cellBC_10x"), rsuffix="_true", how="inner")


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


def test_filters_annotate(sim_factory):
    sim = sim_factory("weissman", "multi")
    df = cas.pp.call_alleles(sim.bam, "weissman")
    out = cas.pp.filter_molecules(df, min_reads=3, min_umi_per_intbc=2)
    assert len(out) == len(df)
    assert {"pass_min_reads", "pass_allele_frac", "pass_umis_per_intbc", "pass"} <= set(out.columns)
    assert (out.loc[out["pass"], "n_reads"] >= 3).all()
    assert out.attrs["qc_params"]["min_reads"] == 3
    cells = cas.pp.filter_cells(out, min_umi=20)
    assert cells.attrs["qc_params"]["min_umi"] == 20


def test_single_clone_pipeline(sim_factory):
    sim = sim_factory("weissman", "single")
    df = _process(sim, "single")
    kept = df.attrs["integrations"]["0"]
    kinds = sim.integrations.set_index("intBC")["kind"]
    assert not any(kinds.get(i) in ("duplicated", "silenced") for i in kept)
    assert len(kept) == (kinds == "normal").sum()
    cells = _truth_cells(sim, df)
    flagged, true = cells["doublet"].astype(bool), cells["doublet_true"].astype(bool)
    assert (flagged & true).sum() >= 0.6 * true.sum()
    assert (flagged & ~true).sum() <= 2


@requires_nova
def test_multi_clone_pipeline(sim_factory):
    from sklearn.metrics import adjusted_rand_score

    sim = sim_factory("weissman", "multi", cells_per_clone=40)
    df = _process(
        sim, "nova_clone", min_intbc_umi_count=2, min_intbc_cellbc_count=5, min_cells_assigned=10
    )
    cells = _truth_cells(sim, df)
    single = cells[
        ~cells["doublet_true"] & cells["clone"].notna() & (cells["clone"] != "multiplet")
    ]
    assert adjusted_rand_score(single["clone_true"], single["clone"].astype(str)) > 0.95
    flagged, true = cells["doublet"].astype(bool), cells["doublet_true"].astype(bool)
    assert (flagged & cells["heterotypic"]).sum() >= 0.9 * cells["heterotypic"].sum()
    assert (flagged & ~true).sum() <= 2


def test_user_clones(sim_factory):
    sim = sim_factory("weissman", "multi")
    clones = dict(zip(sim.cells["cellBC_10x"], sim.cells["clone"], strict=True))
    df = _process(sim, "single", clones=clones)
    assert df.attrs["qc_params"]["clone_method"] == "user"
    assert len(df.attrs["clone_intbcs"]) >= 3
    assert "doublet_clone_mixture" in df.columns


def test_alleles_to_characters(sim_factory, tmp_path):
    sim = sim_factory("petracer", "single")
    df = _process(sim, "single")
    tdata = cas.pp.alleles_to_characters(df, compute_priors=True)
    chars = tdata.obsm["characters"]
    n_int = len(df.attrs["integrations"]["0"])
    assert chars.shape[1] == 3 * n_int
    assert set(np.unique(chars.to_numpy())) <= {"-", "*"} | {str(i) for i in range(1, 9)}
    assert (tdata.uns["missing_state"], tdata.uns["unmodified_state"]) == ("-", "*")
    assert not tdata.obs["doublet"].astype(bool).any()
    states = tdata.uns["character_states"]
    assert set(states.loc[states["state"] != "*", "allele"]) <= {str(i) for i in range(1, 9)}
    assert tdata.uns["priors"]
    # the other mappings encode the same states
    ints = cas.pp.alleles_to_characters(df, mapping="int")
    alleles = cas.pp.alleles_to_characters(df, mapping="allele")
    assert "priors" not in ints.uns
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
    # h5td round trip keeps characters, priors and QC
    tdata.write_h5td(tmp_path / "t.h5td")
    t2 = td.read_h5td(tmp_path / "t.h5td")
    pd.testing.assert_frame_equal(t2.obsm["characters"], chars, check_dtype=False)
    assert "cell_qc" in t2.uns["preprocess"]
    # a solver runs directly on the output (priors survive the round trip)
    cas.solver.nj(t2, priors=True)
    assert "nj" in t2.obst


def test_compute_empirical_indel_priors(sim_factory):
    sim = sim_factory("weissman", "single")
    df = _process(sim, "single")
    priors = cas.pp.compute_empirical_indel_priors(df)
    assert len(priors) > 0 and priors["freq"].between(0, 1).all()
    assert not any("None" in i for i in priors.index)
    assert priors["count"].max() <= df.loc[df["pass"], "intBC"].nunique()
