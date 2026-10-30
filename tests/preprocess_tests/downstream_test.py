"""Tests for the allele-table steps: correction, filters, clones, doublets, characters."""

import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.preprocess.clones import intbc_table

from .conftest import requires_nova


def _process(sim, clones=False, collapse=False, correct=True, **kw):
    df = cas.pp.call_alleles(
        sim.bam, sim.construct, n_threads=2, min_reads_per_umi=1, collapse_umis=collapse
    )
    if correct:
        df = cas.pp.correct_intbcs(df)
    df = cas.pp.filter_alleles(df, min_umi_per_allele=2)
    df = cas.pp.filter_cells(df, min_umi=5)
    if clones:
        df = cas.pp.call_clones(df, **kw)
    df = cas.pp.filter_integrations(df, min_detection=0.4)
    df = cas.pp.filter_cells(df, min_intbc_frac=0.5)
    return cas.pp.filter_doublets(df)


def _truth_cells(sim, df):
    cells = cas.pp.cell_metrics(df)
    truth = sim.cells.set_index("cellBC_10x")
    truth = truth.rename(columns={c: f"{c}_true" for c in truth.columns if c in cells.columns})
    return cells.join(truth, how="inner")


def test_correct_intbcs(sim_factory):
    sim = sim_factory("weissman", "multi")
    raw = cas.pp.call_alleles(sim.bam, "weissman", collapse_umis=False)
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
    col = cas.pp.correct_intbcs(cas.pp.collapse_umis(raw, plot=False))
    assert col["n_umis"].sum() == len(raw) and col["n_reads"].sum() == df["n_reads"].sum()
    assert set(col["intBC"].astype(str)) == set(df["intBC"].astype(str))
    # allele fractions are recomputed for the corrected (cell, intBC) pairs
    for d in (df, col):
        alleles = d.drop_duplicates(["cellBC", "intBC", *d.attrs["sites"]])
        pair = alleles.groupby(["cellBC", "intBC"], observed=True)["allele_frac"].sum()
        np.testing.assert_allclose(pair, 1, rtol=1e-5)


def _keyed(df, key):
    return df.astype(dict.fromkeys(key, str)).set_index(key).sort_index()


def test_collapse_umis(sim_factory):
    sim = sim_factory("weissman", "multi")
    raw = cas.pp.call_alleles(
        sim.bam, "weissman", collapse_umis=False, min_reads_per_umi=1, plot=False
    )
    hist = raw.attrs["read_stats"]["reads_per_umi"]
    assert sum(hist["umis"]) == len(raw)
    assert (
        dict(zip(hist["reads"], hist["umis"], strict=True))
        == raw["n_reads"].value_counts().to_dict()
    )
    sites = raw.attrs["sites"]
    key = ["cellBC", "intBC", *sites]
    # allele_frac: fraction of the (cell, intBC)'s UMIs that carry the allele
    s = raw.astype(dict.fromkeys(key, str))
    expected = s.groupby(key)["n_reads"].transform("size") / s.groupby(["cellBC", "intBC"])[
        "n_reads"
    ].transform("size")
    np.testing.assert_allclose(raw["allele_frac"], expected, rtol=1e-6)
    for thr in (1, 2, 3):
        col = cas.pp.collapse_umis(raw, min_reads_per_umi=thr, plot=False)
        kept = raw[raw["n_reads"] >= thr]
        assert "UMI" not in col.columns and col.attrs["sites"] == sites
        assert col["n_umis"].sum() == len(kept) and col["n_reads"].sum() == kept["n_reads"].sum()
        assert not _keyed(col, key).index.duplicated().any()
        pair = col.groupby(["cellBC", "intBC"], observed=True)["allele_frac"].sum()
        np.testing.assert_allclose(pair, 1, rtol=1e-5)
        # filtering and collapsing while reading the BAM gives the same table
        native = cas.pp.call_alleles(sim.bam, "weissman", min_reads_per_umi=thr, plot=False)
        a, b = _keyed(col, key), _keyed(native, key)
        pd.testing.assert_frame_equal(a[["n_umis", "n_reads"]], b[["n_umis", "n_reads"]])
        np.testing.assert_allclose(a["allele_frac"], b["allele_frac"], rtol=1e-5)
        st = native.attrs["read_stats"]
        assert st["low_read_umis"] == len(raw) - len(kept)
        assert st["low_read_umi_reads"] == raw["n_reads"].sum() - kept["n_reads"].sum()
        # one record per read in the simulated BAM
        assert st["low_read_umi_records"] == st["low_read_umi_reads"]
        assert st["reads_per_umi"] == hist  # before the filter
        assert native.attrs["qc_params"]["min_reads_per_umi"] == thr
    default = cas.pp.call_alleles(sim.bam, "weissman", plot=False)
    assert default.attrs["qc_params"]["min_reads_per_umi"] == 2
    with pytest.raises(Exception, match="already collapsed"):
        cas.pp.collapse_umis(default)


def test_filters_annotate(sim_factory):
    sim = sim_factory("weissman", "multi")
    df = cas.pp.call_alleles(sim.bam, "weissman", collapse_umis=False, plot=False)
    sites = df.attrs["sites"]
    key = ["cellBC", "intBC", *sites]
    out = cas.pp.filter_alleles(df, min_umi_per_allele=2, min_avg_reads_per_umi=4, plot=False)
    assert len(out) == len(df)
    flags = {"umis_per_allele", "avg_reads_per_umi", "allele_frac", "allele_conflict"}
    assert {f"pass_{f}" for f in flags} | {"pass"} <= set(out.columns)
    s = out.astype(dict.fromkeys(key, str))
    g = s[s["pass"]].groupby(key)
    assert (g.size() >= 2).all() and (g["n_reads"].sum() / g.size() >= 4).all()
    assert (s.loc[s["pass"], "allele_frac"] >= 0.5).all()
    # one allele per (cell, intBC), with the most UMIs among the candidates
    low = cas.pp.filter_alleles(df, min_allele_frac=0.2, plot=False)
    s = low.astype(dict.fromkeys(key, str))
    assert not s[s["pass"]].drop_duplicates(key).duplicated(["cellBC", "intBC"]).any()
    assert (~s["pass_allele_conflict"]).any()
    cand = s[s["pass_umis_per_allele"] & s["pass_avg_reads_per_umi"] & s["pass_allele_frac"]]
    best = cand.groupby(key).size().groupby(level=["cellBC", "intBC"]).max()
    kept = s[s["pass"]].groupby(key).size().droplevel(sites)
    assert (kept.to_numpy() == best.reindex(kept.index).to_numpy()).all()
    both = cas.pp.filter_alleles(df, min_allele_frac=0.2, allow_allele_conflicts=True, plot=False)
    assert both["pass_allele_conflict"].all()
    assert both["pass"].sum() == (cand.shape[0])
    # minor alleles count neither towards the cell's UMIs nor for clone calling
    assert cas.pp.cell_metrics(low)["n_umis"].sum() == low["pass"].sum()
    assert intbc_table(low)["UMI_count"].sum() == low["pass"].sum()
    # collapsed tables give the same result
    col = cas.pp.collapse_umis(df, plot=False)
    for kw in ({"min_avg_reads_per_umi": 4}, {"min_allele_frac": 0.2}):
        a = cas.pp.filter_alleles(df, plot=False, **kw)
        b = cas.pp.filter_alleles(col, plot=False, **kw)
        assert b.loc[b["pass"], "n_umis"].sum() == a["pass"].sum()
    cells = cas.pp.filter_cells(out, min_umi=20)
    assert cells.attrs["qc_params"]["min_umi"] == 20
    qc = cas.pp.filter_alleles(df, plot=False).attrs["qc_params"]
    assert qc["min_allele_frac"] == 0.5 and qc["min_umi_per_allele"] == 2
    assert qc["min_avg_reads_per_umi"] == 0 and qc["allow_allele_conflicts"] is False
    with pytest.raises(Exception, match="filter_integrations"):
        cas.pp.filter_cells(out, min_intbc_frac=0.5)
    with pytest.raises(Exception, match="kept no integrations"):
        cas.pp.filter_integrations(out, min_detection=1.1)


def test_single_clone_pipeline(sim_factory):
    sim = sim_factory("weissman", "single")
    df = _process(sim)
    assert "clone" not in df.columns
    kept = df.attrs["integrations"]["0"]
    kinds = sim.integrations.set_index("intBC")["kind"]
    assert not any(kinds.get(i) in ("duplicated", "silenced") for i in kept)
    assert len(kept) == (kinds == "normal").sum()
    cells = _truth_cells(sim, df)
    metrics = cas.pp.cell_metrics(df)
    assert "doublet" not in metrics.columns and "n_informative" not in metrics.columns
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
    all_cells = cas.pp.cell_metrics(df)
    assert metrics.loc["0", "n_cells"] == all_cells["pass_min_umi"].sum()


def test_collapsed_pipeline_matches(sim_factory):
    sim = sim_factory("weissman", "single")
    # without intBC correction, which can merge UMIs only before collapsing
    a = cas.pp.cell_metrics(_process(sim, correct=False))
    b = cas.pp.cell_metrics(_process(sim, collapse=True, correct=False))
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
    df = cas.pp.call_alleles(sim.bam, sim.construct, n_threads=2, min_reads_per_umi=1)
    df = cas.pp.filter_alleles(df, min_umi_per_allele=1)
    df = cas.pp.filter_cells(df, min_umi=5)
    df = cas.pp.call_clones(df, clones=clones)
    # filter_doublets and filter_integrations can run in either order
    other = cas.pp.filter_doublets(cas.pp.filter_integrations(df))
    df = cas.pp.filter_integrations(cas.pp.filter_doublets(df))
    for col in ("pass_doublet", "pass_integration"):
        assert (df[col] == other[col]).all()
    # cells flagged as doublets are left out of the integration metrics
    assert (~df["pass_doublet"]).any()
    with_doublets = cas.pp.integration_metrics(df.assign(pass_doublet=True))
    without = cas.pp.integration_metrics(df)
    assert without["clone_cells"].sum() < with_doublets["clone_cells"].sum()
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
    df = cas.pp.call_alleles(sim.bam, "weissman", min_reads_per_umi=1, plot=False)

    def prepare(**kw):
        d = cas.pp.filter_alleles(df, min_allele_frac=0.2, plot=False, **kw)
        return cas.pp.filter_integrations(d, max_conflict=1.0, plot=False)

    plain, conflicts = prepare(), prepare(allow_allele_conflicts=True)
    with pytest.raises(Exception, match="allow_ambiguous"):
        cas.pp.alleles_to_characters(conflicts)
    p = cas.pp.alleles_to_characters(plain, mapping="allele").obsm["characters"]
    a = cas.pp.alleles_to_characters(conflicts, mapping="allele", allow_ambiguous=True)
    a = a.obsm["characters"]
    assert a.index.equals(p.index) and a.columns.equals(p.columns)
    tuples = a.map(lambda v: isinstance(v, tuple)).to_numpy()
    assert tuples.any()
    # ambiguous entries hold distinct states, including the kept allele
    for r, c in zip(*np.nonzero(tuples), strict=True):
        v = a.iat[r, c]
        assert len(v) >= 2 and len(set(v)) == len(v) and "-" not in v
        assert p.iat[r, c] in (*v, "-")
    # everything else is unchanged where the kept allele was called
    same = ~tuples & (p.to_numpy() != "-")
    assert (p.to_numpy()[same] == a.to_numpy()[same]).all()
    ints = cas.pp.alleles_to_characters(conflicts, mapping="int", allow_ambiguous=True)
    v = ints.obsm["characters"].to_numpy()[tuples][0]
    assert isinstance(v, tuple) and all(isinstance(x, (int, np.integer)) for x in v)
    strs = cas.pp.alleles_to_characters(conflicts, allow_ambiguous=True).obsm["characters"]
    v = strs.to_numpy()[tuples][0]
    assert isinstance(v, tuple) and all(isinstance(x, str) for x in v)


def test_compute_indel_priors(sim_factory, tmp_path):
    sim = sim_factory("weissman", "single")
    df = _process(sim)
    priors = cas.pp.compute_indel_priors(df)
    assert len(priors) > 0 and priors["freq"].between(0, 1).all()
    assert not any("None" in i for i in priors.index.get_level_values("indel"))
    assert priors["count"].max() <= df.loc[df["pass"], "intBC"].nunique()
    # from the character matrix, mapping states back to alleles
    results = []
    for mapping in ("str", "int", "allele"):
        tdata = cas.pp.alleles_to_characters(df, mapping=mapping)
        results.append(cas.pp.compute_indel_priors(tdata).sort_index())
        # TreeData input: priors per character name and state, for the solvers
        chars, stored = tdata.obsm["characters"], tdata.uns["priors"]
        st = tdata.uns["character_states"]
        keys = zip(st["character"], st["state"].astype(str), strict=True)
        allele = dict(zip(keys, st["allele"], strict=True))
        assert stored and all(isinstance(s, str) for d in stored.values() for s in d)
        for name, states in stored.items():
            assert set(states) == {str(s) for s in chars[name].unique()} - {"*", "-", "0", "-1"}
            site = name.rsplit("-", 1)[1]
            for s, p in states.items():
                assert p == results[-1].loc[(site, allele.get((name, s), s)), "freq"]
        tdata.write_h5td(tmp_path / f"{mapping}.h5td")
        cas.solver.greedy(td.read_h5td(tmp_path / f"{mapping}.h5td"), priors=True)
    chars = cas.pp.alleles_to_characters(df, return_type="dataframe")
    results.append(cas.pp.compute_indel_priors(chars).sort_index())
    for r in results[1:]:
        pd.testing.assert_frame_equal(r, results[0])
    from_matrix = results[0]
    assert len(from_matrix) > 0 and set(from_matrix.index) <= set(priors.index)
    assert (from_matrix["count"] <= priors.loc[from_matrix.index, "count"]).all()
    with pytest.raises(ValueError, match="obs"):
        cas.pp.compute_indel_priors(tdata, grouping_variables=["clone", "intBC"])
