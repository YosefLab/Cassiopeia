"""Tests for pp.call_alleles and pp.align on simulated data with ground truth."""

from pathlib import Path

import pandas as pd
import pysam
import pytest

import cassiopeia as cas
from cassiopeia.preprocess._resolve import resolve_alleles

PRESETS = ["weissman", "petracer", "typewriter", "gestalt", "baseline"]
# Outputs of the pre-3.0 pipeline on the simulations below (see data/README.md).
DATA = Path(__file__).parent / "data"
LEGACY_CASES = {
    "substitutions": {"seq_error": 0.003},
    "indels": {"seq_error": 0.001, "indel_error": 0.001, "read_length": 300},
}


def _agreement(sim, df):
    sites = sim.construct.site_names
    truth = sim.alleles[sim.alleles["copy"] == 0].drop_duplicates(["cellBC", "intBC"])
    dup = set(sim.integrations.loc[sim.integrations["kind"] == "duplicated", "intBC"])
    truth = truth[~truth["intBC"].isin(dup)].set_index(["cellBC", "intBC"])[sites]
    res = resolve_alleles(df, sites, use_pass=False)
    res["cellBC"] = res["cellBC"].astype(str).str.replace("-1", "", regex=False)
    res["intBC"] = res["intBC"].astype(str)
    res = res.set_index(["cellBC", "intBC"])
    common = truth.index.intersection(res.index)
    t = truth.loc[common, sites].astype(object).fillna("<NA>").to_numpy()
    r = res.loc[common, sites].astype(object).fillna("<NA>").to_numpy()
    return len(common) / len(truth), (t == r).mean()


@pytest.mark.parametrize("preset", PRESETS)
def test_call_alleles_bam_matches_truth(sim_factory, preset):
    sim = sim_factory(preset, "clean")
    df = cas.pp.call_alleles(sim.bam, preset, n_threads=2)
    recovered, agree = _agreement(sim, df)
    assert agree == 1.0
    assert recovered > 0.6  # silenced integrations are mostly unobserved
    assert set(sim.construct.site_names) <= set(df.columns)
    assert df.attrs["sites"] == sim.construct.site_names
    assert df["cellBC"].dtype == "category"


@pytest.mark.parametrize("preset", PRESETS)
def test_align_fastq_matches_truth(sim_factory, preset, tmp_path):
    sim = sim_factory(preset, "clean")
    res = cas.pp.align(sim.fastqs, tmp_path / "a.bam", preset, whitelist=sim.whitelist, n_threads=2)
    assert res["stats"]["records_aligned"] == res["stats"]["records_written"]
    df = cas.pp.call_alleles(tmp_path / "a.bam", preset, n_threads=2)
    recovered, agree = _agreement(sim, df)
    assert agree == 1.0
    # one molecule per simulated UMI
    assert len(df) == len(sim.molecules)


def test_fastq_and_bam_paths_converge(sim_factory, tmp_path):
    sim = sim_factory("weissman", "multi")
    cas.pp.align(sim.fastqs, tmp_path / "a.bam", "weissman", whitelist=sim.whitelist, n_threads=2)
    fq = cas.pp.call_alleles(tmp_path / "a.bam", "weissman", n_threads=2)
    bam = cas.pp.correct_intbcs(cas.pp.call_alleles(sim.bam, "weissman", n_threads=2))
    sites = sim.construct.site_names

    def resolved(df):
        r = resolve_alleles(df, sites, use_pass=False)
        r["cellBC"] = r["cellBC"].astype(str).str.replace("-1", "", regex=False)
        r["intBC"] = r["intBC"].astype(str)
        return r.set_index(["cellBC", "intBC"])[sites].astype(object).fillna("<NA>")

    a, b = resolved(fq), resolved(bam)
    common = a.index.intersection(b.index)
    assert len(common) > 0.95 * len(a)
    assert (a.loc[common].to_numpy() == b.loc[common].to_numpy()).mean() > 0.995


def test_align_corrects_barcodes_and_umis(sim_factory, tmp_path):
    sim = sim_factory("weissman", "multi")
    res = cas.pp.align(
        sim.fastqs, tmp_path / "a.bam", "weissman", whitelist=sim.whitelist, n_threads=2
    )
    st = res["stats"]
    assert st["reads_cell_corrected"] > 0
    assert st["umis_merged"] > 0
    assert st["molecules"] == len(sim.molecules)
    tags = [
        (r.get_tag("CB"), r.get_tag("UB"), r.get_tag("ZR"))
        for r in pysam.AlignmentFile(str(tmp_path / "a.bam"))
    ]
    got = pd.DataFrame(tags, columns=["cellBC", "UMI", "ZR"]).merge(
        sim.molecules, on=["cellBC", "UMI"]
    )
    # single-read UMIs whose only read carries a UMI error keep the error
    assert len(got) >= 0.995 * len(sim.molecules)
    assert (got["ZR"] == got["n_reads"]).mean() > 0.95
    # without a whitelist, barcodes are not corrected
    res2 = cas.pp.align(sim.fastqs, tmp_path / "b.bam", "weissman", n_threads=2)
    assert res2["stats"]["reads_cell_corrected"] == 0


def test_align_dedup_sequence_and_lanes(sim_factory, tmp_path):
    sim = sim_factory("petracer", "clean")
    res = cas.pp.align(
        [sim.fastqs, sim.fastqs], tmp_path / "a.bam", "petracer", dedup="sequence", n_threads=2
    )
    assert res["stats"]["reads"] == 2 * len(pd.read_csv(sim.fastqs[0], header=None)) // 4
    df = cas.pp.call_alleles(tmp_path / "a.bam", "petracer", n_threads=2)
    _, agree = _agreement(sim, df)
    assert agree == 1.0


def test_align_errors(sim_factory, tmp_path):
    sim = sim_factory("weissman", "clean")
    with pytest.raises(Exception, match="FASTQ"):
        cas.pp.align([sim.fastqs[0]], tmp_path / "a.bam", "weissman")
    with pytest.raises(Exception, match="not found"):
        cas.pp.align(["nope_R1.fq.gz", "nope_R2.fq.gz"], tmp_path / "a.bam", "weissman")


def test_call_alleles_options(sim_factory):
    sim = sim_factory("weissman", "multi")
    cells = sim.cells["cellBC_10x"].iloc[:10].tolist()
    df = cas.pp.call_alleles(sim.bam, "weissman", cells=cells, n_threads=1)
    assert set(df["cellBC"].astype(str)) <= set(cells)
    assert df.attrs["read_stats"]["not_in_cells"] > 0
    # scanning without the index gives the same table
    full = cas.pp.call_alleles(sim.bam, "weissman", contigs=None, n_threads=2)
    idx = cas.pp.call_alleles(sim.bam, "weissman", n_threads=2)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), idx.reset_index(drop=True))
    # bulk mode: one row per (intBC, allele) with read counts
    bulk = cas.pp.call_alleles(sim.bam, "weissman", cb_tag=None, umi_tag=None, sample="s1")
    assert bulk["cellBC"].nunique() == 1 and bulk["UMI"].isna().all()
    assert bulk["n_reads"].sum() == idx.attrs["read_stats"]["reads_used"]
    with pytest.raises(Exception, match="No contig"):
        cas.pp.call_alleles(sim.bam, "petracer")


@pytest.mark.parametrize("case", list(LEGACY_CASES))
def test_legacy_golden(sim_factory, tmp_path, case):
    """pp.align + call_alleles(legacy=True) reproduces the legacy pipeline.

    Reads with a sequencing indel rejoin their UMI through the alignment-based
    merge, as in the legacy collapse, so read counts still match. When a UMI's
    biggest clusters are tied single reads both pipelines keep an arbitrary
    one, so a few alleles differ.
    """
    errors = LEGACY_CASES[case]
    sim = sim_factory("weissman", "multi", cells_per_clone=8, **errors)
    old = pd.read_csv(
        DATA / f"legacy_alleles_{case}.tsv.gz", sep="\t", dtype=str, keep_default_na=False
    ).astype({"readCount": int})
    cas.pp.align(
        sim.fastqs, tmp_path / "new.bam", "weissman", whitelist=sim.whitelist, umi_max_dist=0
    )
    new = cas.pp.call_alleles(tmp_path / "new.bam", "weissman", legacy=True, umi_max_dist=2)
    new = new.astype(
        {"cellBC": str, "UMI": str, "intBC": str, "r1": object, "r2": object, "r3": object}
    )
    new = new.fillna({"r1": "", "r2": "", "r3": ""})
    m = old.merge(new, on=["cellBC", "UMI", "intBC"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    same_alleles = (
        both[["r1_x", "r2_x", "r3_x"]].to_numpy() == both[["r1_y", "r2_y", "r3_y"]].to_numpy()
    ).all(axis=1)
    same_reads = (both["readCount"] == both["n_reads"]).to_numpy()
    if "indel_error" not in errors:
        assert len(both) == len(m)
        assert same_alleles.all() and same_reads.all()
    else:
        assert len(both) >= 0.95 * len(m)
        assert same_reads.mean() >= 0.99
        assert same_alleles.mean() >= 0.95
