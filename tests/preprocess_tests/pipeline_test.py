"""Tests for pp.run and the cassiopeia-preprocess command line."""

import pytest
import yaml

import cassiopeia as cas
from cassiopeia.preprocess import cli


def test_run_from_fastq(sim_factory, tmp_path):
    sim = sim_factory("petracer", "single")
    cfg = {
        "construct": "petracer",
        "input": {"fastq": sim.fastqs, "layout": "10xv3", "whitelist": sim.whitelist},
        "output": {"dir": str(tmp_path / "out"), "name": "s1", "plots": True},
        "threads": 2,
        "call_alleles": {"min_reads_per_umi": 1},
        "filter_alleles": {"min_umi_per_allele": 2},
        "filter_cells": {"min_umi": 5},
    }
    df = cas.pp.run(cfg)
    out = tmp_path / "out"
    assert (out / "s1.aligned.bam").exists() and (out / "s1.cells.csv").exists()
    assert not list(out.glob("*.h5td"))
    assert {p.name for p in (out / "qc").iterdir()} >= {
        "align.png",
        "call_alleles.png",
        "filter_alleles.png",
        "filter_cells.png",
        "filter_cells_coverage.png",
        "filter_doublets.png",
    }
    assert "call_clones.png" not in {p.name for p in (out / "qc").iterdir()}
    # the pipeline ends at the allele table (UMIs collapsed into alleles), saved as parquet
    alleles = cas.pp.read_alleles(out / "s1.alleles.parquet")
    assert alleles.shape == df.shape and alleles.attrs["qc_params"] == df.attrs["qc_params"]
    assert (alleles["pass"].to_numpy() == df["pass"].to_numpy()).all()
    assert "n_umis" in alleles.columns and "UMI" not in alleles.columns
    assert "clone" not in alleles.columns and "pass_doublet" in alleles.columns
    tdata = cas.pp.alleles_to_characters(alleles)
    assert tdata.n_obs > 0.7 * (~sim.cells["doublet"]).sum()


def test_run_from_bam_yaml(sim_factory, tmp_path):
    sim = sim_factory("weissman", "single")
    cfg = {
        "construct": "weissman",
        "input": {"bam": sim.bam},
        "output": {"dir": str(tmp_path / "out"), "plots": False},
        "call_alleles": {"min_reads_per_umi": 1},
        "filter_alleles": {"min_allele_frac": 0.2, "allow_allele_conflicts": True},
        "filter_cells": {"min_umi": 5},
        "filter_integrations": {"max_conflict": 1.0},
        "filter_doublets": False,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    cas.pp.run(path)
    df = cas.pp.read_alleles(tmp_path / "out" / "weissman.alleles.parquet")
    assert "pass_doublet" not in df.columns
    # conflicting alleles are kept, so the characters have ambiguous states
    tdata = cas.pp.alleles_to_characters(df, allow_ambiguous=True, mapping="int")
    chars = tdata.obsm["characters"]
    assert chars.shape[1] > 0 and chars.map(lambda v: isinstance(v, tuple)).to_numpy().any()


def test_config_errors():
    with pytest.raises(Exception, match="construct"):
        cas.pp.run({"input": {"bam": "x.bam"}})
    with pytest.raises(Exception, match="fastq"):
        cas.pp.run({"construct": "weissman", "input": {}})
    with pytest.raises(Exception, match="Unknown config keys"):
        cas.pp.run({"construct": "weissman", "input": {"bam": "x"}, "typo_step": {}})
    with pytest.raises(Exception, match="ends at the allele table"):
        cas.pp.run({"construct": "weissman", "input": {"bam": "x"}, "alleles_to_characters": {}})


def test_cli(sim_factory, tmp_path, capsys):
    assert cli.main(["presets"]) == 0
    assert "petracer" in capsys.readouterr().out
    assert cli.main(["example-config"]) == 0
    example = yaml.safe_load(capsys.readouterr().out)
    assert example["construct"] == "petracer"
    assert cli.main(["build-reference", "gestalt", str(tmp_path / "ref")]) == 0
    assert (tmp_path / "ref" / "gestalt.fa").exists()
    sim = sim_factory("weissman", "clean")
    assert (
        cli.main(["align", *sim.fastqs, "-o", str(tmp_path / "a.bam"), "-c", "weissman", "-t", "2"])
        == 0
    )
    assert (
        cli.main(
            [
                "call-alleles",
                str(tmp_path / "a.bam"),
                "-o",
                str(tmp_path / "m.parquet"),
                "-c",
                "weissman",
            ]
        )
        == 0
    )
    assert (tmp_path / "m.parquet").exists()
