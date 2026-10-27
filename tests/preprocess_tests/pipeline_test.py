"""Tests for pp.run and the cassiopeia-preprocess command line."""

import pytest
import treedata as td
import yaml

import cassiopeia as cas
from cassiopeia.preprocess import cli


def test_run_from_fastq(sim_factory, tmp_path):
    sim = sim_factory("petracer", "single")
    cfg = {
        "construct": "petracer",
        "input": {"fastq": sim.fastqs, "layout": "10xv3", "whitelist": sim.whitelist},
        "output": {"dir": str(tmp_path / "out"), "name": "s1", "plots": True, "checkpoint": True},
        "threads": 2,
        "filter_molecules": {"min_reads": 1, "min_umi_per_intbc": 2},
        "filter_cells": {"min_umi": 5},
        "call_clones": {"method": "single"},
    }
    tdata = cas.pp.run(cfg)
    out = tmp_path / "out"
    assert (out / "s1.h5td").exists() and (out / "s1.aligned.bam").exists()
    assert (out / "s1.molecules.parquet").exists() and (out / "s1.cells.csv").exists()
    assert {p.name for p in (out / "qc").iterdir()} >= {
        "align.png",
        "call_alleles.png",
        "filter_doublets.png",
    }
    t2 = td.read_h5td(out / "s1.h5td")
    assert t2.obsm["characters"].shape == tdata.obsm["characters"].shape
    assert tdata.n_obs > 0.7 * (~sim.cells["doublet"]).sum()


def test_run_from_bam_yaml(sim_factory, tmp_path):
    sim = sim_factory("weissman", "single")
    cfg = {
        "construct": "weissman",
        "input": {"bam": sim.bam},
        "output": {"dir": str(tmp_path / "out"), "plots": False},
        "filter_molecules": {"min_reads": 1, "min_umi_per_intbc": 2},
        "filter_cells": {"min_umi": 5},
        "call_clones": {"method": "single"},
        "filter_doublets": False,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    tdata = cas.pp.run(path)
    assert "doublet" not in tdata.obs.columns
    assert tdata.obsm["characters"].shape[1] > 0


def test_config_errors():
    with pytest.raises(Exception, match="construct"):
        cas.pp.run({"input": {"bam": "x.bam"}})
    with pytest.raises(Exception, match="fastq"):
        cas.pp.run({"construct": "weissman", "input": {}})
    with pytest.raises(Exception, match="Unknown config keys"):
        cas.pp.run({"construct": "weissman", "input": {"bam": "x"}, "typo_step": {}})


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
