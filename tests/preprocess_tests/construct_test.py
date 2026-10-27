"""Tests for constructs, presets, layouts and references."""

import pytest

import cassiopeia as cas
from cassiopeia.preprocess.construct import Construct, Site


@pytest.mark.parametrize("name", ["weissman", "petracer", "typewriter", "gestalt", "baseline"])
def test_presets_load(name):
    c = cas.pp.load_construct(name)
    assert c.name == name
    assert len(c.sites) > 0
    for s in c.sites:
        assert 0 <= s.position <= len(c.reference)


def test_preset_coordinates():
    c = cas.pp.load_construct("weissman")
    assert c.intbc == (20, 34)
    assert [s.position for s in c.sites] == [112, 166, 220]
    assert cas.pp.load_construct("petracer").intbc == (272, 302)
    assert cas.pp.load_construct("gestalt").intbc is None
    assert len(cas.pp.load_construct("baseline").sites) == 50


def test_intbc_from_n_run():
    c = Construct("x", "ACGT" * 5 + "NNNNNN" + "ACGT" * 10, [Site("s1", 40)])
    assert c.intbc == (20, 26)
    with pytest.raises(ValueError):
        Construct("x", "NNNACGTNNN" + "A" * 50, [Site("s1", 40)])
    c = Construct("x", "NNNACGTNNN" + "A" * 50, [Site("s1", 40)], intbc=(7, 10))
    assert c.intbc == (7, 10)


def test_construct_validation():
    with pytest.raises(ValueError):
        Construct("x", "A" * 50, [Site("s", 10, edit_model="nonsense")])
    with pytest.raises(ValueError):
        Construct("x", "A" * 50, [Site("s", 10), Site("s", 20)])
    with pytest.raises(ValueError):
        Construct("x", "A" * 50, [Site("s", 10, edit_model="substitution")])  # needs end


def test_yaml_roundtrip(tmp_path):
    c = cas.pp.load_construct("petracer")
    c.to_yaml(tmp_path / "c.yaml")
    c2 = cas.pp.load_construct(tmp_path / "c.yaml")
    assert c2.reference == c.reference
    assert c2.intbc == c.intbc
    assert [s.lookup for s in c2.sites] == [s.lookup for s in c.sites]


def test_unknown_construct():
    with pytest.raises(ValueError, match="Presets"):
        cas.pp.load_construct("does_not_exist")


def test_layouts():
    lay = cas.pp.get_layout("10xv3")
    assert lay.cb_length == 16 and lay.umi_length == 12
    assert cas.pp.get_layout("indropsv3").cb_length == 16
    custom = cas.pp.get_layout(
        {"n_reads": 2, "cb": [(0, 0, 8)], "umi": [(0, 8, 14)], "cdna": (1, 0, None)}
    )
    assert custom.cb_length == 8
    with pytest.raises(ValueError):
        cas.pp.get_layout("nope")


def test_build_reference(tmp_path):
    out = cas.pp.build_reference("weissman", tmp_path)
    fasta = open(out["fasta"]).read()
    assert fasta.startswith(">weissman\n")
    assert "N" * 14 in fasta.replace("\n", "")
    gtf = open(out["gtf"]).read()
    assert 'gene_id "weissman"' in gtf and "\texon\t1\t750\t" in gtf
    assert "STAR --runMode genomeGenerate" in out["star_genome"]
    assert "--soloUMIlen 12" in out["starsolo"]
    assert "cellranger mkref" in out["cellranger_mkref"]
