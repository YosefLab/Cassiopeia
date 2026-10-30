"""Tests for constructs, presets, layouts and references."""

import pytest

import cassiopeia as cas
from cassiopeia.preprocess.construct import Construct, Site
from cassiopeia.preprocess.layout import PRESETS, parse_layout


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


def test_layout_patterns():
    lay = parse_layout("10xv3")
    assert (lay.n_reads, lay.cb, lay.umi, lay.cdna) == (
        2,
        ((0, 0, 16),),
        ((0, 16, 28),),
        (1, 0, None),
    )
    assert lay.cb_length == 16 and lay.umi_length == 12
    assert parse_layout(" 16c12u, t ") == parse_layout("16C12U,T")
    slide = parse_layout("slideseq2")
    assert slide.cb == ((0, 0, 8), (0, 26, 32)) and slide.umi == ((0, 32, 41),)
    indrops = parse_layout("8C,8C6U,T")
    assert indrops.n_reads == 3 and indrops.cb_length == 16 and indrops.cdna == (2, 0, None)
    assert parse_layout("T").cb == () and parse_layout("T").cdna == (0, 0, None)
    # an ignored read, and the target after a skipped primer
    lay = parse_layout("X,10U,20XT")
    assert lay.n_reads == 3 and lay.umi == ((1, 0, 10),) and lay.cdna == (2, 20, None)
    for name in PRESETS:
        assert parse_layout(name).cdna is not None
    for bad in ["nope", "16C12U", "16C,T,T", "T16C", "16Q,T", "0C,T", "16C,,T", "C16,T"]:
        with pytest.raises(ValueError, match="Invalid layout"):
            parse_layout(bad)


def test_build_reference(tmp_path):
    out = cas.pp.build_reference("weissman", tmp_path)
    fasta = open(out["fasta"]).read()
    assert fasta.startswith(">weissman\n")
    assert "N" * 14 in fasta.replace("\n", "")
    gtf = open(out["gtf"]).read()
    assert 'gene_id "weissman"' in gtf and "\texon\t1\t750\t" in gtf
    assert "STAR --runMode genomeGenerate" in out["star_genome"]
    assert "--readFilesIn <R2.fastq.gz> <R1.fastq.gz>" in out["starsolo"]
    assert "--soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12" in out["starsolo"]
    assert "cellranger mkref" in out["cellranger_mkref"]


def test_starsolo_from_layout(tmp_path):
    slide = cas.pp.build_reference("weissman", tmp_path, layout="slideseq2")["starsolo"]
    assert "--soloType CB_UMI_Complex --soloCBposition 0_0_0_7 0_26_0_31" in slide
    assert "--soloUMIposition 0_32_0_40" in slide
    assert "<whitelist_1> <whitelist_2>" in slide
    bulk = cas.pp.build_reference("weissman", tmp_path, layout="bulk")["starsolo"]
    assert "--readFilesIn <R1.fastq.gz> " in bulk and "--solo" not in bulk
    # barcodes split over two reads: STARsolo cannot read it
    assert cas.pp.build_reference("weissman", tmp_path, layout="indropsv3")["starsolo"] is None
