"""Build references for aligning target-site reads with STARsolo / Cell Ranger."""

from __future__ import annotations

import math
from pathlib import Path

from cassiopeia.mixins import logger

from .construct import Construct, load_construct
from .layout import Layout, parse_layout


def _star_reads(lay: Layout, whitelist: str | None) -> tuple[str, str] | None:
    """``--readFilesIn`` and STARsolo arguments for a layout (None if STARsolo cannot read it)."""
    target = f"<R{lay.cdna[0] + 1}.fastq.gz>"
    if not lay.cb and not lay.umi:  # bulk: plain STAR
        return f"--readFilesIn {target}", ""
    reads = {r for r, _, _ in lay.cb + lay.umi}
    if not lay.cb or len(lay.umi) != 1 or len(reads) > 1 or lay.cdna[0] in reads:
        return None
    barcode = f"<R{reads.pop() + 1}.fastq.gz>"
    (_, us, ue) = lay.umi[0]
    if len(lay.cb) == 1:
        (_, cs, ce) = lay.cb[0]
        solo = (
            f"--soloType CB_UMI_Simple --soloCBstart {cs + 1} --soloCBlen {ce - cs} "
            f"--soloUMIstart {us + 1} --soloUMIlen {ue - us} --soloBarcodeReadLength 0 "
            f"--soloCBwhitelist {whitelist or '<whitelist>'} "
            "--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts"
        )
    else:  # split barcode: one whitelist per segment
        positions = " ".join(f"0_{s}_0_{e - 1}" for _, s, e in lay.cb)
        whitelists = " ".join(f"<whitelist_{i + 1}>" for i in range(len(lay.cb)))
        solo = (
            f"--soloType CB_UMI_Complex --soloCBposition {positions} "
            f"--soloUMIposition 0_{us}_0_{ue - 1} --soloCBwhitelist {whitelists} "
            "--soloCBmatchWLtype 1MM"
        )
    solo += " --soloUMIdedup 1MM_Directional_UMItools --soloFeatures Gene"
    return f"--readFilesIn {target} {barcode}", solo


def build_reference(
    construct: Construct | str,
    out_dir: str | Path,
    layout: str = "10xv3",
    whitelist: str | None = None,
) -> dict:
    """Write a target-site reference (FASTA + GTF) and suggested aligner commands.

    The FASTA holds the construct reference with the intBC masked as N (one
    contig named after the construct). The GTF annotates the whole contig as
    a single-exon gene, which STARsolo and Cell Ranger need to assign
    corrected UMIs (``UB``) to target-site reads. Append both files to a host
    genome FASTA/GTF to align gene expression and target-site reads together,
    or use them alone for target-site libraries.

    Args:
        construct: Construct, preset name or YAML path.
        out_dir: Output directory.
        layout: Read layout (preset or pattern, see :func:`~cassiopeia.pp.align`)
            used to suggest the STAR / STARsolo command.
        whitelist: Cell barcode whitelist path used in the suggested commands.

    Returns:
        Dict with ``fasta``, ``gtf`` paths and ``star_genome``, ``starsolo``,
        ``cellranger_mkref`` command strings. ``starsolo`` is a plain STAR
        command for layouts without barcodes, and None when STARsolo cannot
        read the layout (barcodes split over several reads or sharing the
        target-site read).
    """
    c = load_construct(construct)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fasta = out / f"{c.name}.fa"
    gtf = out / f"{c.name}.gtf"
    with open(fasta, "w") as f:
        f.write(f">{c.name}\n")
        for i in range(0, len(c.reference), 80):
            f.write(c.reference[i : i + 80] + "\n")
    n = len(c.reference)
    attrs = f'gene_id "{c.name}"; transcript_id "{c.name}"; gene_name "{c.name}"; gene_biotype "lineage_tracer";'
    with open(gtf, "w") as f:
        f.write(f"{c.name}\tcassiopeia\tgene\t1\t{n}\t.\t+\t.\t{attrs}\n")
        f.write(f"{c.name}\tcassiopeia\ttranscript\t1\t{n}\t.\t+\t.\t{attrs}\n")
        f.write(f"{c.name}\tcassiopeia\texon\t1\t{n}\t.\t+\t.\t{attrs}\n")

    sa_bases = max(4, min(14, int(math.log2(max(n, 2)) / 2 - 1)))
    genome_dir = out / "star_index"
    star_genome = (
        f"STAR --runMode genomeGenerate --genomeDir {genome_dir} --genomeFastaFiles {fasta} "
        f"--sjdbGTFfile {gtf} --genomeSAindexNbases {sa_bases} --sjdbOverhang 100"
    )
    lay = parse_layout(layout)
    star = _star_reads(lay, whitelist)
    starsolo = None
    if star is None:
        logger.warning(
            f"STARsolo needs the cell barcode and UMI on one read other than the target-site "
            f"read; layout {lay.pattern!r} can only be aligned with pp.align."
        )
    else:
        reads, solo = star
        attrs = "NH HI AS nM CR UR CB UB" if solo else "NH HI AS nM"
        starsolo = (
            f"STAR --genomeDir {genome_dir} {reads} --readFilesCommand zcat "
            + (f"{solo} " if solo else "")
            + f"--outSAMtype BAM SortedByCoordinate --outSAMattributes {attrs} "
            "--alignEndsType Local --outFilterScoreMinOverLread 0.3 "
            "--outFilterMatchNminOverLread 0.3 --runThreadN 8 --outFileNamePrefix <prefix>"
        )
    mkref = f"cellranger mkref --genome={c.name} --fasta={fasta} --genes={gtf}"
    return {
        "fasta": str(fasta),
        "gtf": str(gtf),
        "star_genome": star_genome,
        "starsolo": starsolo,
        "cellranger_mkref": mkref,
    }
