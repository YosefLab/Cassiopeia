"""Build references for aligning target-site reads with STARsolo / Cell Ranger."""

from __future__ import annotations

import math
from pathlib import Path

from .construct import Construct, load_construct
from .layout import get_layout

# STARsolo read-structure arguments per layout.
_STARSOLO = {
    "10xv2": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 10",
    "10xv3": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12",
    "10xv3.1": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12",
    "10xv4": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12",
    "10x5p": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 10",
    "dropseq": "--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 12 --soloUMIstart 13 --soloUMIlen 8",
}


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
        layout: Read layout used to suggest the STARsolo command.
        whitelist: Cell barcode whitelist path used in the suggested commands.

    Returns:
        Dict with ``fasta``, ``gtf`` paths and ``star_genome``, ``starsolo``,
        ``cellranger_mkref`` command strings.
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
    lay = get_layout(layout)
    solo = _STARSOLO.get(
        lay.name, "--soloType CB_UMI_Simple  # set --soloCBstart/len and --soloUMIstart/len"
    )
    wl = whitelist or "<whitelist>"
    starsolo = (
        f"STAR --genomeDir {genome_dir} --readFilesIn <R2.fastq.gz> <R1.fastq.gz> "
        f"--readFilesCommand zcat {solo} --soloCBwhitelist {wl} "
        "--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIdedup 1MM_Directional_UMItools "
        "--soloFeatures Gene --outSAMtype BAM SortedByCoordinate "
        "--outSAMattributes NH HI AS nM CR UR CB UB "
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
