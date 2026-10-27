"""Small published lineage tracing datasets for testing and tutorials.

Each dataset is downloaded from ENA/GEO once and cut down to a few hundred
cells (keeping every read of those cells, including reads whose cell barcode
carries a sequencing error), then cached in ``$CASSIOPEIA_DATA`` (default
``~/.cache/cassiopeia``). Downloads are large (0.3-1.7 GB); run
:func:`fetch` on a compute node.

=====================  ==========  =======================================================
name                   construct   source
=====================  ==========  =======================================================
``petracer_4t1``       petracer    Koblan, Yost et al. 2025 Science, GSM8826705 (3 clones)
``kptracer_l6``        weissman    Yang et al. 2022 Cell, tumor L6 (PRJNA803321)
``typewriter_hek``     typewriter  Choi et al. 2022 Nature, SRR18280234 (monoclonal)
``scgestalt_zf2``      gestalt     Raj et al. 2018 Nat Biotech, GSM2813985 (inDrops)
``baseline_d21``       baseline    Winter ... McKenna 2025 NAR, SRR32369801 (bulk ONT)
=====================  ==========  =======================================================
"""

from __future__ import annotations

import collections
import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from cassiopeia.mixins import logger

ENA = "https://ftp.sra.ebi.ac.uk/vol1/fastq"
WHITELISTS = {
    "10x_v2": "https://teichlab.github.io/scg_lib_structs/data/10X-Genomics/737K-august-2016.txt.gz",
    "10x_v3": "https://teichlab.github.io/scg_lib_structs/data/10X-Genomics/3M-february-2018.txt.gz",
    "10x_v4": "https://teichlab.github.io/scg_lib_structs/data/10X-Genomics/3M-3pgex-may-2023.txt.gz",
}

DATASETS = {
    "petracer_4t1": {
        "construct": "petracer",
        "layout": "10xv3",
        "whitelist": "10x_v3",
        "fastq": [
            f"{ENA}/SRR325/071/SRR32558971/SRR32558971_1.fastq.gz",
            f"{ENA}/SRR325/071/SRR32558971/SRR32558971_2.fastq.gz",
        ],
        "truth": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8826nnn/GSM8826705/suppl/"
        "GSM8826705_4T1_Pool2_allele_counts.csv.gz",
        "clones": "nova_clone",
        "description": "PEtracer 4T1 static barcode pool 2 (3 clones), 10x 3' v3.1 LTC library.",
    },
    "kptracer_l6": {
        "construct": "weissman",
        "layout": "10xv2",
        "whitelist": "10x_v2",
        "fastq": [
            f"{ENA}/SRR178/018/SRR17885818/SRR17885818_1.fastq.gz",
            f"{ENA}/SRR178/018/SRR17885818/SRR17885818_2.fastq.gz",
        ],
        "clones": "nova_clone",
        "description": "KP-tracer tumor L6 target site library, 10x 3' v2.",
    },
    "typewriter_hek": {
        "construct": "typewriter",
        "layout": "10xv3",
        "whitelist": "10x_v3",
        "fastq": [
            f"{ENA}/SRR182/034/SRR18280234/SRR18280234_1.fastq.gz",
            f"{ENA}/SRR182/034/SRR18280234/SRR18280234_2.fastq.gz",
        ],
        "cb_translation": "3M-february-2018",
        "clones": "single",
        "truth": "https://raw.githubusercontent.com/shendurelab/DNATickerTape/main/"
        "Supplementary_File_2_DataTableMOI19.csv",
        "description": "DNA Typewriter monoclonal HEK293T TargetBC-5xTAPE-1 library, 10x 3' v3.1 "
        "with Feature Barcoding (cell barcodes need Cell Ranger's translation table).",
    },
    "scgestalt_zf2": {
        "construct": "gestalt",
        "layout": "16C6U,T",
        "whitelist": None,
        "fastq": [f"{ENA}/SRR617/009/SRR6176749/SRR6176749.fastq.gz"],
        "convert": "scgestalt_names",
        "clones": "single",
        "description": "scGESTALT zebrafish brain ZF2 (inDrops; barcodes moved into read names by the submitters).",
    },
    "baseline_d21": {
        "construct": "baseline_array",
        "layout": "bulk",
        "whitelist": None,
        "fastq": [f"{ENA}/SRR323/001/SRR32369801/SRR32369801_1.fastq.gz"],
        "n_reads": 5000,
        "clones": "single",
        "description": "BASELINE day-21 piggyBac recorder amplicon, Oxford Nanopore (bulk). Align with minimap2.",
    },
}


def data_dir() -> Path:
    """Cache directory for datasets (``$CASSIOPEIA_DATA`` or ``~/.cache/cassiopeia``)."""
    d = Path(os.environ.get("CASSIOPEIA_DATA", Path.home() / ".cache" / "cassiopeia"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")  # unique per process
    logger.info(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    os.replace(tmp, dest)  # atomic; concurrent downloads of the same file are harmless
    return dest


def whitelist(name: str) -> Path:
    """Path to a 10x cell barcode whitelist (downloaded once)."""
    url = WHITELISTS[name]
    d = data_dir() / "whitelists"
    d.mkdir(parents=True, exist_ok=True)
    return _download(url, d / Path(url).name)


def translation_table(name: str = "3M-february-2018") -> Path | None:
    """Cell Ranger's feature-barcoding translation table, if Cell Ranger is installed.

    Looks in ``$CELLRANGER_BARCODES`` and next to the ``cellranger`` executable.
    """
    candidates = []
    if os.environ.get("CELLRANGER_BARCODES"):
        candidates.append(
            Path(os.environ["CELLRANGER_BARCODES"]) / "translation" / f"{name}.txt.gz"
        )
    exe = shutil.which("cellranger")
    if exe:
        root = Path(exe).resolve().parent
        candidates.append(root / "lib/python/cellranger/barcodes/translation" / f"{name}.txt.gz")
    for c in candidates:
        if c.exists():
            return c
    return None


def _fastq_records(path):
    with gzip.open(path, "rt") as f:
        while True:
            h = f.readline()
            if not h:
                return
            s, p, q = f.readline(), f.readline(), f.readline()
            yield h, s, p, q


def _neighbours(bc: str):
    for i, b in enumerate(bc):
        for x in "ACGTN":
            if x != b:
                yield bc[:i] + x + bc[i + 1 :]


def _subset_10x(
    r1: Path, r2: Path, out1: Path, out2: Path, cb_len: int, n_cells: int, min_reads: int = 200
):
    """Keep all reads from the `n_cells` barcodes with most reads (and 1-mismatch variants)."""
    counts = collections.Counter(rec[1][:cb_len] for rec in _fastq_records(r1))
    top = [bc for bc, n in counts.most_common() if n >= min_reads]
    # sample cells across the barcode rank so the subset has cells of varying depth
    step = max(1, len(top) // n_cells)
    keep = set(top[::step][:n_cells])
    allowed = set(keep)
    for bc in keep:
        allowed.update(_neighbours(bc))
    n = 0
    with gzip.open(out1, "wt", compresslevel=6) as f1, gzip.open(out2, "wt", compresslevel=6) as f2:
        for a, b in zip(_fastq_records(r1), _fastq_records(r2), strict=True):
            if a[1][:cb_len] in allowed:
                f1.writelines(a)
                f2.writelines(b)
                n += 1
    return len(keep), n


def _convert_scgestalt(src: Path, out1: Path, out2: Path, n_cells: int):
    """Split a scGESTALT FASTQ with `BC1-BC2:UMI:...` names into R1 (CB+UMI) / R2 (read)."""

    def parse(h):
        tag = h.split()[1] if len(h.split()) > 1 else h
        bc, umi = tag.split(":")[0].replace("-", ""), tag.split(":")[1]
        return bc, umi

    counts = collections.Counter(parse(rec[0])[0] for rec in _fastq_records(src))
    keep = {bc for bc, _ in counts.most_common(n_cells)}
    n = 0
    with gzip.open(out1, "wt") as f1, gzip.open(out2, "wt") as f2:
        for h, s, p, q in _fastq_records(src):
            bc, umi = parse(h)
            if bc not in keep or len(bc) != 16 or len(umi) != 6:
                continue
            name = h.split()[0]
            f1.write(f"{name}\n{bc}{umi}\n+\n{'I' * 22}\n")
            f2.write(f"{name}\n{s}{p}{q}")
            n += 1
    return len(keep), n


def fetch(name: str, n_cells: int = 300, keep_raw: bool = False) -> dict:
    """Download (once) and subset a dataset.

    Args:
        name: Dataset name (see :data:`DATASETS`).
        n_cells: Number of cells to keep.
        keep_raw: Keep the full downloaded FASTQs.

    Returns:
        Dict with ``fastqs``, ``construct``, ``layout``, ``whitelist`` (path or
        None), ``truth`` (path or None) and ``clones`` (suggested method).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    info = DATASETS[name]
    root = data_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    out = {k: info.get(k) for k in ("construct", "layout", "clones", "description")}
    out["whitelist"] = str(whitelist(info["whitelist"])) if info.get("whitelist") else None
    if info.get("cb_translation"):
        t = translation_table(info["cb_translation"])
        out["cb_translation"] = str(t) if t else None
        if t is None:
            logger.warning(
                f"{name}: Cell Ranger translation table not found; cell barcodes stay in "
                "feature-barcode space (set $CELLRANGER_BARCODES)."
            )
    done = root / f"subset_{n_cells}.done"
    sub = [root / f"{name}_{n_cells}cells_R{i + 1}.fastq.gz" for i in range(2)]
    if not done.exists():
        raw = [_download(u, root / Path(u).name) for u in info["fastq"]]
        if info.get("convert") == "scgestalt_names":
            kept, n = _convert_scgestalt(raw[0], sub[0], sub[1], n_cells)
        elif info.get("layout") == "bulk":
            sub = [root / f"{name}_{info['n_reads']}reads.fastq.gz"]
            n = 0
            with gzip.open(sub[0], "wt") as f:
                for rec in _fastq_records(raw[0]):
                    f.writelines(rec)
                    n += 1
                    if n >= info["n_reads"]:
                        break
            kept = 0
        else:
            from .layout import parse_layout

            lay = parse_layout(info["layout"])
            kept, n = _subset_10x(raw[0], raw[1], sub[0], sub[1], lay.cb_length, n_cells)
        logger.info(f"{name}: kept {kept} cells, {n:,} reads")
        if not keep_raw:
            for r in raw:
                r.unlink(missing_ok=True)
        done.write_text(f"{kept}\t{n}\n")
    if info.get("layout") == "bulk":
        sub = [root / f"{name}_{info['n_reads']}reads.fastq.gz"]
    out["fastqs"] = [str(p) for p in sub]
    out["truth"] = (
        str(_download(info["truth"], root / Path(info["truth"]).name))
        if info.get("truth")
        else None
    )
    return out
