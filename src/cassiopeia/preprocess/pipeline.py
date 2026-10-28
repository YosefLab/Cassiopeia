"""Run the full preprocessing pipeline from a configuration."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cassiopeia.mixins import PreprocessError, logger

from .align import align
from .alleles import call_alleles
from .clones import call_clones
from .construct import load_construct
from .correct import correct_intbcs
from .doublets import filter_doublets
from .filter import filter_alleles, filter_cells
from .integrations import filter_integrations
from .layout import parse_layout
from .qc import cell_metrics

#: Steps run by :func:`run`, in order. Each can be configured (keyword
#: arguments) or disabled (``false``) in the config.
STEPS = [
    "align",
    "call_alleles",
    "correct_intbcs",
    "filter_alleles",
    "filter_cells",
    "call_clones",
    "filter_integrations",
    "filter_cells_coverage",
    "filter_doublets",
]

DEFAULTS: dict[str, Any] = {
    "correct_intbcs": {},
    "filter_alleles": {},
    "filter_cells": {"min_umi": 10},
    # only for data with several clones
    "call_clones": False,
    "filter_integrations": {},
    "filter_cells_coverage": {"min_intbc_frac": 0.5},
    "filter_doublets": {},
}

EXAMPLE_CONFIG = """\
# cassiopeia-preprocess run config
construct: petracer            # preset name, YAML path, or inline mapping
input:
  fastq: [sample_R1.fastq.gz, sample_R2.fastq.gz]   # -> pp.align
  layout: 10xv3                 # preset or pattern, e.g. 16C12U,T
  whitelist: 3M-february-2018.txt.gz
  # or, for STARsolo / Cell Ranger output:
  # bam: possorted_genome_bam.bam
  # cells: filtered_feature_bc_matrix/barcodes.tsv.gz
output:
  dir: results/sample1
  name: sample1
  plots: true                  # save one QC figure per step to <dir>/qc/
threads: 8
# Step options (keyword arguments of each pp function); `false` skips a step.
align: {}
call_alleles: {min_reads_per_umi: 2}   # one row per (cell, intBC, allele); adjust to depth
correct_intbcs: {}
filter_alleles: {min_allele_frac: 0.5, min_umi_per_allele: 2}   # adjust to sequencing depth
filter_cells: {min_umi: 10}
call_clones: {method: nova_clone}     # only for data with several clones (default: skipped)
filter_integrations: {min_detection: 0.2, max_conflict: 0.2}
filter_cells_coverage: {min_intbc_frac: 0.5}   # optional
filter_doublets: {}
# The result is <dir>/<name>.alleles.parquet. Convert it to characters in Python:
#   df = cas.pp.read_alleles("results/sample1/sample1.alleles.parquet")
#   tdata = cas.pp.alleles_to_characters(df)
"""


def load_config(config: str | Path | dict) -> dict:
    """Load a run configuration from YAML or a dict."""
    if isinstance(config, dict):
        cfg = copy.deepcopy(config)
        base = Path(".")
    else:
        path = Path(config)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        base = path.parent
        if isinstance(cfg.get("construct"), str):
            cpath = base / cfg["construct"]
            if cpath.suffix in (".yaml", ".yml") and cpath.exists():
                cfg["construct"] = str(cpath)
    if "construct" not in cfg:
        raise PreprocessError("The config needs a `construct`.")
    inp = cfg.get("input") or {}
    if not inp.get("fastq") and not inp.get("bam"):
        raise PreprocessError("The config `input` needs `fastq` or `bam`.")
    unknown = set(cfg) - {"construct", "input", "output", "threads", *STEPS}
    if "alleles_to_characters" in unknown:
        raise PreprocessError(
            "The pipeline ends at the allele table: remove `alleles_to_characters` from the "
            "config and call pp.alleles_to_characters on the saved table."
        )
    if unknown:
        raise PreprocessError(f"Unknown config keys: {sorted(unknown)}")
    return cfg


def _opts(cfg: dict, step: str) -> dict | None:
    val = cfg[step] if step in cfg else DEFAULTS.get(step, {})
    if val is False:
        return None
    opts = dict(DEFAULTS.get(step) or {})
    opts.update(val or {})
    return opts


def run(config: str | Path | dict) -> pd.DataFrame:
    """Run the preprocessing pipeline from a config, from reads to a filtered allele table.

    The pipeline ends at the allele table, saved as ``<name>.alleles.parquet``.
    Convert it to a character matrix with
    :func:`~cassiopeia.pp.alleles_to_characters`::

        df = cas.pp.read_alleles("results/sample1.alleles.parquet")
        tdata = cas.pp.alleles_to_characters(df)

    The steps run in this order; each is the ``pp`` function of the same name:

    1. :func:`~cassiopeia.pp.align`, for FASTQ input only. BAM input
       (STARsolo, Cell Ranger) starts at the next step.
    2. :func:`~cassiopeia.pp.call_alleles`
    3. :func:`~cassiopeia.pp.correct_intbcs`, skipped for constructs without
       an intBC.
    4. :func:`~cassiopeia.pp.filter_alleles`
    5. :func:`~cassiopeia.pp.filter_cells`, with ``min_umi=10`` by default.
    6. :func:`~cassiopeia.pp.call_clones`, skipped unless it is in the config
       (only for data with several clones).
    7. :func:`~cassiopeia.pp.filter_integrations`
    8. ``filter_cells_coverage``: :func:`~cassiopeia.pp.filter_cells` again,
       with ``min_intbc_frac=0.5`` by default. Runs only after
       ``filter_integrations`` or ``call_clones``.
    9. :func:`~cassiopeia.pp.filter_doublets`

    Filters annotate the table (``pass_*`` columns) rather than drop rows, so
    the saved table keeps every allele and can be filtered again.

    The config has these keys:

    * ``construct``: a preset name, a construct YAML (relative to the config
      file) or an inline construct.
    * ``input``: either ``fastq`` (the FASTQ files in read order, e.g.
      ``[R1.fastq.gz, R2.fastq.gz]``) with ``layout`` (default ``10xv3``) and
      an optional cell barcode ``whitelist``; or ``bam`` with optional
      ``cells`` (barcodes to keep, e.g. ``barcodes.tsv.gz``) and ``contigs``.
    * ``output``: ``dir`` (default ``cassiopeia_preprocess``), ``name``
      (default: the construct name) and ``plots`` (save QC figures, default
      ``true``).
    * ``threads``: default is the Slurm allocation, else up to 8.
    * One key per step, holding keyword arguments of its function, or
      ``false`` to skip it. Steps left out run with their defaults.

    For example:

    .. code-block:: yaml

        construct: weissman
        input: {fastq: [R1.fastq.gz, R2.fastq.gz], layout: 10xv3}
        output: {dir: results, name: sample1}
        filter_alleles: {min_umi_per_allele: 2, min_avg_reads_per_umi: 5}
        filter_doublets: false

    ``cassiopeia-preprocess example-config`` prints a config with every step.
    Input paths are relative to the working directory.

    Files written to ``output.dir`` (``<name>`` is ``output.name``):

    * ``<name>.alleles.parquet``: the returned allele table; read it with
      :func:`~cassiopeia.pp.read_alleles`.
    * ``<name>.cells.csv``: :func:`~cassiopeia.pp.cell_metrics` for every cell,
      including filtered ones.
    * ``config.yaml``: the config.
    * ``qc/<step>.png``: one QC figure per step, with ``plots``.
    * ``<name>.aligned.bam``: for FASTQ input (set another path with
      ``align: {output_bam: ...}``).

    Args:
        config: Path to a YAML config, or the config as a dict.

    Returns:
        The allele table, with a ``pass_*`` column per filter.
    """
    t0 = time.time()
    cfg = load_config(config)
    construct = load_construct(cfg["construct"])
    out_cfg = cfg.get("output") or {}
    out_dir = Path(out_cfg.get("dir", "cassiopeia_preprocess"))
    name = out_cfg.get("name", construct.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    plots = bool(out_cfg.get("plots", True))
    if plots:
        qc_dir.mkdir(exist_ok=True)
    threads = cfg.get("threads")
    inp = cfg["input"]

    def plot_args(step):
        return {"plot": False, "save": str(qc_dir / f"{step}.png") if plots else None}

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"Running Cassiopeia preprocessing for {name!r} ({construct!r}) -> {out_dir}")

    if inp.get("fastq"):
        opts = _opts(cfg, "align") or {}
        opts.setdefault("layout", inp.get("layout", "10xv3"))
        if inp.get("whitelist"):
            opts.setdefault("whitelist", inp["whitelist"])
        bam = opts.pop("output_bam", str(out_dir / f"{name}.aligned.bam"))
        align(inp["fastq"], bam, construct, n_threads=threads, **opts, **plot_args("align"))
        lay = parse_layout(opts["layout"])
        ca_defaults = {"cb_tag": "CB" if lay.cb else None, "umi_tag": "UB" if lay.umi else None}
    else:
        bam = inp["bam"]
        ca_defaults = {}
    ca = dict(ca_defaults)
    ca.update(_opts(cfg, "call_alleles") or {})
    if inp.get("cells"):
        ca.setdefault("cells", inp["cells"])
    if inp.get("contigs"):
        ca.setdefault("contigs", inp["contigs"])
    df = call_alleles(bam, construct, n_threads=threads, **ca, **plot_args("call_alleles"))

    if (o := _opts(cfg, "correct_intbcs")) is not None and construct.intbc is not None:
        df = correct_intbcs(df, n_threads=threads, **o, **plot_args("correct_intbcs"))
    if (o := _opts(cfg, "filter_alleles")) is not None:
        df = filter_alleles(df, **o, **plot_args("filter_alleles"))
    if (o := _opts(cfg, "filter_cells")) is not None:
        df = filter_cells(df, **o, **plot_args("filter_cells"))
    if (o := _opts(cfg, "call_clones")) is not None:
        df = call_clones(df, **o, **plot_args("call_clones"))
    if (o := _opts(cfg, "filter_integrations")) is not None:
        df = filter_integrations(df, **o, **plot_args("filter_integrations"))
    expected = df.attrs.get("integrations") or df.attrs.get("clone_intbcs")
    if (o := _opts(cfg, "filter_cells_coverage")) is not None and expected:
        df = filter_cells(df, **o, **plot_args("filter_cells_coverage"))
    if (o := _opts(cfg, "filter_doublets")) is not None:
        df = filter_doublets(df, **o, **plot_args("filter_doublets"))
    path = out_dir / f"{name}.alleles.parquet"
    df.to_parquet(path)
    cells = cell_metrics(df)
    cells.to_csv(out_dir / f"{name}.cells.csv")
    logger.info(
        f"Done in {time.time() - t0:.1f}s: {int(cells['passed_qc'].sum()):,}/{len(cells):,} "
        f"cells pass -> {path}"
    )
    return df


def read_alleles(path: str | Path) -> pd.DataFrame:
    """Read an allele table saved with ``df.to_parquet`` (attrs are restored)."""
    return pd.read_parquet(path)
