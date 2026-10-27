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
from .characters import alleles_to_characters
from .clones import call_clones
from .construct import load_construct
from .correct import correct_intbcs
from .doublets import filter_doublets
from .filter import filter_cells, filter_molecules
from .integrations import filter_integrations
from .layout import parse_layout

#: Steps run by :func:`run`, in order. Each can be configured (keyword
#: arguments) or disabled (``false``) in the config.
STEPS = [
    "align",
    "call_alleles",
    "correct_intbcs",
    "filter_molecules",
    "filter_cells",
    "call_clones",
    "filter_integrations",
    "filter_cells_coverage",
    "filter_doublets",
    "alleles_to_characters",
]

DEFAULTS: dict[str, Any] = {
    "correct_intbcs": {},
    "filter_molecules": {},
    "filter_cells": {"min_umi": 10},
    "call_clones": {},
    "filter_integrations": {},
    "filter_cells_coverage": {"min_intbc_frac": 0.5},
    "filter_doublets": {},
    "alleles_to_characters": {},
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
  checkpoint: true             # save the molecule table as parquet
threads: 8
# Step options (keyword arguments of each pp function); `false` skips a step.
align: {}
call_alleles: {}
correct_intbcs: {}
filter_molecules: {min_reads: 2}
filter_cells: {min_umi: 10}
call_clones: {method: nova_clone}     # use `single` when all cells share intBCs
filter_integrations: {min_detection: 0.4, max_conflict: 0.2}
filter_cells_coverage: {min_intbc_frac: 0.5}
filter_doublets: {}
alleles_to_characters: {}
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
    if unknown:
        raise PreprocessError(f"Unknown config keys: {sorted(unknown)}")
    return cfg


def _opts(cfg: dict, step: str) -> dict | None:
    val = cfg.get(step, DEFAULTS.get(step, {}))
    if val is False:
        return None
    opts = dict(DEFAULTS.get(step, {}))
    opts.update(val or {})
    return opts


def run(config: str | Path | dict):
    """Run the preprocessing pipeline.

    FASTQ input goes through :func:`cassiopeia.pp.align`; BAM input (STARsolo,
    Cell Ranger) goes straight to :func:`cassiopeia.pp.call_alleles`. Then the
    molecule table is corrected, filtered, assigned to clones, screened for
    doublets and converted to a character matrix. See
    ``cassiopeia.preprocess.pipeline.EXAMPLE_CONFIG`` for the format.

    Args:
        config: YAML path or dict.

    Returns:
        TreeData with the character matrix (also written to
        ``<output.dir>/<output.name>.h5td``).
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
        return {
            "plot": False,
            "show": False,
            "save": str(qc_dir / f"{step}.png") if plots else None,
        }

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
    df = call_alleles(bam, construct, n_threads=threads, **ca)
    if out_cfg.get("checkpoint", False):
        df.to_parquet(out_dir / f"{name}.molecules.raw.parquet")

    if (o := _opts(cfg, "correct_intbcs")) is not None and construct.intbc is not None:
        df = correct_intbcs(df, n_threads=threads, **o, **plot_args("correct_intbcs"))
    if (o := _opts(cfg, "filter_molecules")) is not None:
        df = filter_molecules(df, **o, **plot_args("filter_molecules"))
    if (o := _opts(cfg, "filter_cells")) is not None:
        df = filter_cells(df, **o, **plot_args("filter_cells"))
    if (o := _opts(cfg, "call_clones")) is not None:
        o.setdefault("method", construct.extra.get("defaults", {}).get("clones", "nova_clone"))
        df = call_clones(df, **o, **plot_args("call_clones"))
    if (o := _opts(cfg, "filter_integrations")) is not None:
        df = filter_integrations(df, **o, **plot_args("filter_integrations"))
    if (o := _opts(cfg, "filter_cells_coverage")) is not None and "clone" in df.columns:
        df = filter_cells(df, **o, **plot_args("filter_cells_coverage"))
    if (o := _opts(cfg, "filter_doublets")) is not None:
        df = filter_doublets(df, **o, **plot_args("filter_doublets"))
    if out_cfg.get("checkpoint", False):
        df.to_parquet(out_dir / f"{name}.molecules.parquet")
    o = _opts(cfg, "alleles_to_characters") or {}
    tdata = alleles_to_characters(df, **o)
    tdata.uns["preprocess"]["config"] = yaml.safe_dump(cfg, sort_keys=False)
    tdata.write_h5td(out_dir / f"{name}.h5td")
    from .qc import cell_metrics

    cell_metrics(df).to_csv(out_dir / f"{name}.cells.csv")
    logger.info(
        f"Done in {time.time() - t0:.1f}s: {tdata.n_obs:,} cells -> {out_dir / f'{name}.h5td'}"
    )
    return tdata


def read_molecules(path: str | Path) -> pd.DataFrame:
    """Read a molecule table saved with ``df.to_parquet`` (attrs are restored)."""
    return pd.read_parquet(path)
