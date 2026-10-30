"""Run the full preprocessing pipeline from a configuration."""

from __future__ import annotations

import copy
import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cassiopeia.mixins import PreprocessError, logger

from .align import align
from .alleles import call_alleles
from .clones import call_clones
from .construct import Construct, load_construct
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
    "filter_doublets",
    "filter_integrations",
    "filter_cells_coverage",
]

#: The function run by each step.
_STEP_FUNCTIONS: dict[str, Callable] = {
    "align": align,
    "call_alleles": call_alleles,
    "correct_intbcs": correct_intbcs,
    "filter_alleles": filter_alleles,
    "filter_cells": filter_cells,
    "call_clones": call_clones,
    "filter_doublets": filter_doublets,
    "filter_integrations": filter_integrations,
    "filter_cells_coverage": filter_cells,
}

DEFAULTS: dict[str, Any] = {
    "correct_intbcs": {},
    "filter_alleles": {},
    "filter_cells": {"min_umi": 10},
    # only for data with several clones
    "call_clones": False,
    "filter_doublets": {},
    "filter_integrations": {},
    "filter_cells_coverage": {"min_intbc_frac": 0.5},
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
call_clones: false                   # {method: nova_clone} for data with several clones
filter_doublets: {}                  # before filter_integrations, so doublets do not inflate conflicts
filter_integrations: {min_detection: 0.2, max_conflict: 0.2}
filter_cells_coverage: {min_intbc_frac: 0.5}   # optional
# The result is <dir>/<name>.alleles.parquet. Convert it to characters in Python:
#   df = cas.pp.read_alleles("results/sample1/sample1.alleles.parquet")
#   tdata = cas.pp.alleles_to_characters(df)
"""


def load_config(config: str | Path | dict) -> dict:
    """Load and check a run configuration from YAML or a dict."""
    if isinstance(config, dict):
        cfg = copy.deepcopy(config)
    else:
        path = Path(config)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise PreprocessError(f"The config {str(path)!r} is not a YAML mapping.")
        if isinstance(cfg.get("construct"), str):
            cpath = path.parent / cfg["construct"]
            if cpath.suffix in (".yaml", ".yml") and cpath.exists():
                cfg["construct"] = str(cpath)
    if "construct" not in cfg:
        raise PreprocessError("The config needs a `construct`.")
    inp = cfg.get("input") or {}
    if not inp.get("fastq") and not inp.get("bam"):
        raise PreprocessError("The config `input` needs `fastq` or `bam`.")
    if inp.get("fastq") and inp.get("bam"):
        raise PreprocessError("The config `input` needs `fastq` or `bam`, not both.")
    unknown = set(cfg) - {"construct", "input", "output", "threads", *STEPS}
    if "alleles_to_characters" in unknown:
        raise PreprocessError(
            "The pipeline ends at the allele table: remove `alleles_to_characters` from the "
            "config and call pp.alleles_to_characters on the saved table."
        )
    if unknown:
        raise PreprocessError(f"Unknown config keys: {sorted(unknown)}")
    if cfg.get("threads") is not None:
        try:
            cfg["threads"] = int(cfg["threads"])
        except (TypeError, ValueError):
            raise PreprocessError(
                f"`threads` must be an integer, not {cfg['threads']!r}."
            ) from None
    if inp.get("fastq") and cfg.get("align") is False:
        raise PreprocessError(
            "`align` cannot be skipped for FASTQ input; give `input: {bam: ...}` to start "
            "from an aligned BAM."
        )
    for step in STEPS:
        _check_step_options(step, cfg.get(step))
    return cfg


def _check_step_options(step: str, val: Any) -> None:
    """Check that a step's config value is a mapping of its function's arguments (or a bool)."""
    if val is None or isinstance(val, bool):
        return
    if not isinstance(val, dict):
        raise PreprocessError(
            f"`{step}` must be a mapping of options, or false to skip the step, not {val!r}."
        )
    params = inspect.signature(_STEP_FUNCTIONS[step]).parameters
    # set by the pipeline from `construct` and `input`
    fixed = {"fastqs", "construct"} if step == "align" else {"bam", "construct", "df"}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        bad = [k for k in val if k in fixed]
    else:
        bad = [k for k in val if k not in params or k in fixed]
    if bad:
        valid = sorted(
            k for k in params if k not in fixed and params[k].kind != params[k].VAR_KEYWORD
        )
        raise PreprocessError(f"Unknown options for `{step}`: {bad}. Options: {', '.join(valid)}.")


def _opts(cfg: dict, step: str) -> dict | None:
    val = cfg[step] if step in cfg else DEFAULTS.get(step, {})
    if val is False:
        return None
    opts = dict(DEFAULTS.get(step) or {})
    if isinstance(val, dict):
        opts.update(val)
    return opts


def _yaml_safe(obj: Any) -> Any:
    """The config with paths, constructs and tuples converted for ``yaml.safe_dump``."""
    if isinstance(obj, Construct):
        return _yaml_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_yaml_safe(v) for v in obj]
    if obj is None or type(obj) in (str, int, float, bool):
        return obj
    if hasattr(obj, "item"):  # numpy scalars
        return _yaml_safe(obj.item())
    return str(obj)  # paths and other objects


def _call(fn: Callable, *args, opts: dict, threads: int | None, save: str | None):
    """Run a step; its config options override the pipeline's threads and plot settings."""
    kwargs: dict[str, Any] = {"plot": False, "save": save}
    if "n_threads" in inspect.signature(fn).parameters:
        kwargs["n_threads"] = threads
    kwargs.update(opts)
    return fn(*args, **kwargs)


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
    7. :func:`~cassiopeia.pp.filter_doublets`, before
       ``filter_integrations`` so that doublets do not inflate its conflict
       rates.
    8. :func:`~cassiopeia.pp.filter_integrations`
    9. ``filter_cells_coverage``: :func:`~cassiopeia.pp.filter_cells` again,
       with ``min_intbc_frac=0.5`` by default. Runs only after
       ``filter_integrations`` or ``call_clones``.

    Filters annotate the table (``pass_*`` columns) rather than drop rows, so
    the saved table keeps every allele and can be filtered again.

    Each run processes one sample. To analyze several samples together, run
    the pipeline per sample and combine the character matrices (see
    :func:`~cassiopeia.pp.call_alleles`); concatenated allele tables do not
    work with the steps.

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
      ``false`` to skip it (``align`` cannot be skipped for FASTQ input).
      Steps left out run with their defaults. ``n_threads``, ``plot`` and
      ``save`` given for a step override ``threads`` and ``output.plots``.

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

    def save(step: str) -> str | None:
        return str(qc_dir / f"{step}.png") if plots else None

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(_yaml_safe(cfg), f, sort_keys=False)
    logger.info(f"Running Cassiopeia preprocessing for {name!r} ({construct!r}) -> {out_dir}")

    if inp.get("fastq"):
        opts = _opts(cfg, "align") or {}
        opts.setdefault("layout", inp.get("layout", "10xv3"))
        if inp.get("whitelist"):
            opts.setdefault("whitelist", inp["whitelist"])
        bam = opts.pop("output_bam", str(out_dir / f"{name}.aligned.bam"))
        _call(align, inp["fastq"], bam, construct, opts=opts, threads=threads, save=save("align"))
        lay = parse_layout(opts["layout"])
        ca_defaults = {"cb_tag": "CB" if lay.cb else None, "umi_tag": "UB" if lay.umi else None}
    else:
        if cfg.get("align"):
            logger.warning("Ignoring the `align` options: the input is a BAM.")
        bam = inp["bam"]
        ca_defaults = {}
    ca = dict(ca_defaults)
    ca.update(_opts(cfg, "call_alleles") or {})
    if inp.get("cells"):
        ca.setdefault("cells", inp["cells"])
    if inp.get("contigs"):
        ca.setdefault("contigs", inp["contigs"])
    df = _call(call_alleles, bam, construct, opts=ca, threads=threads, save=save("call_alleles"))

    for step in STEPS[2:]:
        o = _opts(cfg, step)
        if o is None:
            continue
        if step == "correct_intbcs" and construct.intbc is None:
            continue
        expected = df.attrs.get("integrations") or df.attrs.get("clone_intbcs")
        if step == "filter_cells_coverage" and not expected:
            continue
        df = _call(_STEP_FUNCTIONS[step], df, opts=o, threads=threads, save=save(step))
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
    """Read an allele table saved with ``df.to_parquet`` or by :func:`~cassiopeia.pp.run`.

    Args:
        path: Parquet file.

    Returns:
        The allele table, with its categorical columns and ``df.attrs``
        (construct, thresholds, read statistics) restored.
    """
    return pd.read_parquet(path)
