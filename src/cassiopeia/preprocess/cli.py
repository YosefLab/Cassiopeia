"""Command line interface: ``cassiopeia-preprocess``."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

import yaml

from cassiopeia.mixins import PreprocessError


def _version() -> str:
    """Installed Cassiopeia version."""
    try:
        return version("cassiopeia-lineage")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``cassiopeia-preprocess``."""
    p = argparse.ArgumentParser(
        prog="cassiopeia-preprocess",
        description="Cassiopeia lineage tracing preprocessing: reads -> allele table.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    p.add_argument("--debug", action="store_true", help="Show the full traceback on errors")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser(
        "run", help="Run the pipeline from a YAML config and save the filtered allele table."
    )
    r.add_argument("config")

    a = sub.add_parser("align", help="FASTQ -> deduplicated, aligned, CB/UB-tagged BAM.")
    a.add_argument(
        "fastqs", nargs="+", help="FASTQ files in read order (e.g. R1 R2), lanes concatenated"
    )
    a.add_argument("-o", "--output", required=True, help="Output BAM")
    a.add_argument("-c", "--construct", required=True, help="Construct preset or YAML")
    a.add_argument(
        "-l", "--layout", default="10xv3", help="Preset (e.g. 10xv3) or pattern (e.g. 16C12U,T)"
    )
    a.add_argument("-w", "--whitelist")
    a.add_argument("-t", "--threads", type=int)
    a.add_argument("--dedup", default="umi", choices=["umi", "sequence"])
    a.add_argument("--method", default="local", choices=["local", "global"])
    a.add_argument("--strand", default="forward", choices=["forward", "reverse", "both"])
    a.add_argument("--plot", help="Save a QC figure to this path")

    c = sub.add_parser("call-alleles", help="Aligned BAM -> allele table (parquet).")
    c.add_argument("bam")
    c.add_argument("-o", "--output", required=True, help="Output parquet")
    c.add_argument("-c", "--construct", required=True)
    c.add_argument("--cells", help="Cell barcodes to keep (e.g. barcodes.tsv.gz)")
    c.add_argument(
        "--contigs",
        nargs="+",
        default=["auto"],
        help="Contigs to read (default: auto, the construct's); 'all' scans every read",
    )
    c.add_argument("--sample", help="Sample name, stored in a `sample` column")
    c.add_argument("--cb-tag", default="CB", help="Cell barcode tag; '' for bulk data")
    c.add_argument("--umi-tag", default="UB", help="UMI tag; '' for data without UMIs")
    c.add_argument("--min-reads-per-umi", type=int, default=2, help="Drop UMIs with fewer reads")
    c.add_argument(
        "--no-collapse-umis",
        dest="collapse_umis",
        action="store_false",
        help="One row per UMI instead of one row per (cell, intBC, allele)",
    )
    c.add_argument("-t", "--threads", type=int)
    c.add_argument("--plot", help="Save a QC figure to this path")

    b = sub.add_parser("build-reference", help="Write FASTA/GTF and STARsolo/Cell Ranger commands.")
    b.add_argument("construct")
    b.add_argument("out_dir")
    b.add_argument(
        "-l", "--layout", default="10xv3", help="Preset (e.g. 10xv3) or pattern (e.g. 16C12U,T)"
    )
    b.add_argument("-w", "--whitelist")

    sub.add_parser("presets", help="List construct presets.")
    sub.add_parser("example-config", help="Print an example run config.")

    args = p.parse_args(argv)
    try:
        _run(args)
    except (PreprocessError, ValueError, OSError, ImportError, yaml.YAMLError) as e:
        if args.debug:
            raise
        print(f"cassiopeia-preprocess: error: {e}", file=sys.stderr)
        return 1
    return 0


def _run(args: argparse.Namespace) -> None:
    """Run a parsed command."""
    if args.command == "run":
        from .pipeline import run

        run(args.config)
    elif args.command == "align":
        from .align import align

        res = align(
            args.fastqs,
            args.output,
            args.construct,
            layout=args.layout,
            whitelist=args.whitelist,
            dedup=args.dedup,
            method=args.method,
            strand=args.strand,
            n_threads=args.threads,
            plot=False,
            save=args.plot,
        )
        yaml.safe_dump(res["stats"], sys.stdout, sort_keys=False)
    elif args.command == "call-alleles":
        from .alleles import call_alleles

        contigs = {"auto": "auto", "all": None}.get(" ".join(args.contigs), args.contigs)
        df = call_alleles(
            args.bam,
            args.construct,
            cells=args.cells,
            contigs=contigs,
            sample=args.sample,
            cb_tag=args.cb_tag or None,
            umi_tag=args.umi_tag or None,
            min_reads_per_umi=args.min_reads_per_umi,
            collapse_umis=args.collapse_umis,
            n_threads=args.threads,
            plot=False,
            save=args.plot,
        )
        df.to_parquet(args.output)
    elif args.command == "build-reference":
        from .reference import build_reference

        res = build_reference(
            args.construct, args.out_dir, layout=args.layout, whitelist=args.whitelist
        )
        for k, v in res.items():
            if v is not None:
                print(f"{k}: {v}")
    elif args.command == "presets":
        from .construct import list_presets, load_construct

        for name in list_presets():
            c = load_construct(name)
            print(f"{name:12s} {c!r}\n{'':12s} {c.description}")
    elif args.command == "example-config":
        from .pipeline import EXAMPLE_CONFIG

        print(EXAMPLE_CONFIG)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
