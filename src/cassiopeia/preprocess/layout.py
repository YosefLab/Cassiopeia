"""Read layouts: where the cell barcode, UMI and target-site sequence are in the reads.

A layout is a preset name or a pattern string with one comma-separated token
per FASTQ file, in read order. Each token lists fixed-length segments:

* ``<n>C``: cell barcode bases (segments are joined in order)
* ``<n>U``: UMI bases (joined in order)
* ``<n>X``: bases to skip
* ``T``: the target-site sequence (the rest of the read)

A token ``X`` ignores the whole read, and exactly one ``T`` is required. For
example ``"16C12U,T"`` is 10x 3' v3 (R1: 16 bp cell barcode + 12 bp UMI, R2:
target site), ``"8C18X6C9U,T"`` Slide-seq v2, ``"8C,8C6U,T"`` inDrops v3 and
``"T"`` a bulk amplicon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A segment is (read index, start, end); end None = to the end of the read.
Segment = tuple[int, int, int | None]

_TENX_V3_WL = "3M-february-2018.txt.gz (10x 3' v3/v3.1)"
# preset name -> (pattern, usual whitelist)
PRESETS: dict[str, tuple[str, str]] = {
    "10xv2": ("16C10U,T", "737K-august-2016.txt"),
    "10xv3": ("16C12U,T", _TENX_V3_WL),
    "10xv3.1": ("16C12U,T", _TENX_V3_WL),
    "10xv4": ("16C12U,T", "3M-3pgex-may-2023.txt.gz (GEM-X 3' v4)"),
    "10x5p": ("16C10U,T", "737K-august-2016.txt"),
    "dropseq": ("12C8U,T", ""),
    "indropsv3": ("8C,8C6U,T", ""),
    "slideseq2": ("8C18X6C9U,T", ""),
    "bulk": ("T", ""),
    "bulk_umi10": ("10U,T", ""),
}

_TOKEN = re.compile(r"(?:\d+[CUX])*T?")
_UNIT = re.compile(r"(\d+)([CUX])|T")


@dataclass(frozen=True)
class Layout:
    """A parsed read layout (see the module docstring)."""

    pattern: str
    n_reads: int
    cb: tuple[Segment, ...]
    umi: tuple[Segment, ...]
    cdna: Segment
    whitelist_hint: str = ""

    @property
    def cb_length(self) -> int:
        """Cell barcode length."""
        return sum(e - s for _, s, e in self.cb)

    @property
    def umi_length(self) -> int:
        """UMI length."""
        return sum(e - s for _, s, e in self.umi)

    def _native(self) -> dict:
        return {
            "n_reads": self.n_reads,
            "cb": list(self.cb),
            "umi": list(self.umi),
            "cdna": self.cdna,
        }


def parse_layout(layout: str) -> Layout:
    """Parse a layout preset name (e.g. ``"10xv3"``) or pattern (e.g. ``"16C12U,T"``)."""
    if not isinstance(layout, str):
        raise TypeError(
            f"layout must be a preset name or pattern string, not {type(layout).__name__}"
        )
    pattern, hint = PRESETS.get(layout.strip().lower(), (layout, ""))
    tokens = [t.strip() for t in pattern.upper().split(",")]

    def invalid(reason):
        return ValueError(
            f"Invalid layout {layout!r}: {reason}. Use a preset ({', '.join(PRESETS)}) or a "
            "pattern with one token per FASTQ, e.g. '16C12U,T' (C cell barcode, U UMI, "
            "X skip, T target site)."
        )

    cb, umi, cdna = [], [], None
    for r, tok in enumerate(tokens):
        if tok == "X":
            continue
        if not tok or not _TOKEN.fullmatch(tok):
            raise invalid(f"cannot parse {tok!r}")
        pos = 0
        for m in _UNIT.finditer(tok):
            if m.group(0) == "T":
                if cdna is not None:
                    raise invalid("more than one T")
                cdna = (r, pos, None)
                continue
            n = int(m.group(1))
            if n == 0:
                raise invalid("segment lengths must be positive")
            if m.group(2) == "C":
                cb.append((r, pos, pos + n))
            elif m.group(2) == "U":
                umi.append((r, pos, pos + n))
            pos += n
    if cdna is None:
        raise invalid("no T (target-site sequence)")
    return Layout(pattern.upper().replace(" ", ""), len(tokens), tuple(cb), tuple(umi), cdna, hint)
