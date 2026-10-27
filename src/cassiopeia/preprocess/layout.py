"""Read layouts (chemistries): where the cell barcode, UMI and cDNA are."""

from __future__ import annotations

from dataclasses import dataclass, field

# A segment is (read index, start, end); end None = to the end of the read.
Segment = tuple[int, int, int | None]


@dataclass
class ReadLayout:
    """Positions of the cell barcode, UMI and target-site sequence in the reads.

    Args:
        name: Layout name.
        n_reads: Number of FASTQ files per lane (e.g. 2 for R1/R2).
        cb: Cell barcode segments (concatenated in order).
        umi: UMI segments (concatenated in order).
        cdna: Segment holding the target-site sequence.
        whitelist_hint: Name of the usual barcode whitelist, for messages.
    """

    name: str
    n_reads: int
    cb: list[Segment] = field(default_factory=list)
    umi: list[Segment] = field(default_factory=list)
    cdna: Segment = (1, 0, None)
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


_TENX_V3_WL = "3M-february-2018.txt.gz (10x 3' v3/v3.1)"

LAYOUTS: dict[str, ReadLayout] = {
    "10xv2": ReadLayout(
        "10xv2", 2, [(0, 0, 16)], [(0, 16, 26)], (1, 0, None), "737K-august-2016.txt"
    ),
    "10xv3": ReadLayout("10xv3", 2, [(0, 0, 16)], [(0, 16, 28)], (1, 0, None), _TENX_V3_WL),
    "10xv3.1": ReadLayout("10xv3.1", 2, [(0, 0, 16)], [(0, 16, 28)], (1, 0, None), _TENX_V3_WL),
    "10xv4": ReadLayout(
        "10xv4",
        2,
        [(0, 0, 16)],
        [(0, 16, 28)],
        (1, 0, None),
        "3M-3pgex-may-2023.txt.gz (GEM-X 3' v4)",
    ),
    "10x5p": ReadLayout(
        "10x5p", 2, [(0, 0, 16)], [(0, 16, 26)], (1, 0, None), "737K-august-2016.txt"
    ),
    "dropseq": ReadLayout("dropseq", 2, [(0, 0, 12)], [(0, 12, 20)], (1, 0, None)),
    "indropsv3": ReadLayout("indropsv3", 3, [(0, 0, 8), (1, 0, 8)], [(1, 8, 14)], (2, 0, None)),
    "slideseq2": ReadLayout("slideseq2", 2, [(0, 0, 8), (0, 26, 32)], [(0, 32, 41)], (1, 0, None)),
    "bulk": ReadLayout("bulk", 1, [], [], (0, 0, None)),
    "bulk_umi10": ReadLayout("bulk_umi10", 2, [], [(0, 0, 10)], (1, 0, None)),
}


def get_layout(layout: str | ReadLayout | dict) -> ReadLayout:
    """Get a :class:`ReadLayout` from a name, dict or layout."""
    if isinstance(layout, ReadLayout):
        return layout
    if isinstance(layout, dict):
        d = dict(layout)
        d.setdefault("name", "custom")
        d["cb"] = [tuple(s) for s in d.get("cb", [])]
        d["umi"] = [tuple(s) for s in d.get("umi", [])]
        d["cdna"] = tuple(d.get("cdna", (1, 0, None)))
        return ReadLayout(**d)
    key = str(layout).lower()
    if key not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}. Available: {', '.join(LAYOUTS)}.")
    return LAYOUTS[key]
