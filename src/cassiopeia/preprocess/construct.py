"""Declarative description of lineage tracing target sites.

A :class:`Construct` describes the target-site reference sequence as it
appears in the reads, where the integration barcode (intBC) is, and the edit
sites with the kind of edits each one records. The same description drives
reference building (:func:`cassiopeia.pp.build_reference`), alignment
(:func:`cassiopeia.pp.align`) and allele calling
(:func:`cassiopeia.pp.call_alleles`).

Constructs can be created in Python, loaded from YAML, or taken from the
bundled presets (see :func:`list_presets`).
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

EditModel = Literal["indel", "insertion_lookup", "sequential_insertion", "substitution"]
_MODEL_IDS = {"indel": 0, "insertion_lookup": 1, "sequential_insertion": 2, "substitution": 3}
_DEFAULT_WINDOWS = {
    "indel": (12, 12),
    "insertion_lookup": (2, 2),
    "sequential_insertion": (0, 0),
    "substitution": (0, 0),
}


@dataclass
class Site:
    """An edit site in the target-site reference.

    Args:
        name: Site name; used as the column name in the molecule table.
        position: 0-based reference coordinate of the cut / insertion point.
        edit_model: Kind of edits recorded at this site:

            * ``"indel"`` - Cas9 indels (Cassiopeia, KP-tracer, GESTALT).
              Alleles are reported as legacy indel strings, e.g.
              ``"ATCGA[112:3D]TTAGC"`` or ``"ATCGA[None]TTAGC"`` if unedited.
            * ``"insertion_lookup"`` - prime-editing insertions of known
              sequences (PEtracer). Insertions found in ``lookup`` are
              reported by their label; others by their sequence.
            * ``"sequential_insertion"`` - one slot of a sequentially edited
              tandem array (DNA Typewriter); see :class:`TapeArray`.
            * ``"substitution"`` - base editing (BASELINE): a string with one
              character per ``from_base`` in ``[start, end)``: ``1`` edited,
              ``0`` unedited, ``N`` unknown, ``D`` deleted.
        window: Bases to the left and right of ``position`` in which edits
            are attributed to this site.
        context: (indel) Include sequence context in the allele string.
        context_size: (indel) Number of context bases.
        lookup: (insertion_lookup) Inserted sequence to state label.
        start: (substitution) Window start (defaults to ``position``).
        end: (substitution) Window end (exclusive).
        from_base: (substitution) Reference base that gets edited.
        to_base: (substitution) Edited base.
        min_base_qual: (substitution) Minimum base quality to call a base.
        slot: (sequential_insertion) 0-based slot index in the array.
    """

    name: str
    position: int = 0
    edit_model: EditModel = "indel"
    window: tuple[int, int] | None = None
    context: bool = True
    context_size: int = 5
    lookup: dict[str, str] | None = None
    start: int | None = None
    end: int | None = None
    from_base: str = "A"
    to_base: str = "G"
    min_base_qual: int = 0
    slot: int | None = None

    def __post_init__(self):
        if self.edit_model not in _MODEL_IDS:
            raise ValueError(f"Unknown edit_model {self.edit_model!r} for site {self.name!r}.")
        if self.window is None:
            self.window = _DEFAULT_WINDOWS[self.edit_model]
        self.window = (int(self.window[0]), int(self.window[1]))
        if self.lookup is not None:
            self.lookup = {str(k).upper(): str(v) for k, v in self.lookup.items()}

    @property
    def edited_states(self) -> list[str] | None:
        """Known edited states (labels) for insertion_lookup sites."""
        if self.lookup is None:
            return None
        return sorted({v for v in self.lookup.values() if v != "None"}, key=lambda x: (len(x), x))


@dataclass
class TapeArray:
    """A tandem array of identical units edited sequentially (DNA Typewriter).

    Each unit is preceded by an optional insertion of ``insert_len`` bases.
    Arrays may be shortened by deletions, so the array is parsed directly
    from the read and slots beyond the observed array length are reported as
    missing.

    Args:
        start: Reference coordinate where the first slot (insertion point +
            first unit) begins.
        unit: Sequence of one unedited unit.
        insert_len: Length of each insertion.
        n_units: Maximum number of slots.
        end_flank: Sequence directly after the last unit (marks the end of
            the array; allows detecting truncated arrays).
        max_mismatch: Mismatches allowed when matching units / flank.
        anchor: Reference coordinate (<= ``start``) where parsing starts; the
            bases ``[anchor, start)`` (e.g. the first monomer) are read from the
            sequence, so aligner placement of indels in the repeat does not matter.
    """

    start: int
    unit: str
    insert_len: int
    n_units: int
    end_flank: str = ""
    max_mismatch: int = 1
    anchor: int | None = None


@dataclass
class Construct:
    r"""Target-site construct description.

    Args:
        name: Construct name; also the contig name used by
            :func:`cassiopeia.pp.build_reference` and :func:`cassiopeia.pp.align`.
        reference: Target-site reference sequence with the intBC masked as N.
        sites: Edit sites.
        intbc: ``(start, end)`` of the intBC in ``reference``; ``"auto"`` to
            use the single N-run in the reference; ``None`` if there is no intBC.
        contig_pattern: Regular expression matching BAM contigs that hold this
            construct with one contig per known intBC (the intBC is then the
            contig name, e.g. ``"intID\\d+"``).
        array: Tandem array description for ``sequential_insertion`` sites.
        description: Free text.
    """

    name: str
    reference: str
    sites: list[Site]
    intbc: tuple[int, int] | Literal["auto"] | None = "auto"
    contig_pattern: str | None = None
    array: TapeArray | None = None
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.reference = self.reference.upper().replace("U", "T")
        self.sites = [s if isinstance(s, Site) else Site(**s) for s in self.sites]
        if isinstance(self.array, dict):
            self.array = TapeArray(**self.array)
        if self.intbc == "auto":
            runs = [(m.start(), m.end()) for m in re.finditer("N+", self.reference)]
            if len(runs) > 1:
                raise ValueError(
                    f"Construct {self.name!r} has {len(runs)} N-runs; specify `intbc` explicitly."
                )
            self.intbc = runs[0] if runs else None
        elif self.intbc is not None:
            self.intbc = (int(self.intbc[0]), int(self.intbc[1]))
        names = [s.name for s in self.sites]
        if len(set(names)) != len(names):
            raise ValueError("Site names must be unique.")
        for s in self.sites:
            if s.edit_model == "sequential_insertion":
                if self.array is None:
                    raise ValueError("sequential_insertion sites require `array`.")
                if s.slot is None:
                    raise ValueError(f"sequential_insertion site {s.name!r} needs `slot`.")
            if s.edit_model == "substitution":
                if s.start is None:
                    s.start = s.position
                if s.end is None:
                    raise ValueError(f"substitution site {s.name!r} needs `end`.")
            if not 0 <= s.position <= len(self.reference):
                raise ValueError(f"Site {s.name!r} position {s.position} is outside the reference.")

    # ------------------------------------------------------------ helpers
    @property
    def site_names(self) -> list[str]:
        """Names of the edit sites (molecule table columns)."""
        return [s.name for s in self.sites]

    @property
    def edit_models(self) -> set[str]:
        """Edit models used by the sites."""
        return {s.edit_model for s in self.sites}

    def to_dict(self) -> dict:
        """JSON/YAML-serializable representation."""
        d = asdict(self)
        d["sites"] = [{k: v for k, v in asdict(s).items() if v is not None} for s in self.sites]
        if self.array is None:
            d.pop("array")
        if isinstance(self.intbc, tuple):
            d["intbc"] = list(self.intbc)
        for s in d["sites"]:
            s["window"] = list(s["window"])
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | Path | None = None) -> Construct:
        """Create a construct from a dictionary (as in YAML).

        ``reference`` may be a sequence, or a mapping with ``fasta`` (path,
        relative to ``base_dir``), optional ``contig`` (record name) and
        optional ``mask`` (list of ``[start, end]`` intervals to replace with N).
        """
        d = copy.deepcopy(d)
        ref = d.get("reference")
        if isinstance(ref, dict):
            fasta = Path(ref["fasta"])
            if base_dir is not None and not fasta.is_absolute():
                fasta = Path(base_dir) / fasta
            seq = _read_fasta(fasta, ref.get("contig"))
            for start, end in ref.get("mask", []) or []:
                seq = seq[:start] + "N" * (end - start) + seq[end:]
            d["reference"] = seq
        if isinstance(d.get("intbc"), list):
            d["intbc"] = tuple(d["intbc"])
        for s in d.get("sites", []):
            if isinstance(s.get("window"), list):
                s["window"] = tuple(s["window"])
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        d.setdefault("extra", {}).update(extra)
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Construct:
        """Load a construct from a YAML file."""
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        if "construct" in d and isinstance(d["construct"], dict):
            d = d["construct"]
        return cls.from_dict(d, base_dir=path.parent)

    def to_yaml(self, path: str | Path) -> None:
        """Write the construct to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def _native(
        self, legacy: bool = False, normalize_indels: bool = True, clip_missing: bool = True
    ) -> dict:
        """Dictionary consumed by the C++ kernels."""
        sites = []
        for s in self.sites:
            sites.append(
                {
                    "name": s.name,
                    "model": _MODEL_IDS[s.edit_model],
                    "position": s.position,
                    "window_left": s.window[0],
                    "window_right": s.window[1],
                    "context": s.context,
                    "context_size": s.context_size,
                    "lookup": s.lookup or {},
                    "start": s.start if s.start is not None else s.position,
                    "end": s.end if s.end is not None else s.position,
                    "from_base": s.from_base,
                    "to_base": s.to_base,
                    "min_base_qual": s.min_base_qual,
                    "slot": s.slot if s.slot is not None else 0,
                }
            )
        out = {
            "reference": self.reference,
            "intbc_start": self.intbc[0] if self.intbc else -1,
            "intbc_end": self.intbc[1] if self.intbc else -1,
            "legacy": legacy,
            "clip_missing": clip_missing,
            "normalize_indels": normalize_indels,
            "sites": sites,
        }
        if self.array is not None:
            out["array"] = asdict(self.array)
        return out

    def __repr__(self) -> str:
        bc = f"intBC={self.intbc}" if self.intbc else "no intBC"
        models = ", ".join(sorted(self.edit_models))
        return (
            f"Construct({self.name!r}, {len(self.reference)}bp, {bc}, "
            f"{len(self.sites)} sites [{models}])"
        )


def _read_fasta(path: str | Path, contig: str | None = None) -> str:
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    if not seqs:
        raise ValueError(f"No sequences in {path}.")
    if contig is None:
        return "".join(next(iter(seqs.values())))
    if contig not in seqs:
        raise ValueError(f"Contig {contig!r} not found in {path}.")
    return "".join(seqs[contig])


def _preset_dir():
    return resources.files("cassiopeia.preprocess") / "presets"


def list_presets() -> list[str]:
    """Names of the bundled construct presets."""
    return sorted(p.name[:-5] for p in _preset_dir().iterdir() if p.name.endswith(".yaml"))


def load_construct(construct: Construct | str | Path | dict) -> Construct:
    """Get a :class:`Construct` from an object, preset name, YAML path or dict."""
    if isinstance(construct, Construct):
        return construct
    if isinstance(construct, dict):
        return Construct.from_dict(construct)
    path = Path(construct)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return Construct.from_yaml(path)
    name = str(construct)
    preset = _preset_dir() / f"{name}.yaml"
    if preset.is_file():
        with resources.as_file(preset) as p:
            return Construct.from_yaml(p)
    raise ValueError(f"Unknown construct {construct!r}. Presets: {', '.join(list_presets())}.")
