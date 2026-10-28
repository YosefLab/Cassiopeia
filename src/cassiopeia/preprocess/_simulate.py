"""Simulate lineage tracing reads with known ground truth (tests and benchmarks).

The simulator builds clones (each with its own intBCs, or a single clone in
which all cells share intBCs), lets edits accumulate down a lineage tree for
every edit model, and writes:

* 10x-style FASTQs (R1 = cell barcode + UMI, R2 = target site),
* a coordinate-sorted, indexed BAM as STARsolo / Cell Ranger would produce
  (true alignments, corrected ``CB``/``UB`` tags),
* a cell barcode whitelist and truth tables.

It includes the usual artifacts: sequencing (substitution and, optionally,
indel), cell barcode and UMI errors,
PCR duplicates, doublets, ambient molecules, a duplicated integration, a
silenced integration and (for arrays) truncated tapes.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .construct import Construct, load_construct

BASES = np.array(list("ACGT"))


@dataclass
class Edit:
    kind: str  # "I", "D", "S"
    pos: int
    length: int = 0
    seq: str = ""


@dataclass
class Cassette:
    """One integration in one cell: intBC plus its edits."""

    intbc: str
    edits: list[Edit] = field(default_factory=list)
    truncated_units: int = 0


@dataclass
class SimResult:
    """Paths and truth tables written by :func:`simulate`."""

    fastqs: list[str]
    bam: str
    whitelist: str
    cells: pd.DataFrame  # cellBC, clone, doublet, cells (source cells)
    alleles: pd.DataFrame  # cellBC, intBC, copy, site tokens (expected)
    molecules: pd.DataFrame  # cellBC, intBC, UMI, copy, n_reads, source cell
    integrations: pd.DataFrame  # clone, intBC, kind (normal/duplicated/silenced)
    construct: Construct


def _rand_seq(rng, n):
    return "".join(rng.choice(BASES, n))


def _apply(ref: str, edits: list[Edit]) -> tuple[str, list[int | None]]:
    """Edited sequence and, for each base, its reference position (None = inserted)."""
    ins: dict[int, str] = {}
    dels = np.zeros(len(ref) + 1, dtype=bool)
    subs: dict[int, str] = {}
    for e in edits:
        if e.kind == "I":
            ins[e.pos] = ins.get(e.pos, "") + e.seq
        elif e.kind == "D":
            dels[e.pos : e.pos + e.length] = True
        elif e.kind == "S":
            subs[e.pos] = e.seq
    out, rpos = [], []
    for i in range(len(ref) + 1):
        if i in ins:
            out.extend(ins[i])
            rpos.extend([None] * len(ins[i]))
        if i == len(ref):
            break
        if dels[i]:
            continue
        out.append(subs.get(i, ref[i]))
        rpos.append(i)
    return "".join(out), rpos


def _cigar(rpos: list[int | None]) -> tuple[int, list[tuple[int, int]]]:
    """CIGAR (pysam tuples) for a read whose bases map to `rpos`."""
    ops: list[list[int]] = []

    def push(op, n=1):
        if ops and ops[-1][0] == op:
            ops[-1][1] += n
        else:
            ops.append([op, n])

    first = next((r for r in rpos if r is not None), None)
    last_ref = None
    lead = 0
    for r in rpos:
        if r is None:
            if last_ref is None:
                lead += 1
            else:
                push(1)
            continue
        if last_ref is not None and r > last_ref + 1:
            push(2, r - last_ref - 1)
        if last_ref is None and lead:
            push(4, lead)
        push(0)
        last_ref = r
    if last_ref is None:
        return -1, []
    # trailing insertions -> soft clip
    if ops and ops[-1][0] == 1:
        ops[-1][0] = 4
    return first, [tuple(o) for o in ops]


class _Model:
    """Draws heritable edits for a construct."""

    def __init__(self, c: Construct, rng, edit_rate: float):
        self.c = c
        self.rng = rng
        self.rate = edit_rate

    def unedited(self, intbc: str, truncated_units: int = 0) -> Cassette:
        cas = Cassette(intbc=intbc, truncated_units=truncated_units)
        for s in self.c.sites:
            if s.edit_model == "insertion_lookup" and s.lookup:
                for k, v in s.lookup.items():
                    if v == "None":
                        cas.edits.append(Edit("I", s.position, seq=k))
        if truncated_units and self.c.array is not None:
            a = self.c.array
            end = a.start + len(a.unit) * (a.n_units - 1)
            cas.edits.append(
                Edit("D", end - truncated_units * len(a.unit), truncated_units * len(a.unit))
            )
        return cas

    def mutate(self, cas: Cassette, state: dict) -> None:
        """Edit unedited sites in place; `state` tracks per-site status."""
        rng = self.rng
        c = self.c
        for s in c.sites:
            if state.get(s.name):
                continue
            if s.edit_model == "sequential_insertion":
                continue
            if rng.random() >= self.rate:
                continue
            if s.edit_model == "indel":
                if rng.random() < 0.5:
                    n = int(rng.integers(1, 5))
                    cas.edits.append(Edit("I", s.position, seq=_rand_seq(rng, n)))
                else:
                    n = int(rng.integers(1, 15))
                    start = s.position - int(rng.integers(0, min(n, 4) + 1))
                    cas.edits.append(Edit("D", start, n))
                state[s.name] = True
            elif s.edit_model == "insertion_lookup":
                choices = [k for k, v in (s.lookup or {}).items() if v != "None"]
                k = choices[int(rng.integers(len(choices)))]
                none_keys = [kk for kk, v in (s.lookup or {}).items() if v == "None"]
                if none_keys:
                    cas.edits = [
                        e for e in cas.edits if not (e.kind == "I" and e.pos == s.position)
                    ]
                cas.edits.append(Edit("I", s.position, seq=k))
                state[s.name] = k
            elif s.edit_model == "substitution":
                pos = [p for p in range(s.start, s.end) if c.reference[p] == s.from_base]
                for p in pos:
                    if rng.random() < 0.3:
                        cas.edits.append(Edit("S", p, seq=s.to_base))
                state[s.name] = True
        if c.array is not None:
            a = c.array
            n_slots = a.n_units - cas.truncated_units
            filled = state.get("_slots", 0)
            if filled < n_slots and rng.random() < self.rate * 2:
                pos = a.start + filled * len(a.unit)
                ins = (
                    _rand_seq(rng, a.insert_len - 3) + "GGA"
                    if a.insert_len > 3
                    else _rand_seq(rng, a.insert_len)
                )
                cas.edits.append(Edit("I", pos, seq=ins))
                state["_slots"] = filled + 1


def _expected_tokens(c: Construct, cassette: Cassette) -> list[str | None]:
    from .alleles import call_read

    ref = c.reference
    if c.intbc:
        ref = ref[: c.intbc[0]] + cassette.intbc + ref[c.intbc[1] :]
    seq, rpos = _apply(ref, cassette.edits)
    start, cig = _cigar(rpos)
    res = call_read(c, start, cig, seq)
    return [t if t else None for t in res["tokens"]]


def _indel_errors(rng, read, qual, tail, rate):
    """1-bp sequencing insertions/deletions at `rate` per base; the read keeps its length."""
    bases, quals = [], []
    for b, q in zip(read, qual, strict=True):
        r = rng.random()
        if r < rate / 2:
            continue
        if r < rate:
            bases.append("ACGT"[int(rng.integers(4))])
            quals.append("F")
        bases.append(b)
        quals.append(q)
    bases += list(tail)
    quals += ["F"] * len(tail)
    return "".join(bases[: len(read)]), "".join(quals[: len(read)])


def simulate(
    out_dir: str | Path,
    construct: str | Construct = "weissman",
    *,
    n_clones: int = 4,
    cells_per_clone: int = 40,
    intbcs_per_clone: int = 5,
    generations: int = 8,
    edit_rate: float = 0.12,
    umis_per_intbc: float = 6.0,
    reads_per_umi: float = 5.0,
    seq_error: float = 0.002,
    indel_error: float = 0.0,
    cb_error: float = 0.02,
    umi_error: float = 0.02,
    doublet_rate: float = 0.08,
    ambient_rate: float = 0.02,
    duplicated_integration: bool = True,
    silenced_integration: bool = True,
    truncated_array: bool = True,
    read_length: int | None = None,
    read_start: int | None = None,
    extra_whitelist: int = 2000,
    seed: int = 0,
) -> SimResult:
    """Simulate a lineage tracing experiment; see the module docstring."""
    rng = np.random.default_rng(seed)
    c = load_construct(construct)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = _Model(c, rng, edit_rate)
    intbc_len = (c.intbc[1] - c.intbc[0]) if c.intbc else 0
    if read_start is None:
        read_start = max(0, c.intbc[0] - 21) if c.intbc and c.intbc[0] > 30 else 0
    if read_length is None:
        read_length = len(c.reference) - read_start + 20

    # --- integrations per clone
    integ_rows = []
    clone_intbcs: dict[int, list[tuple[str, int, int]]] = {}  # (intbc, copies, truncated)
    for k in range(n_clones):
        lst = []
        for j in range(intbcs_per_clone if c.intbc else 1):
            bc = _rand_seq(rng, intbc_len) if c.intbc else c.name
            copies, trunc, kind = 1, 0, "normal"
            if k == 0 and j == 0 and duplicated_integration and c.intbc:
                copies, kind = 2, "duplicated"
            if k == 0 and j == 1 and silenced_integration and c.intbc:
                kind = "silenced"
            if k == 0 and j == 2 and truncated_array and c.array is not None:
                trunc, kind = 1, "truncated"
            lst.append((bc, copies, trunc))
            integ_rows.append({"clone": str(k), "intBC": bc, "kind": kind})
        clone_intbcs[k] = lst
    integrations = pd.DataFrame(integ_rows)
    silenced = set(integrations.loc[integrations["kind"] == "silenced", "intBC"])

    # --- lineage per clone: binary tree, edits accumulate
    cells = []  # (cell_id, clone, list of (intbc, copy, Cassette))
    for k in range(n_clones):
        roots = []
        for bc, copies, trunc in clone_intbcs[k]:
            for cp in range(copies):
                roots.append((bc, cp, model.unedited(bc, trunc), {}))
        population = [roots]
        for _ in range(generations):
            nxt = []
            for cell in population:
                for _child in range(2):
                    child = []
                    for bc, cp, cas, st in cell:
                        cas2 = Cassette(cas.intbc, list(cas.edits), cas.truncated_units)
                        st2 = dict(st)
                        model.mutate(cas2, st2)
                        child.append((bc, cp, cas2, st2))
                    nxt.append(child)
            population = nxt
        idx = rng.choice(len(population), size=min(cells_per_clone, len(population)), replace=False)
        for i in idx:
            cells.append((len(cells), k, population[i]))

    # --- barcodes, doublets
    n_cells = len(cells)
    cbs = set()
    while len(cbs) < n_cells:
        cbs.add(_rand_seq(rng, 16))
    cbs = sorted(cbs)
    rng.shuffle(cbs)
    n_doublets = int(round(doublet_rate * n_cells))
    order = rng.permutation(n_cells)
    droplets = []  # (cellBC, [cell indices])
    used = set()
    i = 0
    while len(droplets) < n_cells - n_doublets and i < n_cells:
        a = int(order[i])
        i += 1
        if a in used:
            continue
        if len(droplets) < n_doublets:
            partners = [int(b) for b in order[i:] if int(b) not in used]
            if partners:
                b = partners[0]
                used.update([a, b])
                droplets.append([a, b])
                continue
        used.add(a)
        droplets.append([a])
    cell_rows = []
    for d, members in enumerate(droplets):
        cb = cbs[d]
        clones = sorted({str(cells[m][1]) for m in members})
        cell_rows.append(
            {
                "cellBC": cb,
                "clone": clones[0] if len(clones) == 1 else "|".join(clones),
                "doublet": len(members) > 1,
                "heterotypic": len(clones) > 1,
                "cells": ",".join(map(str, members)),
            }
        )
    cell_df = pd.DataFrame(cell_rows)

    # --- expected alleles per (cell, intBC, copy)
    allele_rows = []
    for d, members in enumerate(droplets):
        for m in members:
            for bc, cp, cas, _ in cells[m][2]:
                toks = _expected_tokens(c, cas)
                allele_rows.append(
                    {
                        "cellBC": cbs[d],
                        "source": m,
                        "intBC": bc,
                        "copy": cp,
                        **dict(zip(c.site_names, toks, strict=True)),
                    }
                )
    alleles = pd.DataFrame(allele_rows)

    # --- molecules and reads
    ref_cache: dict[tuple[str, int, int], tuple[str, list]] = {}
    r1_lines, r2_lines, bam_reads, mol_rows = [], [], [], []
    for d, members in enumerate(droplets):
        cb = cbs[d]
        pool = []
        for m in members:
            for bc, cp, cas, _ in cells[m][2]:
                if bc in silenced and rng.random() > 0.15:
                    continue
                copies = sum(1 for b2, *_ in cells[m][2] if b2 == bc)
                lam = umis_per_intbc / copies
                for _ in range(int(rng.poisson(lam))):
                    pool.append((bc, cp, cas, m, False))
        # ambient molecules from random other cells
        n_amb = int(rng.binomial(max(len(pool), 1), ambient_rate))
        for _ in range(n_amb):
            o = int(rng.integers(n_cells))
            bc, cp, cas, _ = cells[o][2][int(rng.integers(len(cells[o][2])))]
            pool.append((bc, cp, cas, o, True))
        for bc, cp, cas, src, ambient in pool:
            umi = _rand_seq(rng, 12)
            key = (bc, src, cp)
            if key not in ref_cache:
                ref = c.reference
                if c.intbc:
                    ref = ref[: c.intbc[0]] + bc + ref[c.intbc[1] :]
                ref_cache[key] = _apply(ref, cas.edits)
            seq, rpos = ref_cache[key]
            # read covers [first edited base mapping to >= read_start, +read_length)
            q0 = next((q for q, r in enumerate(rpos) if r is not None and r >= read_start), 0)
            frag, frag_pos = seq[q0 : q0 + read_length], rpos[q0 : q0 + read_length]
            n_reads = 1 if ambient else max(1, int(rng.poisson(reads_per_umi - 1)) + 1)
            mol_rows.append(
                {
                    "cellBC": cb,
                    "intBC": bc,
                    "UMI": umi,
                    "copy": cp,
                    "source": src,
                    "ambient": ambient,
                    "n_reads": n_reads,
                }
            )
            start, cig = _cigar(frag_pos)
            for _ in range(n_reads):
                s = np.array(list(frag))
                errs = rng.random(len(s)) < seq_error
                for e in np.flatnonzero(errs):
                    s[e] = rng.choice([b for b in "ACGT" if b != s[e]])
                read = "".join(s)
                qual = "".join("#" if e else "F" for e in errs)
                fq_read, fq_qual = read, qual
                if (
                    indel_error > 0
                ):  # sequencing indels (FASTQ only; the BAM keeps the true alignment)
                    tail = seq[q0 + len(frag) : q0 + len(frag) + 10]
                    fq_read, fq_qual = _indel_errors(rng, read, qual, tail, indel_error)
                raw_cb, raw_umi = cb, umi
                if rng.random() < cb_error:
                    p = int(rng.integers(16))
                    raw_cb = cb[:p] + rng.choice([b for b in "ACGT" if b != cb[p]]) + cb[p + 1 :]
                if rng.random() < umi_error:
                    p = int(rng.integers(12))
                    raw_umi = (
                        umi[:p] + rng.choice([b for b in "ACGT" if b != umi[p]]) + umi[p + 1 :]
                    )
                name = f"r{len(r1_lines)}"
                r1_lines.append(f"@{name}\n{raw_cb}{raw_umi}\n+\n{'F' * 28}\n")
                r2_lines.append(f"@{name}\n{fq_read}\n+\n{fq_qual}\n")
                bam_reads.append((name, start, cig, read, qual, cb, umi))
    molecules = pd.DataFrame(mol_rows)

    # --- write FASTQs (shuffled like a real run)
    perm = rng.permutation(len(r1_lines))
    r1p, r2p = out / "sim_S1_L001_R1_001.fastq.gz", out / "sim_S1_L001_R2_001.fastq.gz"
    with gzip.open(r1p, "wt") as f1, gzip.open(r2p, "wt") as f2:
        for i in perm:
            f1.write(r1_lines[i])
            f2.write(r2_lines[i])
    wl = out / "whitelist.txt"
    extra = {_rand_seq(rng, 16) for _ in range(extra_whitelist)} - set(cbs)
    with open(wl, "w") as f:
        for cb in sorted(set(cbs) | extra):
            f.write(cb + "\n")

    # --- write a Cell Ranger-like BAM
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": c.name, "LN": len(c.reference)}],
    }
    unsorted = out / "sim.unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as bf:
        for name, start, cig, read, qual, cb, umi in bam_reads:
            a = pysam.AlignedSegment(bf.header)
            a.query_name = name
            a.query_sequence = read
            a.query_qualities = pysam.qualitystring_to_array(qual)
            if start < 0:
                a.flag = 4
            else:
                a.reference_id = 0
                a.reference_start = start
                a.cigartuples = cig
                a.mapping_quality = 255
            a.set_tag("CB", cb + "-1")
            a.set_tag("UB", umi)
            bf.write(a)
    bam = out / "sim.bam"
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))
    unsorted.unlink()
    cell_df["cellBC_10x"] = cell_df["cellBC"] + "-1"
    cell_df.to_csv(out / "truth_cells.csv", index=False)
    alleles.to_csv(out / "truth_alleles.csv", index=False)
    molecules.to_csv(out / "truth_molecules.csv", index=False)
    integrations.to_csv(out / "truth_integrations.csv", index=False)
    return SimResult(
        [str(r1p), str(r2p)], str(bam), str(wl), cell_df, alleles, molecules, integrations, c
    )
