"""Tests for the C++ kernels (alignment, allele calling, edit distance)."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cassiopeia as cas
from cassiopeia.preprocess._native import (
    align_sequences,
    correct_intbcs,
    levenshtein,
    merge_check,
)

REF = cas.pp.load_construct("weissman").reference
# Outputs of the pre-3.0 pipeline on the inputs below (see data/README.md).
DATA = Path(__file__).parent / "data"


def _mutate(rng, seq, rate=0.01):
    out, i = [], 0
    while i < len(seq):
        r = rng.random()
        if r < rate:
            out.append(rng.choice("ACGT"))
            i += 1
        elif r < 2 * rate:
            i += rng.randint(1, 30)
        elif r < 3 * rate:
            out.extend(rng.choice("ACGT") for _ in range(rng.randint(1, 8)))
        else:
            out.append(seq[i])
            i += 1
    return "".join(out)


def _queries():
    rng = random.Random(0)
    qs = []
    for _ in range(300):
        start = rng.randint(0, 40)
        qs.append(_mutate(rng, REF[start : start + rng.randint(100, 300)]))
    for _ in range(50):  # deletions inside repeats / homopolymers
        p = rng.randint(30, 250)
        qs.append((REF[:p] + REF[p + rng.randint(1, 4) :])[:250])
    return qs


@pytest.fixture(scope="module")
def queries():
    return _queries()


@pytest.fixture(scope="module")
def legacy_reads():
    """Legacy alignments (local, global) of `queries`."""
    df = pd.read_csv(DATA / "legacy_reads.tsv", sep="\t", dtype=str, keep_default_na=False)
    ints = ["local_qstart", "local_rstart", "local_score", "global_score"]
    return df.astype(dict.fromkeys(ints, int))


@pytest.mark.parametrize("method", ["local", "global"])
def test_smith_waterman_matches_legacy(queries, legacy_reads, method):
    native = align_sequences(REF, queries, None, method == "global", 2)
    for n, row in zip(native, legacy_reads.itertuples(), strict=True):
        if method == "local":
            assert (n[3], n[2], n[1], n[0]) == (
                row.local_cigar,
                row.local_qstart,
                row.local_rstart,
                row.local_score,
            )
        else:
            assert (n[3], n[0]) == (row.global_cigar, row.global_score)


def _read_pair(rng):
    """Two reads of one molecule: substitutions, small indels, Ns, other molecules."""
    start = rng.randint(0, 200)
    a = REF[start : start + rng.randint(80, 250)]
    b = list(a)
    for _ in range(rng.choice([0, 0, 1, 2, 3, 4, 6])):
        b[rng.randrange(len(b))] = rng.choice("ACGTN")
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        p = rng.randrange(1, len(b) - 1)
        if rng.random() < 0.5:
            del b[p : p + rng.randint(1, 3)]
        else:
            b[p:p] = rng.choices("ACGT", k=rng.randint(1, 3))
    b = "".join(b)
    if rng.random() < 0.1:
        b = REF[rng.randint(0, 400) :][: len(b)]
    if rng.random() < 0.3:  # reads of different lengths
        b = b[: rng.randint(len(b) // 2, len(b))]
    quals = [2, 12, 20, 21, 23, 30, 37, 37, 37, 37]
    return a, b, [rng.choice(quals) for _ in a], [rng.choice(quals) for _ in b]


def _read_pairs(n=2000):
    rng = random.Random(3)
    return [_read_pair(rng) for _ in range(n)]


def test_cluster_merge_matches_legacy():
    # one legacy merge decision per pair: global alignment with <= 2 gap bases and
    # <= 3 mismatches between bases of quality > 20
    expected = (DATA / "legacy_merge.txt").read_text().strip()
    assert 300 < expected.count("1") < 1700

    def mask(seq, qual):
        return "".join(c if q > 20 else c.lower() for c, q in zip(seq, qual, strict=True))

    got = "".join(
        "1" if merge_check(mask(a, qa), mask(b, qb), 2, 3) else "0"
        for a, b, qa, qb in _read_pairs(len(expected))
    )
    assert got == expected


def test_indel_model_calls():
    c = cas.pp.load_construct("weissman")
    bc = "ACGTACGTACGTAC"
    seq = REF[:20] + bc + REF[34:300]
    res = cas.pp.call_read(c, 0, [(0, len(seq))], seq)
    assert res["intBC"] == bc
    assert all("[None]" in t for t in res["tokens"])
    # 5bp deletion at the first cut site
    edited = seq[:110] + seq[115:]
    res = cas.pp.call_read(c, 0, [(0, 110), (2, 5), (0, len(edited) - 110)], edited)
    assert "[111:5D]" in res["tokens"][0]
    assert "[None]" in res["tokens"][1]


def test_clipped_site_is_missing():
    c = cas.pp.load_construct("weissman")
    seq = REF[:20] + "A" * 14 + REF[34:200]
    # alignment ends at 200: third cut site (220) not covered
    res = cas.pp.call_read(c, 0, [(0, len(seq))], seq)
    assert res["tokens"][2] == ""
    # soft clip right after the second cut site -> missing, not unedited
    res = cas.pp.call_read(c, 0, [(0, 170), (4, len(seq) - 170)], seq)
    assert res["tokens"][1] == "" and res["clipped"][1]


def test_insertion_lookup_model():
    c = cas.pp.load_construct("petracer")
    ref = c.reference
    unedited = ref[:448] + "CTTGGG" + ref[448:]
    read = unedited[251:]
    res = cas.pp.call_read(c, 251, [(0, 448 - 251), (1, 6), (0, len(ref) - 448)], read)
    assert res["tokens"] == ["None", "None", "None"]
    # LM1 at RNF2 (aligner places the insertion one base off), LM6 replaces CTTGGG at EMX1
    edited = ref[:332] + "ACAGT" + ref[332:448] + "CCTTT" + ref[448:]
    read = edited[251:]
    res = cas.pp.call_read(
        c, 251, [(0, 333 - 251), (1, 5), (0, 448 - 333), (1, 5), (0, len(ref) - 448)], read
    )
    assert res["tokens"][0] == "1"
    assert res["tokens"][2] == "6"
    # no insertion at EMX1 relative to the short reference = 6bp deletion
    read = ref[251:]
    res = cas.pp.call_read(c, 251, [(0, len(read))], read)
    assert res["tokens"][2] == "D6"


def test_sequential_insertion_handles_truncated_tapes():
    c = cas.pp.load_construct("typewriter")
    a = c.array
    ref = c.reference[:35] + "ACGTACGT" + c.reference[43:]
    # two edits, full 5x tape
    seq = ref[: a.start] + "AAAGGA" + ref[a.start : a.start + 14] + "CCCGGA" + ref[a.start + 14 :]
    res = cas.pp.call_read(
        c, 0, [(0, a.start), (1, 6), (0, 14), (1, 6), (0, len(ref) - a.start - 14)], seq
    )
    assert res["intBC"] == "ACGTACGT"
    assert res["tokens"] == ["AAAGGA", "CCCGGA", "None", "None", "None"]
    # tape shortened by one unit: 4 slots, last slot missing (not unedited)
    short = ref[: a.start + 42] + ref[a.start + 56 :]
    res = cas.pp.call_read(
        c, 0, [(0, a.start + 42), (2, 14), (0, len(short) - a.start - 42)], short
    )
    assert res["tokens"] == ["None", "None", "None", "None", ""]
    # read ends inside the tape: later slots missing
    part = seq[: a.start + 26]
    res = cas.pp.call_read(c, 0, [(0, a.start), (1, 6), (0, 20)], part)
    assert res["tokens"][0] == "AAAGGA"
    assert res["tokens"][2:] == ["", "", ""]


def test_substitution_model():
    c = cas.pp.load_construct("baseline")
    s = c.sites[0]
    ref = c.reference
    seq = list(ref)
    a_pos = [p for p in range(s.start, s.end) if ref[p] == "A"]
    seq[a_pos[1]] = "G"
    seq = "".join(seq)
    res = cas.pp.call_read(c, 0, [(0, len(seq))], seq)
    expected = "".join("1" if i == 1 else "0" for i in range(len(a_pos)))
    assert res["tokens"][0] == expected
    assert res["tokens"][1] == "None"


def test_levenshtein():
    rng = np.random.default_rng(0)

    def dp(a, b):
        prev = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            cur = [i] + [0] * len(b)
            for j in range(1, len(b) + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[-1]

    for _ in range(200):
        a = "".join(rng.choice(list("ACGT"), rng.integers(0, 40)))
        b = "".join(rng.choice(list("ACGT"), rng.integers(0, 40)))
        assert levenshtein(a, b) == dp(a, b)
    long = "A" * 70
    assert levenshtein(long, long[:-2]) == 2


def test_correct_intbcs_kernel():
    seqs = ["AAAACCCCGGGGTT", "AAAACCCCGGGGTA", "TTTTGGGGCCCCAA", "TTTTGGGGCCCCA"]
    counts = [100, 2, 50, 1]
    res = correct_intbcs(seqs, counts, None, 2, 5.0, 1)
    assert list(res) == [0, 0, 2, 2]
    res = correct_intbcs(seqs, counts, ["AAAACCCCGGGGTT", "TTTTGGGGCCCCAA"], 1, 5.0, 1)
    assert list(res) == [0, 0, 1, 1]
