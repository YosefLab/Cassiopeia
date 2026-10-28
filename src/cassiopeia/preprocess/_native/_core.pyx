# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython bindings for the C++ preprocessing kernels.

These are private; use the functions in :mod:`cassiopeia.preprocess`.
"""

from libc.stdint cimport int32_t, int64_t, uint32_t
from libc.string cimport memcpy
from libcpp cimport bool as cbool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np


cdef extern from "src/sw.hpp" namespace "cas":
    cdef cppclass Scoring:
        int match
        int mismatch
        int n_score
        int gap_open
        int gap_extend


cdef extern from "src/caller.hpp" namespace "cas":
    cdef cppclass SiteSpec:
        SiteSpec() except +
        string name
        int position
        int window_left
        int window_right
        cbool context
        int context_size
        unordered_map[string, string] lookup
        int start
        int end
        char from_base
        char to_base
        int min_base_qual
        int slot

    void set_site_model(SiteSpec &, int)

    cdef cppclass ArraySpec:
        cbool enabled
        int start
        string unit
        int insert_len
        int n_units
        string end_flank
        int max_mismatch
        int anchor

    cdef cppclass ConstructSpec:
        string reference
        int intbc_start
        int intbc_end
        cbool clip_missing
        cbool normalize_indels
        vector[SiteSpec] sites
        ArraySpec array

    cdef cppclass ReadCall:
        string intbc
        vector[string] tokens
        vector[unsigned char] clipped

    cbool call_read(const ConstructSpec &, int, const vector[uint32_t] &, const string &,
                    const unsigned char *, ReadCall &) except +


cdef extern from "src/alleles.hpp" namespace "cas":
    cdef cppclass CallAllelesOptions:
        string path
        vector[int] tids
        cbool use_index
        string cb_tag
        string umi_tag
        string count_tag
        int min_mapq
        int n_threads
        vector[string] cells
        int umi_max_dist
        int64_t min_reads
        cbool intbc_from_contig
        string default_intbc

    cdef cppclass Categorical:
        vector[string] categories
        vector[int32_t] codes

    cdef cppclass MoleculeTable:
        Categorical cell
        Categorical umi
        Categorical intbc
        vector[Categorical] sites
        vector[int32_t] n_reads
        vector[pair[string, int64_t]] stats
        vector[int64_t] site_missing
        vector[int64_t] site_clipped
        vector[int64_t] hist_reads
        vector[int64_t] hist_umis

    MoleculeTable c_call_alleles "cas::call_alleles"(const ConstructSpec &, const CallAllelesOptions &) except + nogil
    void bam_references(const string &, vector[string] &, vector[int32_t] &) except +


cdef extern from "src/align.hpp" namespace "cas":
    cdef cppclass Segment:
        int read
        int start
        int end

    cdef cppclass LayoutSpec:
        int n_reads
        vector[Segment] cb
        vector[Segment] umi
        Segment cdna

    cdef cppclass AlignOptions:
        vector[vector[string]] lanes
        LayoutSpec layout
        string output_bam
        string whitelist
        int min_bc_qual
        int min_base_qual
        int umi_max_dist
        cbool dedup_umi
        int max_mismatches
        int max_indels
        cbool c_global "global"
        Scoring scoring
        int min_score
        int strand
        int n_threads
        string ref_name
        string ref_seq
        double cb_confidence
        string header_extra
        size_t batch_size

    cdef cppclass AlignSummary:
        vector[pair[string, int64_t]] stats
        vector[string] cells
        vector[int64_t] cell_reads
        vector[int64_t] cell_umis
        vector[int64_t] reads_per_umi_hist
        vector[int64_t] score_hist

    AlignSummary c_align_fastqs "cas::align_fastqs"(const AlignOptions &) except + nogil

    cdef cppclass SimpleAlignment:
        int score
        int ref_start
        int query_start
        string cigar

    vector[SimpleAlignment] c_align_sequences "cas::align_sequences"(
        const string &, const vector[string] &, const Scoring &, cbool, int) except + nogil
    cbool c_merge_check "cas::merge_check"(const string &, const string &, int, int) except +


cdef extern from "src/intbc.hpp" namespace "cas":
    int edit_distance(const string &, const string &)
    vector[int32_t] c_correct_intbcs "cas::correct_intbcs"(
        const vector[string] &, const vector[int64_t] &, const vector[string] &, int, double, int) except + nogil


# --------------------------------------------------------------- helpers

cdef inline string _b(object s):
    if isinstance(s, bytes):
        return s
    return str(s).encode("utf-8")


cdef inline str _s(const string &s):
    return s.decode("utf-8", "replace")


cdef object _i32(const vector[int32_t] &v):
    arr = np.empty(v.size(), dtype=np.int32)
    cdef int32_t[::1] view = arr
    if v.size():
        memcpy(&view[0], v.data(), v.size() * sizeof(int32_t))
    return arr


cdef object _i64(const vector[int64_t] &v):
    arr = np.empty(v.size(), dtype=np.int64)
    cdef int64_t[::1] view = arr
    if v.size():
        memcpy(&view[0], v.data(), v.size() * sizeof(int64_t))
    return arr


cdef list _strs(const vector[string] &v):
    cdef size_t i
    return [v[i].decode("utf-8", "replace") for i in range(v.size())]


cdef dict _stats(const vector[pair[string, int64_t]] &v):
    cdef size_t i
    return {v[i].first.decode(): v[i].second for i in range(v.size())}


cdef ConstructSpec _construct(dict c):
    cdef ConstructSpec spec
    cdef SiteSpec site
    spec.reference = _b(c["reference"])
    spec.intbc_start = c.get("intbc_start", -1)
    spec.intbc_end = c.get("intbc_end", -1)
    spec.clip_missing = c.get("clip_missing", True)
    spec.normalize_indels = c.get("normalize_indels", True)
    for s in c["sites"]:
        site = SiteSpec()
        site.name = _b(s["name"])
        set_site_model(site, s["model"])
        site.position = s.get("position", 0)
        site.window_left = s.get("window_left", 0)
        site.window_right = s.get("window_right", 0)
        site.context = s.get("context", True)
        site.context_size = s.get("context_size", 5)
        for k, v in (s.get("lookup") or {}).items():
            site.lookup[_b(k)] = _b(v)
        site.start = s.get("start", 0)
        site.end = s.get("end", 0)
        site.from_base = ord(s.get("from_base", "A"))
        site.to_base = ord(s.get("to_base", "G"))
        site.min_base_qual = s.get("min_base_qual", 0)
        site.slot = s.get("slot", 0)
        spec.sites.push_back(site)
    a = c.get("array")
    if a:
        spec.array.enabled = True
        spec.array.start = a["start"]
        spec.array.unit = _b(a["unit"])
        spec.array.insert_len = a["insert_len"]
        spec.array.n_units = a["n_units"]
        spec.array.end_flank = _b(a.get("end_flank", ""))
        spec.array.max_mismatch = a.get("max_mismatch", 1)
        spec.array.anchor = a.get("anchor") if a.get("anchor") is not None else -1
    return spec


cdef Scoring _scoring(dict s):
    cdef Scoring sc
    sc.match = s.get("match", 5)
    sc.mismatch = s.get("mismatch", -4)
    sc.n_score = s.get("n_score", 0)
    sc.gap_open = s.get("gap_open", 20)
    sc.gap_extend = s.get("gap_extend", 1)
    return sc


cdef Segment _segment(object t):
    cdef Segment s
    s.read = t[0]
    s.start = t[1]
    s.end = -1 if t[2] is None else t[2]
    return s


# ----------------------------------------------------------------- api

def references(path):
    """Return (names, lengths) of the references in a BAM header."""
    cdef vector[string] names
    cdef vector[int32_t] lengths
    bam_references(_b(path), names, lengths)
    return _strs(names), [lengths[i] for i in range(lengths.size())]


def call_alleles(dict construct, dict options):
    """Run the native BAM -> molecule table pass. Returns a dict of columns."""
    cdef ConstructSpec spec = _construct(construct)
    cdef CallAllelesOptions opt
    opt.path = _b(options["path"])
    for t in options.get("tids", []):
        opt.tids.push_back(t)
    opt.use_index = options.get("use_index", True)
    opt.cb_tag = _b(options.get("cb_tag") or "")
    opt.umi_tag = _b(options.get("umi_tag") or "")
    opt.count_tag = _b(options.get("count_tag") or "")
    opt.min_mapq = options.get("min_mapq", 0)
    opt.n_threads = options.get("n_threads", 1)
    for c in options.get("cells") or []:
        opt.cells.push_back(_b(c))
    opt.umi_max_dist = options.get("umi_max_dist", 0)
    opt.min_reads = options.get("min_reads", 0)
    opt.intbc_from_contig = options.get("intbc_from_contig", False)
    opt.default_intbc = _b(options.get("default_intbc", "0"))
    cdef MoleculeTable res
    with nogil:
        res = c_call_alleles(spec, opt)
    cdef size_t i
    out = {
        "cellBC": (_i32(res.cell.codes), _strs(res.cell.categories)),
        "UMI": (_i32(res.umi.codes), _strs(res.umi.categories)),
        "intBC": (_i32(res.intbc.codes), _strs(res.intbc.categories)),
        "sites": [(_i32(res.sites[i].codes), _strs(res.sites[i].categories)) for i in range(res.sites.size())],
        "n_reads": _i32(res.n_reads),
        "stats": _stats(res.stats),
        "site_missing": _i64(res.site_missing),
        "site_clipped": _i64(res.site_clipped),
        "hist_reads": _i64(res.hist_reads),
        "hist_umis": _i64(res.hist_umis),
    }
    return out


def call_read_py(dict construct, int ref_pos, list cigar, str seq, qual=None):
    """Call a single read (for testing). `cigar` is a list of (op, length)."""
    cdef ConstructSpec spec = _construct(construct)
    cdef vector[uint32_t] cig
    for op, length in cigar:
        cig.push_back((<uint32_t>length << 4) | <uint32_t>op)
    cdef string s = _b(seq)
    cdef ReadCall call
    cdef bytes qb
    cdef const unsigned char *qp = NULL
    if qual is not None:
        qb = bytes(qual)
        qp = <const unsigned char *> qb
    ok = call_read(spec, ref_pos, cig, s, qp, call)
    if not ok:
        return None
    return {
        "intBC": _s(call.intbc),
        "tokens": [_s(call.tokens[i]) for i in range(call.tokens.size())],
        "clipped": [bool(call.clipped[i]) for i in range(call.clipped.size())],
    }


def align_fastqs(dict options):
    """Run the native FASTQ -> aligned BAM pass. Returns a summary dict."""
    cdef AlignOptions opt
    cdef vector[string] lane_v
    for lane in options["lanes"]:
        lane_v.clear()
        for p in lane:
            lane_v.push_back(_b(p))
        opt.lanes.push_back(lane_v)
    layout = options["layout"]
    opt.layout.n_reads = layout["n_reads"]
    for t in layout["cb"]:
        opt.layout.cb.push_back(_segment(t))
    for t in layout["umi"]:
        opt.layout.umi.push_back(_segment(t))
    opt.layout.cdna = _segment(layout["cdna"])
    opt.output_bam = _b(options["output_bam"])
    opt.whitelist = _b(options.get("whitelist") or "")
    opt.min_bc_qual = options.get("min_bc_qual", 10)
    opt.min_base_qual = options.get("min_base_qual", 20)
    opt.umi_max_dist = options.get("umi_max_dist", 1)
    opt.dedup_umi = options.get("dedup_umi", True)
    opt.max_mismatches = options.get("max_mismatches", 3)
    opt.max_indels = options.get("max_indels", 2)
    opt.c_global = options.get("global", False)
    opt.scoring = _scoring(options.get("scoring", {}))
    opt.min_score = options.get("min_score", 1)
    opt.strand = options.get("strand", 0)
    opt.n_threads = options.get("n_threads", 1)
    opt.ref_name = _b(options["ref_name"])
    opt.ref_seq = _b(options["ref_seq"])
    opt.cb_confidence = options.get("cb_confidence", 0.975)
    opt.header_extra = _b(options.get("header_extra", ""))
    opt.batch_size = options.get("batch_size", 200000)
    cdef AlignSummary res
    with nogil:
        res = c_align_fastqs(opt)
    return {
        "stats": _stats(res.stats),
        "cells": _strs(res.cells),
        "cell_reads": _i64(res.cell_reads),
        "cell_umis": _i64(res.cell_umis),
        "reads_per_umi_hist": _i64(res.reads_per_umi_hist),
        "score_hist": _i64(res.score_hist),
    }


def align_sequences(str ref, list queries, dict scoring=None, bint global_=False, int n_threads=1):
    """Align sequences to a reference; returns list of (score, ref_start, query_start, cigar)."""
    cdef vector[string] q
    for s in queries:
        q.push_back(_b(s))
    cdef Scoring sc = _scoring(scoring or {})
    cdef string r = _b(ref)
    cdef vector[SimpleAlignment] res
    cdef cbool g = global_
    with nogil:
        res = c_align_sequences(r, q, sc, g, n_threads)
    cdef size_t i
    return [(res[i].score, res[i].ref_start, res[i].query_start, _s(res[i].cigar)) for i in range(res.size())]


def merge_check(str a, str b, int max_indels=2, int max_mismatches=3):
    """Legacy test for merging two clusters of a UMI's reads (lowercase = low quality)."""
    return c_merge_check(_b(a), _b(b), max_indels, max_mismatches)


def levenshtein(str a, str b):
    """Edit distance between two strings."""
    return edit_distance(_b(a), _b(b))


def correct_intbcs(list seqs, counts, list whitelist=None, int max_dist=2, double min_ratio=5.0, int n_threads=1):
    """Return an index per intBC into `whitelist` (if given) or `seqs` (-1 = unplaced)."""
    cdef vector[string] s
    cdef vector[int64_t] c
    cdef vector[string] w
    for x in seqs:
        s.push_back(_b(x))
    for x in counts:
        c.push_back(int(x))
    for x in whitelist or []:
        w.push_back(_b(x))
    cdef vector[int32_t] res
    with nogil:
        res = c_correct_intbcs(s, c, w, max_dist, min_ratio, n_threads)
    return _i32(res)
