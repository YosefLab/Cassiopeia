#include "caller.hpp"

#include <algorithm>

#include "common.hpp"
#include "sw.hpp"

namespace cas {

namespace {

struct AlnBlock {
    int r, q, len;
};
struct InsEvent {
    int r, q, len;
};
struct DelEvent {
    int r, len, q;
};

struct Walk {
    std::vector<AlnBlock> blocks;
    std::vector<InsEvent> ins;
    std::vector<DelEvent> dels;
    int ref_lo = 0, ref_hi = 0;  // aligned reference span [lo, hi)
    int lead_clip = 0;
    std::vector<uint32_t> core_ops;  // ops without clips (M/I/D/N)

    int query_at(int r) const {
        for (const auto &b : blocks)
            if (r >= b.r && r < b.r + b.len) return b.q + (r - b.r);
        return -1;
    }
    bool covered(int r0, int r1) const { return r0 >= ref_lo && r1 <= ref_hi; }
    bool deleted(int r) const {
        for (const auto &d : dels)
            if (r >= d.r && r < d.r + d.len) return true;
        return false;
    }
};

void build_walk(int ref_pos, const std::vector<uint32_t> &cigar, Walk &w) {
    w.blocks.clear();
    w.ins.clear();
    w.dels.clear();
    w.core_ops.clear();
    w.ref_lo = w.ref_hi = 0;
    w.lead_clip = 0;
    int r = ref_pos, q = 0;
    bool seen_aligned = false;
    for (uint32_t c : cigar) {
        uint32_t op = c & 0xf;
        int len = static_cast<int>(c >> 4);
        switch (op) {
            case kM:
            case kEq:
            case kX:
                w.blocks.push_back({r, q, len});
                // merge adjacent M ops created by =/X
                if (!w.core_ops.empty() && (w.core_ops.back() & 0xf) == kM)
                    w.core_ops.back() += static_cast<uint32_t>(len) << 4;
                else
                    w.core_ops.push_back((static_cast<uint32_t>(len) << 4) | kM);
                r += len;
                q += len;
                seen_aligned = true;
                break;
            case kI:
                w.ins.push_back({r, q, len});
                w.core_ops.push_back(c);
                q += len;
                break;
            case kD:
            case kN:
                w.dels.push_back({r, len, q});
                if (!w.core_ops.empty() && (w.core_ops.back() & 0xf) == kD)
                    w.core_ops.back() += static_cast<uint32_t>(len) << 4;
                else
                    w.core_ops.push_back((static_cast<uint32_t>(len) << 4) | kD);
                r += len;
                break;
            case kS:
                if (!seen_aligned && w.core_ops.empty()) w.lead_clip += len;
                q += len;
                break;
            default:
                break;  // H, P
        }
    }
    if (!w.blocks.empty()) {
        w.ref_lo = w.blocks.front().r;
        w.ref_hi = w.blocks.back().r + w.blocks.back().len;
    }
}

// Port of the legacy Cassiopeia parse_cigar (alignment_utilities.py) so that
// indel allele strings are identical.
void legacy_parse(const std::string &ref, int ref_start, int query_start, const std::vector<uint32_t> &ops,
                  const std::string &seq, int bs, int be, const std::vector<const SiteSpec *> &sites,
                  std::string &intBC, bool &intbc_found, std::vector<std::string> &indels,
                  std::vector<std::string> &uncut, bool ref_context) {
    // ref_context: take context bases from the reference instead of the read
    // so sequencing errors next to an edit do not create distinct alleles.
    const size_t n = sites.size();
    indels.assign(n, "");
    uncut.assign(n, "");
    intBC = "NC";
    const bool has_bc = bs >= 0;
    const std::string ref_anchor = has_bc ? py_slice(ref, bs - 11, bs) : std::string();
    const int intBC_length = has_bc ? be - bs : 0;
    int ref_pointer = ref_start, query_pointer = query_start, query_pad = 0;
    for (uint32_t c : ops) {
        uint32_t type = c & 0xf;
        int length = static_cast<int>(c >> 4);
        if (type == kM) {
            int pos_start = ref_pointer, pos_end = ref_pointer + length;
            if (has_bc && bs >= pos_start && be <= pos_end) {
                int intBC_offset = bs - pos_start;
                if (query_pad > -intBC_length) {
                    int end_index = query_pointer + intBC_offset + intBC_length + query_pad;
                    intBC = py_slice(seq, query_pointer + intBC_offset, end_index);
                }
            }
            for (size_t i = 0; i < n; ++i) {
                int site = sites[i]->position;
                if (site >= pos_start && site <= pos_end && indels[i].empty()) {
                    int loc = query_pointer + (site - pos_start);
                    if (sites[i]->context) {
                        int cs = sites[i]->context_size;
                        if (ref_context)
                            uncut[i] = py_slice(ref, site - cs, site) + "[None]" + py_slice(ref, site, site + cs);
                        else
                            uncut[i] = py_slice(seq, loc - cs, loc) + "[None]" + py_slice(seq, loc, loc + cs);
                    } else {
                        uncut[i] = "None";
                    }
                }
            }
            if (has_bc && pos_start >= bs && pos_start <= be && pos_end > be) {
                int new_len = be - pos_start + 1;
                intBC = py_slice(seq, query_pointer, query_pointer + new_len);
            }
            if (has_bc && pos_end >= bs && pos_end <= be && pos_start < bs) {
                int new_len = pos_end - bs + 1;
                int off = bs - pos_start;
                intBC = py_slice(seq, query_pointer + off, query_pointer + off + new_len);
            }
            ref_pointer += length;
            query_pointer += length;
        } else if (type == kI) {
            if (has_bc && ref_pointer == bs) query_pad = length;
            int pos_start = query_pointer;
            query_pointer += length;
            for (size_t i = 0; i < n; ++i) {
                int w0 = sites[i]->position - sites[i]->window_left;
                int w1 = sites[i]->position + sites[i]->window_right;
                if (ref_pointer >= w0 && ref_pointer <= w1) {
                    if (sites[i]->context) {
                        int cs = sites[i]->context_size;
                        std::string l = ref_context ? py_slice(ref, ref_pointer - cs, ref_pointer)
                                                    : py_slice(seq, pos_start - cs, pos_start);
                        std::string r = ref_context ? py_slice(seq, pos_start, pos_start + length) +
                                                          py_slice(ref, ref_pointer, ref_pointer + cs)
                                                    : py_slice(seq, pos_start, pos_start + cs + length);
                        indels[i] += l + "[" + std::to_string(ref_pointer + 1) + ":" + std::to_string(length) + "I]" + r;
                    } else {
                        indels[i] += std::to_string(ref_pointer + 1) + ":" + std::to_string(length) + "I";
                    }
                }
            }
        } else if (type == kD || type == kN) {
            int pos_start = ref_pointer;
            ref_pointer += length;
            if (has_bc && ref_pointer == bs) query_pad = -length;
            for (size_t i = 0; i < n; ++i) {
                int w0 = sites[i]->position - sites[i]->window_left;
                int w1 = sites[i]->position + sites[i]->window_right;
                if ((w0 <= ref_pointer && ref_pointer <= w1) || (w0 <= pos_start && pos_start <= w1) ||
                    (pos_start <= w0 && ref_pointer >= w1) || (pos_start >= w0 && ref_pointer <= w1)) {
                    if (sites[i]->context) {
                        int cs = sites[i]->context_size;
                        std::string l = ref_context ? py_slice(ref, pos_start - cs, pos_start)
                                                    : py_slice(seq, query_pointer - cs, query_pointer);
                        std::string r = ref_context ? py_slice(ref, ref_pointer, ref_pointer + cs)
                                                    : py_slice(seq, query_pointer, query_pointer + cs);
                        indels[i] += l + "[" + std::to_string(pos_start + 1) + ":" + std::to_string(length) + "D]" + r;
                    } else {
                        indels[i] += std::to_string(pos_start + 1) + ":" + std::to_string(length) + "D";
                    }
                }
            }
        }
    }
    if (has_bc && (intBC == "NC" || static_cast<int>(intBC.size()) < intBC_length)) {
        std::string anchor = py_slice(seq, bs - 11, bs);
        if (anchor == ref_anchor) intBC = py_slice(seq, bs, be);
    }
    intbc_found = has_bc && intBC != "NC";
}

std::string flank_intbc(const Walk &w, const std::string &seq, int bs, int be, int ref_len) {
    if (bs <= 0 || be >= ref_len) return std::string();
    int qL = w.query_at(bs - 1);
    int qR = w.query_at(be);
    if (qL < 0 || qR < 0 || qR <= qL) return std::string();
    return seq.substr(qL + 1, qR - qL - 1);
}

std::string call_insertion_site_raw(const SiteSpec &s, const Walk &w, const std::string &ref,
                                    const std::string &seq, bool &clipped);

// Some references lack a sequence that is present in the unedited construct
// (e.g. PEtracer EMX1 is 6bp short); the lookup then maps that "insertion"
// to "None" and a read without any insertion carries a deletion.
const char kNoIndel[] = "\x01";

std::string call_insertion_site(const SiteSpec &s, const Walk &w, const std::string &ref, const std::string &seq,
                                bool &clipped) {
    std::string tok = call_insertion_site_raw(s, w, ref, seq, clipped);
    if (tok == kNoIndel) {
        for (const auto &kv : s.lookup)
            if (kv.second == "None" && !kv.first.empty()) return "D" + std::to_string(kv.first.size());
        return "None";
    }
    return tok;
}

std::string call_insertion_site_raw(const SiteSpec &s, const Walk &w, const std::string &ref,
                                    const std::string &seq, bool &clipped) {
    const int p = s.position;
    const int rl = p - s.window_left - 1;
    const int rr = p + s.window_right;
    clipped = false;
    if (rl < 0 || rr >= static_cast<int>(ref.size())) return std::string();
    int qL = w.query_at(rl);
    int qR = w.query_at(rr);
    if (qL >= 0 && qR > qL) {
        // Sequence-level matching of the window: unedited, or a known
        // insertion at any insertion point inside the window. This does not
        // depend on where the aligner placed the insertion.
        std::string_view seg(seq.data() + qL, qR - qL + 1);
        std::string_view win(ref.data() + rl, rr - rl + 1);
        if (seg == win) return kNoIndel;
        if (seg.size() > win.size()) {
            size_t ins_len = seg.size() - win.size();
            for (int o = p - s.window_left; o <= p + s.window_right; ++o) {
                size_t lo = static_cast<size_t>(o - rl);
                if (seg.substr(0, lo) != win.substr(0, lo)) continue;
                if (seg.substr(lo + ins_len) != win.substr(lo)) continue;
                std::string ins(seg.substr(lo, ins_len));
                auto it = s.lookup.find(ins);
                if (it != s.lookup.end()) return it->second;
            }
        }
    } else if (!w.covered(rl, rr + 1)) {
        clipped = true;
        return std::string();
    }
    // Fall back to the CIGAR events inside the window.
    const int w0 = p - s.window_left, w1 = p + s.window_right;
    std::string ins_seq;
    int del_len = 0;
    for (const auto &e : w.ins)
        if (e.r >= w0 && e.r <= w1) ins_seq += seq.substr(e.q, e.len);
    for (const auto &d : w.dels)
        if (d.r <= w1 && d.r + d.len >= w0) del_len += d.len;
    if (!w.covered(w0, w1 + 1) && ins_seq.empty() && del_len == 0) {
        clipped = true;
        return std::string();
    }
    if (ins_seq.empty() && del_len == 0) return kNoIndel;
    if (del_len == 0) {
        auto it = s.lookup.find(ins_seq);
        if (it != s.lookup.end()) return it->second;
        return ins_seq;
    }
    std::string tok = "D" + std::to_string(del_len);
    if (!ins_seq.empty()) tok += "I" + ins_seq;
    return tok;
}

std::string call_substitution_site(const SiteSpec &s, const Walk &w, const std::string &ref, const std::string &seq,
                                   const uint8_t *qual, bool &clipped) {
    clipped = false;
    int b = std::max(0, s.start), e = std::min(static_cast<int>(ref.size()), s.end);
    if (!w.covered(b, e)) {
        clipped = true;
        return std::string();
    }
    std::string tok;
    bool edited = false;
    for (int r = b; r < e; ++r) {
        if (ref[r] != s.from_base) continue;
        int q = w.query_at(r);
        char c;
        if (q < 0) {
            c = w.deleted(r) ? 'D' : 'N';
        } else if (qual && qual[0] != 0xff && qual[q] < s.min_base_qual) {
            c = 'N';
        } else if (seq[q] == s.to_base) {
            c = '1';
            edited = true;
        } else if (seq[q] == s.from_base) {
            c = '0';
        } else {
            c = 'N';
        }
        if (c == 'D') edited = true;
        tok.push_back(c);
    }
    if (!edited && tok.find('N') == std::string::npos) return "None";
    return tok;
}

int hamming_prefix(std::string_view a, std::string_view b) {
    int d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += a[i] != b[i];
    return d;
}

// Parses a sequentially edited tandem array directly from the read so that
// arrays shortened by deletions are handled (no assumption about length).
void call_array(const ArraySpec &a, const Walk &w, const std::string &ref, const std::string &seq,
                std::vector<std::string> &slot_tokens, std::vector<uint8_t> &slot_clipped) {
    slot_tokens.assign(a.n_units, std::string());
    slot_clipped.assign(a.n_units, 0);
    if (a.start <= 0) return;
    const int anchor = (a.anchor > 0 && a.anchor <= a.start) ? a.anchor : a.start;
    int qa = w.query_at(anchor - 1);
    if (qa < 0) {
        for (auto &c : slot_clipped) c = 1;
        return;
    }
    // consume the fixed prefix [anchor, start) from the read
    const int plen = a.start - anchor;
    if (plen > 0) {
        if (qa + 1 + plen > static_cast<int>(seq.size())) {
            for (auto &c : slot_clipped) c = 1;
            return;
        }
        if (hamming_prefix(std::string_view(seq).substr(qa + 1, plen), std::string_view(ref).substr(anchor, plen)) >
            a.max_mismatch + plen / 10)
            return;  // unparseable
    }
    int q0 = qa + plen;
    const int U = static_cast<int>(a.unit.size());
    const int L = a.insert_len;
    const int F = static_cast<int>(a.end_flank.size());
    const int n = static_cast<int>(seq.size());
    std::string_view sv(seq);
    auto full_unit = [&](int pos) {
        return pos >= 0 && pos + U <= n && hamming_prefix(sv.substr(pos, U), a.unit) <= a.max_mismatch;
    };
    // End flank match that tolerates the read ending inside the flank.
    auto flank_at = [&](int pos) {
        if (F == 0 || pos < 0) return false;
        int avail = std::min(n - pos, F);
        if (avail < std::min(8, F)) return false;
        return hamming_prefix(sv.substr(pos, avail), std::string_view(a.end_flank).substr(0, avail)) <=
               a.max_mismatch;
    };
    // Each slot is an optional insertion followed by a unit, or - for the
    // last slot of the array - by the end flank. Arrays shortened by
    // deletions simply reach the end flank early; their missing slots stay
    // missing.
    int q = q0 + 1;
    for (int k = 0; k < a.n_units; ++k) {
        if (full_unit(q)) {
            slot_tokens[k] = "None";
            q += U;
            continue;
        }
        if (L > 0 && full_unit(q + L)) {
            slot_tokens[k] = seq.substr(q, L);
            q += L + U;
            continue;
        }
        if (flank_at(q)) {
            slot_tokens[k] = "None";
            break;
        }
        if (L > 0 && q + L <= n && flank_at(q + L)) {
            slot_tokens[k] = seq.substr(q, L);
            break;
        }
        // Read ended or array unparseable: remaining slots missing.
        for (int j = k; j < a.n_units; ++j) slot_clipped[j] = (q + U + L > n) ? 1 : 0;
        break;
    }
}

}  // namespace

void left_align_indels(const std::string &ref, int ref_pos, const std::string &seq, std::vector<uint32_t> &cigar) {
    bool changed = true;
    int guard = 0;
    while (changed && guard++ < 64) {
        changed = false;
        int r = ref_pos, q = 0;
        for (size_t i = 0; i < cigar.size(); ++i) {
            uint32_t op = cigar[i] & 0xf;
            int len = static_cast<int>(cigar[i] >> 4);
            if ((op == kI || op == kD) && i > 0 && i + 1 < cigar.size() && (cigar[i - 1] & 0xf) == kM &&
                (cigar[i + 1] & 0xf) == kM) {
                while ((cigar[i - 1] >> 4) >= 2 && r > 0 && q > 0) {
                    bool ok;
                    if (op == kI)
                        ok = q + len - 1 < static_cast<int>(seq.size()) && seq[q + len - 1] == ref[r - 1];
                    else
                        ok = r + len - 1 < static_cast<int>(ref.size()) && seq[q - 1] == ref[r + len - 1];
                    if (!ok) break;
                    cigar[i - 1] -= 1u << 4;
                    cigar[i + 1] += 1u << 4;
                    --r;
                    --q;
                    changed = true;
                }
            }
            if (op == kM || op == kEq || op == kX) {
                r += len;
                q += len;
            } else if (op == kI || op == kS) {
                q += len;
            } else if (op == kD || op == kN) {
                r += len;
            }
        }
    }
}

bool call_read(const ConstructSpec &spec, int ref_pos, const std::vector<uint32_t> &cigar_in, const std::string &seq,
               const uint8_t *qual, ReadCall &out) {
    const size_t n_sites = spec.sites.size();
    out.intbc.clear();
    out.tokens.resize(n_sites);
    for (auto &t : out.tokens) t.clear();
    out.clipped.assign(n_sites, 0);

    const std::vector<uint32_t> *cigar = &cigar_in;
    thread_local std::vector<uint32_t> normalized;
    if (spec.normalize_indels && !spec.legacy) {
        normalized = cigar_in;
        left_align_indels(spec.reference, ref_pos, seq, normalized);
        cigar = &normalized;
    }
    thread_local Walk w;
    build_walk(ref_pos, *cigar, w);
    if (w.blocks.empty()) return false;
    const int ref_len = static_cast<int>(spec.reference.size());

    thread_local std::vector<const SiteSpec *> indel_sites;
    thread_local std::vector<size_t> indel_idx;
    indel_sites.clear();
    indel_idx.clear();
    for (size_t i = 0; i < n_sites; ++i)
        if (spec.sites[i].model == EditModel::Indel) {
            indel_sites.push_back(&spec.sites[i]);
            indel_idx.push_back(i);
        }

    bool intbc_done = false;
    if (!indel_sites.empty() || (spec.legacy && spec.intbc_start >= 0)) {
        std::string intBC;
        bool found = false;
        thread_local std::vector<std::string> indels, uncut;
        legacy_parse(spec.reference, ref_pos, w.lead_clip, w.core_ops, seq, spec.intbc_start, spec.intbc_end,
                     indel_sites, intBC, found, indels, uncut, !spec.legacy);
        if (spec.legacy) {
            out.intbc = found ? intBC : std::string();
            intbc_done = true;
        }
        for (size_t k = 0; k < indel_sites.size(); ++k) {
            const SiteSpec &s = *indel_sites[k];
            std::string tok = indels[k].empty() ? uncut[k] : indels[k];
            if (!spec.legacy && spec.clip_missing && indels[k].empty()) {
                if (!w.covered(s.position - s.window_left, s.position + s.window_right + 1)) {
                    tok.clear();
                    out.clipped[indel_idx[k]] = 1;
                }
            }
            out.tokens[indel_idx[k]] = std::move(tok);
        }
    }
    if (!intbc_done && spec.intbc_start >= 0)
        out.intbc = flank_intbc(w, seq, spec.intbc_start, spec.intbc_end, ref_len);

    thread_local std::vector<std::string> slot_tokens;
    thread_local std::vector<uint8_t> slot_clipped;
    bool array_done = false;
    for (size_t i = 0; i < n_sites; ++i) {
        const SiteSpec &s = spec.sites[i];
        bool clipped = false;
        switch (s.model) {
            case EditModel::Indel:
                break;
            case EditModel::InsertionLookup:
                out.tokens[i] = call_insertion_site(s, w, spec.reference, seq, clipped);
                out.clipped[i] = clipped;
                break;
            case EditModel::Substitution:
                out.tokens[i] = call_substitution_site(s, w, spec.reference, seq, qual, clipped);
                out.clipped[i] = clipped;
                break;
            case EditModel::SequentialInsertion:
                if (!array_done) {
                    call_array(spec.array, w, spec.reference, seq, slot_tokens, slot_clipped);
                    array_done = true;
                }
                if (s.slot >= 0 && s.slot < static_cast<int>(slot_tokens.size())) {
                    out.tokens[i] = slot_tokens[s.slot];
                    out.clipped[i] = slot_clipped[s.slot];
                }
                break;
        }
    }
    return true;
}

}  // namespace cas
