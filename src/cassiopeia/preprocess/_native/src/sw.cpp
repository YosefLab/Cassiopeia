#include "sw.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "common.hpp"

namespace cas {

// The DP keeps two rolling rows of scores and one byte of traceback pointers
// per cell (2 bits per matrix). Pointers record the predecessor matrix with
// seq-align's tie preference (gap-in-a, then gap-in-b, then match), so the
// traceback is identical to the full-matrix implementation in seq-align /
// pyseq_align while using ~12x less memory.

namespace {
constexpr int32_t kMin = std::numeric_limits<int32_t>::min() / 4;
enum : uint8_t { PM = 0, PA = 1, PB = 2, PSTOP = 3 };

inline uint8_t pick(int32_t vm, int32_t va, int32_t vb, int32_t &best) {
    // seq-align traceback checks gap_a, then gap_b, then match
    best = std::max(vm, std::max(va, vb));
    if (va == best) return PA;
    if (vb == best) return PB;
    return PM;
}

void push_op(std::vector<uint32_t> &ops, uint32_t op, uint32_t len = 1) {
    if (!ops.empty() && (ops.back() & 0xf) == op)
        ops.back() += len << 4;
    else
        ops.push_back((len << 4) | op);
}
}  // namespace

Aligner::Aligner(const Scoring &s) : s_(s) {
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            table_[x * 5 + y] = (x == 4 || y == 4) ? s.n_score : (x == y ? s.match : s.mismatch);
    open_ = -s.gap_open + 1;
    ext_ = -s.gap_extend;
}

// Fills pointers (into ptr_, row-major by query) and returns final rows.
// For local alignment also tracks the best match cell.
void Aligner::fill(const uint8_t *a, size_t la, const uint8_t *b, size_t lb, bool sw) {
    w_ = la + 1;
    h_ = lb + 1;
    if (ptr_.size() < w_ * h_) ptr_.resize(w_ * h_);
    rows_.resize(6 * w_);
    int32_t *Mp = rows_.data(), *GAp = Mp + w_, *GBp = GAp + w_;
    int32_t *Mc = GBp + w_, *GAc = Mc + w_, *GBc = GAc + w_;
    const int32_t gopen = ext_ + open_;
    const int32_t gext = ext_;
    const int32_t floor = sw ? 0 : kMin;
    // row 0
    Mp[0] = GAp[0] = GBp[0] = 0;
    for (size_t i = 1; i < w_; ++i) {
        if (sw) {
            Mp[i] = GAp[i] = GBp[i] = 0;
        } else {
            Mp[i] = kMin;
            GAp[i] = kMin;
            GBp[i] = open_ + static_cast<int32_t>(i) * ext_;
        }
    }
    best_ = 0;
    best_i_ = 0;
    best_j_ = 0;
    uint8_t *P = ptr_.data();
    for (size_t i = 0; i < w_; ++i) P[i] = PSTOP;
    // score profile: prof[base][c] = score of ref base c-1 vs query base
    prof_.resize(5 * w_);
    for (int bq = 0; bq < 5; ++bq) {
        int32_t *pr = prof_.data() + bq * w_;
        pr[0] = 0;
        for (size_t i = 0; i < la; ++i) pr[i + 1] = table_[a[i] * 5 + bq];
    }
    for (size_t j = 0; j < lb; ++j) {
        const int32_t *pr = prof_.data() + b[j] * w_;
        const bool last_row = (j == lb - 1);
        uint8_t *prow = P + (j + 1) * w_;
        if (sw) {
            Mc[0] = GAc[0] = GBc[0] = 0;
        } else {
            Mc[0] = kMin;
            GAc[0] = open_ + static_cast<int32_t>(j + 1) * ext_;
            GBc[0] = kMin;
        }
        prow[0] = PSTOP;
        // Pass 1: match and gap-in-a depend only on the previous row
        // (branch-free so the compiler can vectorize it).
        {
            const int32_t *__restrict mp = Mp;
            const int32_t *__restrict ap = GAp;
            const int32_t *__restrict bp = GBp;
            const int32_t *__restrict prf = pr;
            int32_t *__restrict mc = Mc;
            int32_t *__restrict ac = GAc;
            uint8_t *__restrict pw = prow;
            for (size_t c = 1; c <= la; ++c) {
                int32_t m0 = mp[c - 1], a0 = ap[c - 1], b0 = bp[c - 1];
                int32_t bm = std::max(m0, std::max(a0, b0));
                int32_t ia = a0 == bm, ib = (b0 == bm) & (ia ^ 1);
                int32_t pm = ia | (ib << 1);
                mc[c] = std::max(bm + prf[c], floor);
                int32_t m1 = mp[c] + gopen, a1 = ap[c] + gext, b1 = bp[c] + gopen;
                int32_t ba = std::max(m1, std::max(a1, b1));
                int32_t ja = a1 == ba, jb = (b1 == ba) & (ja ^ 1);
                int32_t pa = ja | (jb << 1);
                ac[c] = std::max(ba, floor);
                pw[c] = static_cast<uint8_t>(pm | (pa << 2));
            }
        }
        if (!sw) {
            // free end gaps in the last column for gap-in-a
            const size_t c = la;
            int32_t ba;
            uint8_t pa = pick(Mp[c], GAp[c], GBp[c], ba);
            GAc[c] = ba;
            prow[c] = static_cast<uint8_t>((prow[c] & 0x3) | (pa << 2));
        }
        // Pass 2: gap-in-b runs along the row.
        const int32_t bopen = (!sw && last_row) ? 0 : gopen;
        const int32_t bext = (!sw && last_row) ? 0 : gext;
        const int32_t bfloor = (!sw && last_row) ? kMin : floor;
        int32_t mprev = Mc[0], aprev = GAc[0], bprev = GBc[0];
        for (size_t c = 1; c <= la; ++c) {
            int32_t m1 = mprev + bopen, a1 = aprev + bopen, b1 = bprev + bext;
            int32_t bb = std::max(m1, std::max(a1, b1));
            int32_t ka = a1 == bb, kb = (b1 == bb) & (ka ^ 1);
            uint8_t pb = static_cast<uint8_t>(ka | (kb << 1));
            bb = std::max(bb, bfloor);
            GBc[c] = bb;
            prow[c] = static_cast<uint8_t>(prow[c] | (pb << 4));
            mprev = Mc[c];
            aprev = GAc[c];
            bprev = bb;
        }
        if (sw) {
            for (size_t c = 1; c <= la; ++c) {
                int32_t vm = Mc[c];
                if (vm > best_ || (vm == best_ && vm > 0 && c < best_i_)) {
                    best_ = vm;
                    best_i_ = c;
                    best_j_ = j + 1;
                }
            }
        }
        std::swap(Mp, Mc);
        std::swap(GAp, GAc);
        std::swap(GBp, GBc);
    }
    // keep the final row (needed to start the global traceback)
    last_m_.assign(Mp, Mp + w_);
    last_a_.assign(GAp, GAp + w_);
    last_b_.assign(GBp, GBp + w_);
}

AlignResult Aligner::local(std::string_view ref, std::string_view query) {
    AlignResult res;
    if (ref.empty() || query.empty()) return res;
    a_.resize(ref.size());
    b_.resize(query.size());
    for (size_t i = 0; i < ref.size(); ++i) a_[i] = base_code(ref[i]);
    for (size_t j = 0; j < query.size(); ++j) b_[j] = base_code(query[j]);
    fill(a_.data(), a_.size(), b_.data(), b_.size(), true);
    if (best_ <= 0) return res;
    // Traceback: in local mode a cell with score 0 ends the alignment. We
    // need the score of the cell we move to; recompute it from pointers by
    // walking and accumulating scores backwards from the best score.
    const int32_t gopen = ext_ + open_;
    const int32_t gext = ext_;
    size_t x = best_i_, y = best_j_;
    int mat = 0;  // 0 match, 1 gap_a, 2 gap_b
    int32_t cur = best_;
    std::vector<uint32_t> rev;
    while (cur > 0) {
        uint8_t p = ptr_[y * w_ + x];
        int next;
        int32_t pen;
        if (mat == 0) {
            push_op(rev, kM);
            pen = sub(a_[x - 1], b_[y - 1]);
            next = p & 3;
            --x;
            --y;
        } else if (mat == 1) {
            push_op(rev, kI);
            next = (p >> 2) & 3;
            pen = next == PA ? gext : gopen;
            --y;
        } else {
            push_op(rev, kD);
            next = (p >> 4) & 3;
            pen = next == PB ? gext : gopen;
            --x;
        }
        cur -= pen;
        if (next == PSTOP || x == 0 || y == 0) {
            if (cur > 0 && (x == 0 || y == 0)) cur = 0;  // reached the border
        }
        mat = next == PSTOP ? 0 : next;
    }
    std::reverse(rev.begin(), rev.end());
    res.ops = std::move(rev);
    res.score = best_;
    res.ref_start = static_cast<int>(x);
    res.query_start = static_cast<int>(y);
    res.ref_end = static_cast<int>(best_i_);
    res.query_end = static_cast<int>(best_j_);
    res.ok = true;
    return res;
}

AlignResult Aligner::global(std::string_view ref, std::string_view query) {
    AlignResult res;
    if (ref.empty() || query.empty()) return res;
    a_.resize(ref.size());
    b_.resize(query.size());
    for (size_t i = 0; i < ref.size(); ++i) a_[i] = base_code(ref[i]);
    for (size_t j = 0; j < query.size(); ++j) b_[j] = base_code(query[j]);
    fill(a_.data(), a_.size(), b_.data(), b_.size(), false);
    const size_t len_i = w_ - 1, len_j = h_ - 1;
    int mat = 0;
    int32_t cur = last_m_[len_i];
    if (last_b_[len_i] >= cur) {
        mat = 2;
        cur = last_b_[len_i];
    }
    if (last_a_[len_i] >= cur) {
        mat = 1;
        cur = last_a_[len_i];
    }
    res.score = cur;
    size_t x = len_i, y = len_j;
    std::vector<uint32_t> rev;
    while (x > 0 && y > 0) {
        uint8_t p = ptr_[y * w_ + x];
        int next;
        if (mat == 0) {
            push_op(rev, kM);
            next = p & 3;
            --x;
            --y;
        } else if (mat == 1) {
            push_op(rev, kI);
            next = (p >> 2) & 3;
            --y;
        } else {
            push_op(rev, kD);
            next = (p >> 4) & 3;
            --x;
        }
        if (next == PSTOP) break;
        mat = next;
    }
    while (y > 0) {
        push_op(rev, kI);
        --y;
    }
    while (x > 0) {
        push_op(rev, kD);
        --x;
    }
    std::reverse(rev.begin(), rev.end());
    res.ops = std::move(rev);
    res.ref_start = 0;
    res.query_start = 0;
    res.ref_end = static_cast<int>(len_i);
    res.query_end = static_cast<int>(len_j);
    res.ok = true;
    return res;
}

std::vector<uint32_t> to_bam_cigar(const AlignResult &r, int query_len, int &ref_pos) {
    std::vector<uint32_t> ops = r.ops;
    ref_pos = r.ref_start;
    int lead_clip = r.query_start, tail_clip = query_len - r.query_end;
    while (!ops.empty()) {
        uint32_t op = ops.front() & 0xf, len = ops.front() >> 4;
        if (op == kD) {
            ref_pos += static_cast<int>(len);
            ops.erase(ops.begin());
        } else if (op == kI) {
            lead_clip += static_cast<int>(len);
            ops.erase(ops.begin());
        } else {
            break;
        }
    }
    while (!ops.empty()) {
        uint32_t op = ops.back() & 0xf, len = ops.back() >> 4;
        if (op == kD) {
            ops.pop_back();
        } else if (op == kI) {
            tail_clip += static_cast<int>(len);
            ops.pop_back();
        } else {
            break;
        }
    }
    std::vector<uint32_t> out;
    if (lead_clip > 0) out.push_back((static_cast<uint32_t>(lead_clip) << 4) | kS);
    out.insert(out.end(), ops.begin(), ops.end());
    if (tail_clip > 0) out.push_back((static_cast<uint32_t>(tail_clip) << 4) | kS);
    return out;
}

}  // namespace cas
