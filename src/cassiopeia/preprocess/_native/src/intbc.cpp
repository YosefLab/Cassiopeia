#include "intbc.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "common.hpp"
#include "parallel.hpp"

namespace cas {

namespace {

struct Pattern {
    std::array<uint64_t, 256> peq{};
    int m = 0;
    explicit Pattern(const std::string &a) : m(static_cast<int>(a.size())) {
        for (int i = 0; i < m; ++i) peq[static_cast<unsigned char>(a[i])] |= 1ULL << i;
    }
};

// Hyyrö's bit-parallel global edit distance for |a| <= 64.
int myers(const Pattern &p, const std::string &b) {
    const int m = p.m;
    if (m == 0) return static_cast<int>(b.size());
    const uint64_t mask = m == 64 ? ~0ULL : ((1ULL << m) - 1);
    const uint64_t high = 1ULL << (m - 1);
    uint64_t pv = mask, mv = 0;
    int score = m;
    for (unsigned char c : b) {
        uint64_t eq = p.peq[c];
        uint64_t xv = eq | mv;
        uint64_t xh = ((((eq & pv) + pv) & mask) ^ pv) | eq;
        uint64_t ph = mv | (~(xh | pv) & mask);
        uint64_t mh = pv & xh;
        if (ph & high)
            ++score;
        else if (mh & high)
            --score;
        ph = ((ph << 1) | 1) & mask;
        mh = (mh << 1) & mask;
        pv = mh | (~(xv | ph) & mask);
        mv = ph & xv;
    }
    return score;
}

int dp_distance(const std::string &a, const std::string &b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    std::iota(prev.begin(), prev.end(), 0);
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

}  // namespace

int edit_distance(const std::string &a, const std::string &b) {
    if (a.size() <= 64 && !a.empty()) return myers(Pattern(a), b);
    if (b.size() <= 64 && !b.empty()) return myers(Pattern(b), a);
    return dp_distance(a, b);
}

std::vector<int32_t> correct_intbcs(const std::vector<std::string> &seqs, const std::vector<int64_t> &counts,
                                    const std::vector<std::string> &whitelist, int max_dist, double min_ratio,
                                    int n_threads) {
    const size_t n = seqs.size();
    std::vector<int32_t> out(n, -1);
    if (!whitelist.empty()) {
        std::unordered_map<std::string, int32_t> exact;
        for (size_t i = 0; i < whitelist.size(); ++i) exact.emplace(whitelist[i], static_cast<int32_t>(i));
        parallel_dynamic(
            n, n_threads,
            [&](int, size_t i) {
                auto it = exact.find(seqs[i]);
                if (it != exact.end()) {
                    out[i] = it->second;
                    return;
                }
                if (max_dist <= 0) return;
                const std::string &s = seqs[i];
                bool use_bits = !s.empty() && s.size() <= 64;
                Pattern pat(use_bits ? s : std::string());
                int best = max_dist + 1, best_j = -1;
                bool tie = false;
                for (size_t j = 0; j < whitelist.size(); ++j) {
                    const std::string &w = whitelist[j];
                    int ld = static_cast<int>(w.size()) - static_cast<int>(s.size());
                    if (std::abs(ld) > max_dist) continue;
                    int d = use_bits ? myers(pat, w) : dp_distance(s, w);
                    if (d < best) {
                        best = d;
                        best_j = static_cast<int>(j);
                        tie = false;
                    } else if (d == best) {
                        tie = true;
                    }
                }
                if (best_j >= 0 && !tie) out[i] = best_j;
            },
            16);
        return out;
    }
    // Population-level directional merge.
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (counts[a] != counts[b]) return counts[a] > counts[b];
        return seqs[a] < seqs[b];
    });
    std::vector<int32_t> rank(n);
    for (size_t r = 0; r < n; ++r) rank[order[r]] = static_cast<int32_t>(r);
    std::vector<int32_t> parent(n, -1);
    if (max_dist > 0) {
        parallel_dynamic(
            n, n_threads,
            [&](int, size_t r) {
                int32_t i = order[r];
                const std::string &s = seqs[i];
                double need = min_ratio * static_cast<double>(counts[i]);
                bool use_bits = !s.empty() && s.size() <= 64;
                Pattern pat(use_bits ? s : std::string());
                int best = max_dist + 1, best_j = -1;
                for (size_t k = 0; k < r; ++k) {
                    int32_t j = order[k];
                    if (static_cast<double>(counts[j]) < need) break;  // sorted by count
                    const std::string &w = seqs[j];
                    int ld = static_cast<int>(w.size()) - static_cast<int>(s.size());
                    if (std::abs(ld) > max_dist) continue;
                    int d = use_bits ? myers(pat, w) : dp_distance(s, w);
                    if (d < best) {
                        best = d;
                        best_j = j;
                    }
                }
                if (best_j >= 0) parent[i] = best_j;
            },
            64);
    }
    for (size_t i = 0; i < n; ++i) {
        int32_t r = static_cast<int32_t>(i);
        int guard = 0;
        while (parent[r] >= 0 && guard++ < 64) r = parent[r];
        out[i] = r;
    }
    return out;
}

}  // namespace cas
