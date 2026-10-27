#include "align.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

#include "bam_io.hpp"
#include "common.hpp"
#include "fastq.hpp"
#include "parallel.hpp"

namespace cas {

namespace {

// Concurrent sequence interner (ids are stable; lookups by id only after
// all insertions are done).
class SeqStore {
   public:
    static constexpr uint32_t kShards = 256;
    SeqStore() {
        for (auto &s : shards_) s = std::make_unique<Shard>();
    }
    uint32_t intern(std::string_view s) {
        uint64_t h = hash_sv(s);
        uint32_t sh = static_cast<uint32_t>(h & (kShards - 1));
        Shard &shard = *shards_[sh];
        std::lock_guard<std::mutex> lk(shard.m);
        auto it = shard.index.find(s);
        if (it != shard.index.end()) return it->second;
        uint32_t local = static_cast<uint32_t>(shard.by_id.size());
        if (local >= (1u << 24)) throw std::runtime_error("too many unique sequences");
        uint32_t id = (local << 8) | sh;
        auto res = shard.index.emplace(std::string(s), id);
        shard.by_id.push_back(&res.first->first);
        return id;
    }
    std::string_view get(uint32_t id) const { return *shards_[id & (kShards - 1)]->by_id[id >> 8]; }
    size_t size() const {
        size_t n = 0;
        for (auto &s : shards_) n += s->by_id.size();
        return n;
    }

   private:
    struct Shard {
        std::mutex m;
        std::unordered_map<std::string, uint32_t, SvHash, SvEq> index;
        std::vector<const std::string *> by_id;
    };
    std::array<std::unique_ptr<Shard>, kShards> shards_;
};

struct MolKey {
    uint64_t cb, umi;
    bool operator==(const MolKey &o) const { return cb == o.cb && umi == o.umi; }
};
struct MolKeyHash {
    size_t operator()(const MolKey &k) const { return mix64(k.cb * 0x9e3779b97f4a7c15ULL ^ mix64(k.umi)); }
};
struct SeqCount {
    uint32_t seq;
    uint32_t count;
};
struct MolAcc {
    std::vector<SeqCount> seqs;
    uint32_t reads = 0;
    void add(uint32_t seq, uint32_t n = 1) {
        reads += n;
        for (auto &s : seqs)
            if (s.seq == seq) {
                s.count += n;
                return;
            }
        seqs.push_back({seq, n});
    }
};
struct CbStat {
    uint64_t reads = 0;
    std::array<uint64_t, 32> qsum{};
};

struct Shard {
    std::unordered_map<MolKey, MolAcc, MolKeyHash> mols;
    std::unordered_map<uint64_t, CbStat> cbs;
    int64_t short_reads = 0, low_quality = 0, n_reads = 0;
};

struct MolRec {
    uint64_t cb, umi;
    uint32_t reads;
    uint32_t n_umis;
    std::vector<SeqCount> seqs;
};

struct OutRec {
    uint64_t cb, umi;
    std::string seq;
    uint32_t reads;
    uint32_t n_umis;
    float frac;
    int32_t aln = -1;
};

struct AlnOut {
    int score = 0;
    int pos = -1;
    bool reverse = false;
    bool mapped = false;
    std::vector<uint32_t> cigar;
};

bool slice(const Segment &s, std::string_view seq, std::string_view qual, std::string_view &sseq,
           std::string_view &squal) {
    int len = static_cast<int>(seq.size());
    int end = s.end < 0 ? len : s.end;
    if (s.start > len || end > len || end < s.start) return false;
    sseq = seq.substr(s.start, end - s.start);
    squal = qual.substr(s.start, end - s.start);
    return true;
}

std::vector<uint64_t> load_whitelist(const std::string &path, int len) {
    std::vector<uint64_t> wl;
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) throw std::runtime_error("cannot open whitelist: " + path);
    std::vector<char> line(4096);
    while (gzgets(gz, line.data(), static_cast<int>(line.size()))) {
        std::string_view s(line.data());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.remove_suffix(1);
        size_t tab = s.find_first_of("\t ,");
        if (tab != std::string_view::npos) s = s.substr(0, tab);
        size_t dash = s.find('-');
        if (dash != std::string_view::npos) s = s.substr(0, dash);
        if (static_cast<int>(s.size()) != len) continue;
        uint64_t v;
        if (pack2bit(s, v)) wl.push_back(v);
    }
    gzclose(gz);
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
    return wl;
}

// Two-column barcode translation (e.g. 10x feature barcoding): column 1 is
// the reported (gene expression) barcode, column 2 the barcode in the reads.
// Returns read barcode -> reported barcode; `wl` receives the read barcodes.
std::unordered_map<uint64_t, uint64_t> load_translation(const std::string &path, int len, std::vector<uint64_t> &wl) {
    std::unordered_map<uint64_t, uint64_t> map;
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) throw std::runtime_error("cannot open barcode translation: " + path);
    std::vector<char> line(4096);
    while (gzgets(gz, line.data(), static_cast<int>(line.size()))) {
        std::string_view s(line.data());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.remove_suffix(1);
        size_t sep = s.find_first_of("\t ,");
        if (sep == std::string_view::npos) continue;
        std::string_view a = s.substr(0, sep), b = s.substr(sep + 1);
        uint64_t va, vb;
        if (static_cast<int>(a.size()) != len || static_cast<int>(b.size()) != len) continue;
        if (pack2bit(a, va) && pack2bit(b, vb)) {
            map.emplace(vb, va);
            wl.push_back(vb);
        }
    }
    gzclose(gz);
    if (map.empty()) throw std::runtime_error("no barcode pairs of length " + std::to_string(len) + " in " + path);
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
    return map;
}

bool in_sorted(const std::vector<uint64_t> &v, uint64_t x) { return std::binary_search(v.begin(), v.end(), x); }

// Directional (UMI-tools) clustering. Returns root index per UMI.
std::vector<int> directional_clusters(const std::vector<uint64_t> &umis, const std::vector<uint32_t> &counts,
                                      int umi_len, int max_dist) {
    const size_t n = umis.size();
    std::vector<int> root(n, -1);
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (counts[a] != counts[b]) return counts[a] > counts[b];
        return umis[a] < umis[b];
    });
    std::unordered_map<uint64_t, uint32_t> index;
    const bool enumerate = n > 256 && max_dist <= 2;
    if (enumerate) {
        index.reserve(n * 2);
        for (uint32_t i = 0; i < n; ++i) index.emplace(umis[i], i);
    }
    std::vector<uint32_t> stack, nbrs;
    auto neighbours = [&](uint32_t u) {
        nbrs.clear();
        if (!enumerate) {
            for (uint32_t v = 0; v < n; ++v)
                if (v != u && packed_hamming(umis[u], umis[v]) <= max_dist) nbrs.push_back(v);
            return;
        }
        const uint64_t x = umis[u];
        for (int p = 0; p < umi_len; ++p) {
            int sh = 2 * (umi_len - 1 - p);
            uint64_t cur = (x >> sh) & 3;
            for (uint64_t b = 0; b < 4; ++b) {
                if (b == cur) continue;
                uint64_t y = (x & ~(3ULL << sh)) | (b << sh);
                auto it = index.find(y);
                if (it != index.end()) nbrs.push_back(it->second);
                if (max_dist >= 2) {
                    for (int p2 = p + 1; p2 < umi_len; ++p2) {
                        int sh2 = 2 * (umi_len - 1 - p2);
                        uint64_t cur2 = (y >> sh2) & 3;
                        for (uint64_t b2 = 0; b2 < 4; ++b2) {
                            if (b2 == cur2) continue;
                            uint64_t z = (y & ~(3ULL << sh2)) | (b2 << sh2);
                            auto it2 = index.find(z);
                            if (it2 != index.end()) nbrs.push_back(it2->second);
                        }
                    }
                }
            }
        }
    };
    for (uint32_t idx : order) {
        if (root[idx] >= 0) continue;
        root[idx] = static_cast<int>(idx);
        stack.assign(1, idx);
        while (!stack.empty()) {
            uint32_t u = stack.back();
            stack.pop_back();
            neighbours(u);
            for (uint32_t v : nbrs) {
                if (root[v] >= 0) continue;
                if (static_cast<uint64_t>(counts[u]) + 1 >= 2ULL * counts[v]) {
                    root[v] = static_cast<int>(idx);
                    stack.push_back(v);
                }
            }
        }
    }
    return root;
}

struct DedupOut {
    std::string seq;
    uint32_t reads;
    uint32_t rescued;  // reads joined by the alignment-based merge
};

inline bool is_hq(char c) { return c >= 'A' && c <= 'Z'; }
inline char upper(char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
inline char lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

// Count-weighted consensus where high-quality (uppercase) bases dominate.
std::string weighted_consensus(const std::vector<std::string_view> &seqs, const std::vector<uint32_t> &counts,
                               size_t len) {
    std::vector<std::array<double, 5>> votes(len, std::array<double, 5>{0, 0, 0, 0, 0});
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (seqs[i].size() != len) continue;
        for (size_t p = 0; p < len; ++p)
            votes[p][base_code(seqs[i][p])] += counts[i] * (is_hq(seqs[i][p]) ? 1.0 : 0.01);
    }
    std::string out(len, 'N');
    for (size_t p = 0; p < len; ++p) {
        int best = 4;
        double bv = 0;
        for (int b = 0; b < 5; ++b)
            if (votes[p][b] > bv) {
                bv = votes[p][b];
                best = b;
            }
        out[p] = code_base(static_cast<uint8_t>(best));
    }
    return out;
}

// Lowercases the consensus positions the legacy collapse calls low quality:
// high quality needs a majority of the cluster's reads to carry the
// consensus base, at least one of them a high-quality (uppercase) base.
void mask_consensus(std::string &cons, const std::vector<std::string_view> &seqs,
                    const std::vector<uint32_t> &counts) {
    uint64_t total = 0;
    for (uint32_t c : counts) total += c;
    for (size_t p = 0; p < cons.size(); ++p) {
        uint64_t agree = 0;
        bool hq = false;
        for (size_t i = 0; i < seqs.size(); ++i)
            if (p < seqs[i].size() && upper(seqs[i][p]) == cons[p]) {
                agree += counts[i];
                hq = hq || is_hq(seqs[i][p]);
            }
        if (!hq || 2 * agree <= total) cons[p] = lower(cons[p]);
    }
}

constexpr int32_t kNegInf = std::numeric_limits<int32_t>::min() / 2;

struct MergeScratch {
    std::vector<int32_t> score;
    std::vector<uint8_t> dir;  // 0 diagonal, 1 left (deletion), 2 up (insertion)
};

// Can cluster `b` be merged into the biggest cluster `a`? Reproduces the
// legacy test (hits `sw.global_alignment`): the alignment is anchored at both
// starts and ends at the best cell of the last row or column, scoring +2 per
// match (N matches anything), -1 per mismatch and -5 per gap base, ties
// preferring diagonal, then deletion, then insertion. Mergeable if it has at
// most `max_indels` gap bases and `max_mm` mismatches between high-quality
// (uppercase) bases.
//
// Banded to |i - j| <= W. A path leaving the band has more than W gap bases
// and so scores at most 2 * min(n, m) - 5 * (W + 1); once the banded optimum
// beats that bound, the banded traceback is the full one.
bool clusters_mergeable(std::string_view a, std::string_view b, int max_indels, int max_mm) {
    const int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    if (n == 0 || m == 0) return false;
    auto hq_mismatch = [&](int i, int j) {
        uint8_t x = base_code(a[i]), y = base_code(b[j]);
        return x != 4 && y != 4 && x != y && is_hq(a[i]) && is_hq(b[j]);
    };
    // quick reject: every path within the indel band has too many mismatches
    {
        const int D = std::max(max_indels, 0), bw = 2 * D + 1;
        std::vector<int> prev(bw, INT32_MAX / 2), cur(bw);
        for (int j = 0; j <= std::min(m, D); ++j) prev[j + D] = 0;
        int best = (m <= D) ? 0 : INT32_MAX / 2;
        for (int i = 1; i <= n; ++i) {
            std::fill(cur.begin(), cur.end(), INT32_MAX / 2);
            for (int j = std::max(0, i - D); j <= std::min(m, i + D); ++j) {
                int k = j - i + D, v = INT32_MAX / 2;
                if (j == 0) v = 0;
                else {
                    v = prev[k] + (hq_mismatch(i - 1, j - 1) ? 1 : 0);             // diagonal
                    if (k - 1 >= 0) v = std::min(v, cur[k - 1]);                    // left
                }
                if (k + 1 < bw) v = std::min(v, prev[k + 1]);                       // up
                cur[k] = v;
                if (j == m || i == n) best = std::min(best, v);
            }
            std::swap(prev, cur);
        }
        if (best > max_mm) return false;
    }
    thread_local MergeScratch s;
    int W = std::max(max_indels, 0) + 4;
    for (;;) {
        W = std::min(W, std::max(n, m));
        const int bw = 2 * W + 1;
        s.score.assign(static_cast<size_t>(n + 1) * bw, kNegInf);
        s.dir.assign(static_cast<size_t>(n + 1) * bw, 0);
        auto at = [&](int i, int j) { return static_cast<size_t>(i) * bw + (j - i + W); };
        s.score[at(0, 0)] = 0;
        for (int j = 1; j <= std::min(m, W); ++j) {
            s.score[at(0, j)] = -5 * j;
            s.dir[at(0, j)] = 1;
        }
        for (int i = 1; i <= n; ++i) {
            for (int j = std::max(0, i - W); j <= std::min(m, i + W); ++j) {
                if (j == 0) {
                    s.score[at(i, 0)] = -5 * i;
                    s.dir[at(i, 0)] = 2;
                    continue;
                }
                uint8_t x = base_code(a[i - 1]), y = base_code(b[j - 1]);
                int32_t diag = s.score[at(i - 1, j - 1)] + ((x == 4 || y == 4 || x == y) ? 2 : -1);
                int32_t left = (j - 1 >= i - W) ? s.score[at(i, j - 1)] - 5 : kNegInf;
                int32_t up = (j - (i - 1) <= W) ? s.score[at(i - 1, j)] - 5 : kNegInf;
                int32_t best = std::max(diag, std::max(left, up));
                s.score[at(i, j)] = best;
                s.dir[at(i, j)] = best == diag ? 0 : (best == left ? 1 : 2);
            }
        }
        // best end on the last column (first row on ties) or the last row (first column)
        int end_i = -1, end_j = -1;
        int32_t col_best = kNegInf, row_best = kNegInf;
        int col_i = -1, row_j = -1;
        for (int i = std::max(0, m - W); i <= std::min(n, m + W); ++i)
            if (s.score[at(i, m)] > col_best) {
                col_best = s.score[at(i, m)];
                col_i = i;
            }
        for (int j = std::max(0, n - W); j <= std::min(m, n + W); ++j)
            if (s.score[at(n, j)] > row_best) {
                row_best = s.score[at(n, j)];
                row_j = j;
            }
        if (col_best > row_best) {
            end_i = col_i;
            end_j = m;
        } else {
            end_i = n;
            end_j = row_j;
        }
        int i = end_i, j = end_j, gaps = 0, mm = 0;
        while (i > 0 || j > 0) {
            uint8_t d = s.dir[at(i, j)];
            if (d == 0) {
                mm += hq_mismatch(i - 1, j - 1);
                --i;
                --j;
            } else {
                ++gaps;
                if (d == 1) --j;
                else --i;
            }
            // Failing in the band means failing outright: if the full optimum
            // stays in the band it is this path, otherwise it has > W gap bases.
            if (gaps > max_indels || mm > max_mm) return false;
        }
        const int32_t best = std::max(col_best, row_best);
        const int need = (2 * std::min(n, m) - best) / 5;  // band width that makes a pass exact
        if (W >= need || W >= std::max(n, m)) return true;
        W = need;
    }
}

// One consensus sequence per UMI, following the legacy collapse
// (form_collapsed_clusters). Low-quality bases are stored lowercase.
// 1. Clusters are seeded with the most frequent remaining sequence (or the
//    consensus of the remaining reads when none repeats) and take the reads
//    within `max_mm` high-quality mismatches of the seed, until no read joins;
//    reads left over are singletons.
// 2. Clusters are merged into the biggest when an alignment shows at most
//    `max_indels` gap bases and `max_mm` high-quality mismatches (so reads with
//    a sequencing indel still count). As in the legacy code, clusters with as
//    many reads as the biggest are not merged. `max_indels` < 0 skips this.
// Returns the biggest cluster's consensus and read count.
DedupOut dedup_molecule(const MolRec &m, const SeqStore &store, int max_mm, int max_indels) {
    constexpr size_t kMaxClusters = 64;  // bounds the seeding loop on pathological UMIs
    struct Item {
        std::string_view seq;
        uint32_t count;
    };
    std::vector<Item> items;
    items.reserve(m.seqs.size());
    for (auto &s : m.seqs) items.push_back({store.get(s.seq), s.count});
    // deterministic order (interned ids depend on thread timing)
    std::sort(items.begin(), items.end(), [](const Item &x, const Item &y) {
        return x.count != y.count ? x.count > y.count : x.seq < y.seq;
    });
    struct Cluster {
        std::string seq;  // lowercase = low-confidence position
        uint32_t reads;
    };
    std::vector<Cluster> clusters;
    std::vector<char> used(items.size(), 0);
    uint64_t left = 0;
    for (auto &it : items) left += it.count;
    std::unordered_map<std::string, uint32_t> up_counts;
    std::unordered_map<size_t, uint32_t> len_counts;
    std::vector<std::string_view> seqs, members;
    std::vector<uint32_t> counts, member_counts;
    std::vector<size_t> member_idx;
    while (left > 1 && clusters.size() < kMaxClusters) {
        up_counts.clear();
        len_counts.clear();
        seqs.clear();
        counts.clear();
        std::string best_up;
        uint32_t best_n = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            if (used[i]) continue;
            std::string u(items[i].seq);
            for (auto &ch : u) ch = upper(ch);
            uint32_t c = (up_counts[u] += items[i].count);
            len_counts[u.size()] += items[i].count;
            if (c > best_n || (c == best_n && u < best_up)) {
                best_n = c;
                best_up = std::move(u);
            }
            seqs.push_back(items[i].seq);
            counts.push_back(items[i].count);
        }
        std::string seed;
        if (best_n > 1) {
            seed = std::move(best_up);
        } else {
            size_t mode_len = 0;
            uint32_t mode_n = 0;
            for (auto &kv : len_counts)
                if (kv.second > mode_n || (kv.second == mode_n && kv.first > mode_len)) {
                    mode_n = kv.second;
                    mode_len = kv.first;
                }
            seed = weighted_consensus(seqs, counts, mode_len);
        }
        members.clear();
        member_counts.clear();
        member_idx.clear();
        uint32_t reads = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            if (used[i] || items[i].seq.size() != seed.size()) continue;
            std::string_view s = items[i].seq;
            int d = 0;
            for (size_t p = 0; p < s.size() && d <= max_mm; ++p)
                if (is_hq(s[p]) && upper(s[p]) != seed[p]) ++d;
            if (d <= max_mm) {
                members.push_back(s);
                member_counts.push_back(items[i].count);
                member_idx.push_back(i);
                reads += items[i].count;
            }
        }
        if (members.empty()) break;
        for (size_t i : member_idx) used[i] = 1;
        left -= reads;
        std::string cons;
        if (members.size() == 1) {
            cons.assign(members[0]);
            for (auto &ch : cons) ch = upper(ch);
        } else {
            cons = weighted_consensus(members, member_counts, seed.size());
        }
        mask_consensus(cons, members, member_counts);
        clusters.push_back({std::move(cons), reads});
    }
    for (size_t i = 0; i < items.size(); ++i)
        if (!used[i]) clusters.push_back({std::string(items[i].seq), items[i].count});
    std::stable_sort(clusters.begin(), clusters.end(),
                     [](const Cluster &x, const Cluster &y) { return x.reads > y.reads; });
    const Cluster &big = clusters[0];
    uint32_t total = big.reads;
    if (max_indels >= 0)
        for (size_t k = 1; k < clusters.size(); ++k) {
            if (clusters[k].reads == total) continue;
            if (clusters_mergeable(big.seq, clusters[k].seq, max_indels, max_mm)) total += clusters[k].reads;
        }
    std::string out(big.seq);
    for (auto &ch : out) ch = upper(ch);
    return DedupOut{std::move(out), total, total - big.reads};
}

std::string cigar_string(const std::vector<uint32_t> &ops) {
    std::string s;
    for (uint32_t c : ops) {
        s += std::to_string(c >> 4);
        s.push_back("MIDNSHP=X"[c & 0xf]);
    }
    return s;
}

}  // namespace

bool merge_check(const std::string &a, const std::string &b, int max_indels, int max_mismatches) {
    return clusters_mergeable(a, b, max_indels, max_mismatches);
}

std::vector<SimpleAlignment> align_sequences(const std::string &ref, const std::vector<std::string> &queries,
                                             const Scoring &scoring, bool global, int n_threads) {
    std::vector<SimpleAlignment> out(queries.size());
    std::vector<std::unique_ptr<Aligner>> aligners;
    int t = std::max(1, n_threads);
    for (int i = 0; i < t; ++i) aligners.push_back(std::make_unique<Aligner>(scoring));
    parallel_dynamic(
        queries.size(), t,
        [&](int tid, size_t i) {
            AlignResult r = global ? aligners[tid]->global(ref, queries[i]) : aligners[tid]->local(ref, queries[i]);
            out[i] = SimpleAlignment{r.score, r.ref_start, r.query_start, cigar_string(r.ops)};
        },
        64);
    return out;
}

AlignSummary align_fastqs(const AlignOptions &opt) {
    const int T = std::max(1, opt.n_threads);
    const LayoutSpec &L = opt.layout;
    int cb_len = 0, umi_len = 0;
    for (auto &s : L.cb) {
        if (s.end < 0) throw std::runtime_error("cell barcode segments need an explicit end");
        cb_len += s.end - s.start;
    }
    for (auto &s : L.umi) {
        if (s.end < 0) throw std::runtime_error("UMI segments need an explicit end");
        umi_len += s.end - s.start;
    }
    if (cb_len > 32 || umi_len > 32) throw std::runtime_error("barcodes longer than 32bp are not supported");
    const bool has_cb = cb_len > 0, has_umi = umi_len > 0;
    const bool dedup_umi = opt.dedup_umi && has_umi;
    const char qoff = 33;

    SeqStore store;
    const int W = T;
    std::vector<std::unique_ptr<Shard>> shards;
    for (int i = 0; i < W; ++i) shards.push_back(std::make_unique<Shard>());

    auto t_phase = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, int64_t>> phase_ms;
    // ---------------------------------------------------------------- read
    for (const auto &lane : opt.lanes) {
        if (static_cast<int>(lane.size()) != L.n_reads)
            throw std::runtime_error("each lane needs " + std::to_string(L.n_reads) + " FASTQ files");
        const int R = L.n_reads;
        std::vector<std::unique_ptr<FastqReader>> readers;
        std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<FastqBatch>>>> queues;
        for (int r = 0; r < R; ++r) {
            readers.push_back(std::make_unique<FastqReader>(lane[r]));
            queues.push_back(std::make_unique<BoundedQueue<std::shared_ptr<FastqBatch>>>(4));
        }
        std::vector<std::exception_ptr> errors(R);
        std::vector<std::thread> producers;
        for (int r = 0; r < R; ++r) {
            producers.emplace_back([&, r]() {
                try {
                    while (true) {
                        auto b = std::make_shared<FastqBatch>();
                        size_t n = readers[r]->read_batch(*b, opt.batch_size);
                        if (n == 0) break;
                        queues[r]->push(std::move(b));
                    }
                } catch (...) {
                    errors[r] = std::current_exception();
                }
                queues[r]->finish();
            });
        }
        std::exception_ptr consumer_error;
        try {
            std::vector<std::shared_ptr<FastqBatch>> cur(R);
            while (true) {
                int ended = 0;
                for (int r = 0; r < R; ++r)
                    if (!queues[r]->pop(cur[r])) ++ended;
                if (ended == R) break;
                if (ended > 0) throw std::runtime_error("FASTQ files in a lane have different numbers of reads");
                size_t n = cur[0]->size();
                for (int r = 1; r < R; ++r)
                    if (cur[r]->size() != n)
                        throw std::runtime_error("FASTQ files in a lane have different numbers of reads");
                parallel_blocks(static_cast<size_t>(W), W, [&](int, size_t wb, size_t we) {
                    for (size_t w = wb; w < we; ++w) {
                        Shard &sh = *shards[w];
                        char cbuf[64], ubuf[64];
                        std::string masked;
                        uint8_t cq[64];
                        for (size_t i = 0; i < n; ++i) {
                            bool ok = true;
                            int cpos = 0, upos = 0;
                            bool lowq = false;
                            for (const auto &s : L.cb) {
                                std::string_view ss, qq;
                                if (!slice(s, cur[s.read]->seq(i), cur[s.read]->qual(i), ss, qq)) {
                                    ok = false;
                                    break;
                                }
                                for (size_t k = 0; k < ss.size(); ++k) {
                                    cbuf[cpos] = ss[k];
                                    cq[cpos] = static_cast<uint8_t>(qq[k] - qoff);
                                    if (cq[cpos] < opt.min_bc_qual || base_code(ss[k]) > 3) lowq = true;
                                    ++cpos;
                                }
                            }
                            if (ok)
                                for (const auto &s : L.umi) {
                                    std::string_view ss, qq;
                                    if (!slice(s, cur[s.read]->seq(i), cur[s.read]->qual(i), ss, qq)) {
                                        ok = false;
                                        break;
                                    }
                                    for (size_t k = 0; k < ss.size(); ++k) {
                                        ubuf[upos++] = ss[k];
                                        if (static_cast<uint8_t>(qq[k] - qoff) < opt.min_bc_qual ||
                                            base_code(ss[k]) > 3)
                                            lowq = true;
                                    }
                                }
                            std::string_view cdna, cdq;
                            if (ok && !slice(L.cdna, cur[L.cdna.read]->seq(i), cur[L.cdna.read]->qual(i), cdna, cdq))
                                ok = false;
                            if (ok && cdna.empty()) ok = false;
                            if (!ok || lowq) {
                                if (w == 0) {
                                    if (!ok)
                                        sh.short_reads++;
                                    else
                                        sh.low_quality++;
                                }
                                continue;
                            }
                            uint64_t cb = 0, umi = 0;
                            if (has_cb) pack2bit(std::string_view(cbuf, cpos), cb);
                            if (has_umi)
                                pack2bit(std::string_view(ubuf, upos), umi);
                            else
                                umi = hash_sv(cdna);
                            MolKey key{cb, umi};
                            if (MolKeyHash()(key) % W != w) continue;
                            sh.n_reads++;
                            if (has_cb) {
                                CbStat &st = sh.cbs[cb];
                                st.reads++;
                                for (int k = 0; k < cpos; ++k) st.qsum[k] += cq[k];
                            }
                            // low-quality bases are stored lowercase (used by the UMI consensus)
                            masked.assign(cdna.data(), cdna.size());
                            if (opt.min_base_qual > 0)
                                for (size_t k = 0; k < masked.size(); ++k)
                                    if (static_cast<uint8_t>(cdq[k] - qoff) < opt.min_base_qual)
                                        masked[k] = static_cast<char>(std::tolower(static_cast<unsigned char>(masked[k])));
                            uint32_t sid = store.intern(masked);
                            sh.mols[key].add(sid);
                        }
                    }
                });
            }
        } catch (...) {
            consumer_error = std::current_exception();
            for (auto &q : queues) q->finish();
        }
        for (auto &p : producers) p.join();
        if (consumer_error) std::rethrow_exception(consumer_error);
        for (auto &e : errors)
            if (e) std::rethrow_exception(e);
    }

    int64_t n_reads = 0, n_short = 0, n_lowq = 0;
    for (auto &s : shards) {
        n_reads += s->n_reads;
        n_short += s->short_reads;
        n_lowq += s->low_quality;
    }

    {
        auto now = std::chrono::steady_clock::now();
        phase_ms.emplace_back("ms_read", std::chrono::duration_cast<std::chrono::milliseconds>(now - t_phase).count());
        t_phase = now;
    }
    // --------------------------------------------------- CB correction
    std::unordered_map<uint64_t, uint64_t> cb_map;  // raw -> corrected (UINT64_MAX = invalid)
    int64_t reads_exact = 0, reads_corrected = 0, reads_invalid = 0;
    const uint64_t kInvalid = ~uint64_t(0);
    if (has_cb) {
        std::unordered_map<uint64_t, CbStat> all;
        for (auto &s : shards)
            for (auto &kv : s->cbs) {
                CbStat &a = all[kv.first];
                a.reads += kv.second.reads;
                for (int k = 0; k < cb_len; ++k) a.qsum[k] += kv.second.qsum[k];
            }
        cb_map.reserve(all.size() * 2);
        if (!opt.whitelist.empty() || !opt.cb_translation.empty()) {
            std::vector<uint64_t> wl;
            std::unordered_map<uint64_t, uint64_t> translate;
            if (!opt.cb_translation.empty())
                translate = load_translation(opt.cb_translation, cb_len, wl);
            else
                wl = load_whitelist(opt.whitelist, cb_len);
            if (wl.empty()) throw std::runtime_error("no barcodes of length " + std::to_string(cb_len) + " in whitelist");
            std::unordered_map<uint64_t, uint64_t> exact;
            for (auto &kv : all)
                if (in_sorted(wl, kv.first)) exact[kv.first] = kv.second.reads;
            for (auto &kv : all) {
                const uint64_t raw = kv.first;
                if (exact.count(raw)) {
                    cb_map[raw] = translate.empty() ? raw : translate.at(raw);
                    reads_exact += kv.second.reads;
                    continue;
                }
                double total = 0, best = 0;
                uint64_t best_cb = kInvalid;
                for (int p = 0; p < cb_len; ++p) {
                    int sh = 2 * (cb_len - 1 - p);
                    uint64_t curb = (raw >> sh) & 3;
                    double q = static_cast<double>(kv.second.qsum[p]) / static_cast<double>(kv.second.reads);
                    double perr = std::pow(10.0, -q / 10.0);
                    for (uint64_t b = 0; b < 4; ++b) {
                        if (b == curb) continue;
                        uint64_t cand = (raw & ~(3ULL << sh)) | (b << sh);
                        if (!in_sorted(wl, cand)) continue;
                        auto it = exact.find(cand);
                        double prior = (it == exact.end() ? 0.0 : static_cast<double>(it->second)) + 1.0;
                        double lik = prior * perr;
                        total += lik;
                        if (lik > best) {
                            best = lik;
                            best_cb = cand;
                        }
                    }
                }
                if (best_cb != kInvalid && total > 0 && best / total >= opt.cb_confidence) {
                    cb_map[raw] = translate.empty() ? best_cb : translate.at(best_cb);
                    reads_corrected += kv.second.reads;
                } else {
                    cb_map[raw] = kInvalid;
                    reads_invalid += kv.second.reads;
                }
            }
        } else {
            for (auto &kv : all) {
                cb_map[kv.first] = kv.first;
                reads_exact += kv.second.reads;
            }
        }
    }

    {
        auto now = std::chrono::steady_clock::now();
        phase_ms.emplace_back("ms_correct_cells", std::chrono::duration_cast<std::chrono::milliseconds>(now - t_phase).count());
        t_phase = now;
    }
    // ------------------------------------------- re-bucket by corrected CB
    const size_t B = static_cast<size_t>(W);
    std::vector<std::vector<std::vector<MolRec>>> moved(W, std::vector<std::vector<MolRec>>(B));
    int64_t raw_molecules = 0;
    for (auto &s : shards) raw_molecules += static_cast<int64_t>(s->mols.size());
    parallel_blocks(static_cast<size_t>(W), W, [&](int, size_t wb, size_t we) {
        for (size_t w = wb; w < we; ++w) {
            Shard &sh = *shards[w];
            for (auto &kv : sh.mols) {
                uint64_t cb = kv.first.cb;
                if (has_cb) {
                    auto it = cb_map.find(cb);
                    if (it == cb_map.end() || it->second == kInvalid) continue;
                    cb = it->second;
                }
                size_t bucket = mix64(cb) % B;
                moved[w][bucket].push_back(MolRec{cb, kv.first.umi, kv.second.reads, 1, std::move(kv.second.seqs)});
            }
            std::unordered_map<MolKey, MolAcc, MolKeyHash>().swap(sh.mols);
            std::unordered_map<uint64_t, CbStat>().swap(sh.cbs);
        }
    });
    shards.clear();

    {
        auto now = std::chrono::steady_clock::now();
        phase_ms.emplace_back("ms_rebucket", std::chrono::duration_cast<std::chrono::milliseconds>(now - t_phase).count());
        t_phase = now;
    }
    // --------------------------- merge, UMI correction and deduplication
    std::vector<std::vector<OutRec>> outs(B);
    std::vector<std::vector<std::pair<uint64_t, std::pair<int64_t, int64_t>>>> cell_counts(B);
    std::vector<std::vector<int64_t>> hists(B, std::vector<int64_t>(1001, 0));
    std::vector<int64_t> merged_umis(B, 0), molecules(B, 0), rescued(B, 0);
    parallel_dynamic(B, W, [&](int, size_t b) {
        std::vector<MolRec> recs;
        for (int w = 0; w < W; ++w) {
            auto &v = moved[w][b];
            for (auto &m : v) recs.push_back(std::move(m));
            std::vector<MolRec>().swap(v);
        }
        std::sort(recs.begin(), recs.end(), [](const MolRec &a, const MolRec &x) {
            if (a.cb != x.cb) return a.cb < x.cb;
            return a.umi < x.umi;
        });
        // merge identical (cb, umi) from different raw barcodes
        std::vector<MolRec> merged;
        for (auto &m : recs) {
            if (!merged.empty() && merged.back().cb == m.cb && merged.back().umi == m.umi) {
                MolRec &t = merged.back();
                t.reads += m.reads;
                for (auto &s : m.seqs) {
                    bool found = false;
                    for (auto &x : t.seqs)
                        if (x.seq == s.seq) {
                            x.count += s.count;
                            found = true;
                            break;
                        }
                    if (!found) t.seqs.push_back(s);
                }
            } else {
                merged.push_back(std::move(m));
            }
        }
        recs.clear();
        auto &out = outs[b];
        size_t i = 0;
        while (i < merged.size()) {
            size_t j = i;
            while (j < merged.size() && merged[j].cb == merged[i].cb) ++j;
            // UMI correction within this cell
            if (has_umi && opt.umi_max_dist > 0 && j - i > 1) {
                std::vector<uint64_t> u(j - i);
                std::vector<uint32_t> c(j - i);
                for (size_t k = i; k < j; ++k) {
                    u[k - i] = merged[k].umi;
                    c[k - i] = merged[k].reads;
                }
                std::vector<int> root = directional_clusters(u, c, umi_len, opt.umi_max_dist);
                for (size_t k = i; k < j; ++k) {
                    int r = root[k - i];
                    if (r == static_cast<int>(k - i)) continue;
                    MolRec &dst = merged[i + r];
                    MolRec &src = merged[k];
                    dst.reads += src.reads;
                    dst.n_umis += src.n_umis;
                    for (auto &s : src.seqs) {
                        bool found = false;
                        for (auto &x : dst.seqs)
                            if (x.seq == s.seq) {
                                x.count += s.count;
                                found = true;
                                break;
                            }
                        if (!found) dst.seqs.push_back(s);
                    }
                    src.reads = 0;
                    merged_umis[b]++;
                }
            }
            int64_t cell_reads = 0, cell_umis = 0;
            for (size_t k = i; k < j; ++k) {
                MolRec &m = merged[k];
                if (m.reads == 0) continue;
                cell_reads += m.reads;
                cell_umis++;
                molecules[b]++;
                hists[b][std::min<uint32_t>(m.reads, 1000)]++;
                std::sort(m.seqs.begin(), m.seqs.end(), [](const SeqCount &a, const SeqCount &x) {
                    if (a.count != x.count) return a.count > x.count;
                    return a.seq < x.seq;
                });
                if (dedup_umi) {
                    DedupOut d = dedup_molecule(m, store, opt.max_mismatches, opt.max_indels);
                    rescued[b] += d.rescued;
                    out.push_back(OutRec{m.cb, m.umi, std::move(d.seq), d.reads, m.n_umis,
                                         static_cast<float>(d.reads) / static_cast<float>(m.reads)});
                } else {
                    // one record per distinct sequence (case-insensitive)
                    std::unordered_map<std::string, uint32_t> upper;
                    for (auto &s : m.seqs) {
                        std::string u(store.get(s.seq));
                        for (auto &ch : u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                        upper[u] += s.count;
                    }
                    for (auto &kv : upper)
                        out.push_back(OutRec{m.cb, m.umi, kv.first, kv.second, m.n_umis,
                                             static_cast<float>(kv.second) / static_cast<float>(m.reads)});
                }
            }
            if (has_cb) cell_counts[b].push_back({merged[i].cb, {cell_reads, cell_umis}});
            i = j;
        }
    });
    const size_t n_unique_input = store.size();

    {
        auto now = std::chrono::steady_clock::now();
        phase_ms.emplace_back("ms_collapse", std::chrono::duration_cast<std::chrono::milliseconds>(now - t_phase).count());
        t_phase = now;
    }
    // ------------------------------------------------------------ align
    std::vector<OutRec *> all;
    for (auto &o : outs)
        for (auto &r : o) all.push_back(&r);
    std::unordered_map<std::string_view, int32_t, SvHash, SvEq> uniq;
    std::vector<std::string_view> useqs;
    uniq.reserve(all.size());
    for (OutRec *r : all) {
        auto it = uniq.find(std::string_view(r->seq));
        if (it == uniq.end()) {
            int32_t id = static_cast<int32_t>(useqs.size());
            uniq.emplace(std::string_view(r->seq), id);
            useqs.push_back(std::string_view(r->seq));
            r->aln = id;
        } else {
            r->aln = it->second;
        }
    }
    std::vector<AlnOut> alns(useqs.size());
    {
        std::vector<std::unique_ptr<Aligner>> aligners;
        for (int i = 0; i < W; ++i) aligners.push_back(std::make_unique<Aligner>(opt.scoring));
        const std::string &ref = opt.ref_seq;
        parallel_dynamic(
            useqs.size(), W,
            [&](int tid, size_t i) {
                Aligner &al = *aligners[tid];
                std::string_view q = useqs[i];
                AlignResult best;
                bool rev = false;
                if (opt.strand != 1) best = opt.global ? al.global(ref, q) : al.local(ref, q);
                if (opt.strand != 0) {
                    std::string rc = reverse_complement(q);
                    AlignResult r2 = opt.global ? al.global(ref, rc) : al.local(ref, rc);
                    if (!best.ok || (r2.ok && r2.score > best.score)) {
                        best = std::move(r2);
                        rev = true;
                    }
                }
                AlnOut &o = alns[i];
                o.score = best.ok ? best.score : 0;
                o.reverse = rev;
                if (best.ok && best.score >= opt.min_score) {
                    int pos = 0;
                    o.cigar = to_bam_cigar(best, static_cast<int>(q.size()), pos);
                    o.pos = pos;
                    o.mapped = true;
                }
            },
            32);
    }

    {
        auto now = std::chrono::steady_clock::now();
        phase_ms.emplace_back("ms_align", std::chrono::duration_cast<std::chrono::milliseconds>(now - t_phase).count());
        t_phase = now;
    }
    // ------------------------------------------------------------ write
    std::sort(all.begin(), all.end(), [&](const OutRec *a, const OutRec *b) {
        const AlnOut &x = alns[a->aln], &y = alns[b->aln];
        if (x.mapped != y.mapped) return x.mapped;
        if (x.mapped && x.pos != y.pos) return x.pos < y.pos;
        if (a->cb != b->cb) return a->cb < b->cb;
        if (a->umi != b->umi) return a->umi < b->umi;
        return a->seq < b->seq;
    });
    int64_t n_written = 0, n_mapped = 0, n_score_fail = 0;
    std::vector<int64_t> score_hist;
    {
        std::ostringstream text;
        text << "@HD\tVN:1.6\tSO:coordinate\n";
        text << "@SQ\tSN:" << opt.ref_name << "\tLN:" << opt.ref_seq.size() << "\n";
        text << opt.header_extra;
        BamHeader hdr;
        hdr.text = text.str();
        hdr.names = {opt.ref_name};
        hdr.lengths = {static_cast<int32_t>(opt.ref_seq.size())};
        BamWriter writer(opt.output_bam, hdr, W);
        BamOutRecord rec;
        std::string cbs, umis;
        for (OutRec *r : all) {
            const AlnOut &a = alns[r->aln];
            if (a.score >= 0) {
                size_t s = static_cast<size_t>(a.score);
                if (score_hist.size() <= s) score_hist.resize(s + 1, 0);
                score_hist[s] += 1;
            }
            cbs = has_cb ? unpack2bit(r->cb, cb_len) : std::string();
            umis = has_umi ? unpack2bit(r->umi, umi_len) : std::string();
            rec.name = (has_cb ? cbs + "_" : std::string()) + (has_umi ? umis + "_" : std::string()) +
                       std::to_string(n_written);
            rec.seq = a.reverse ? reverse_complement(r->seq) : r->seq;
            rec.qual.clear();
            rec.aux.clear();
            if (has_cb) aux_append_string(rec.aux, "CB", cbs);
            if (has_umi) aux_append_string(rec.aux, "UB", umis);
            aux_append_int(rec.aux, "ZR", static_cast<int32_t>(r->reads));
            aux_append_int(rec.aux, "ZU", static_cast<int32_t>(r->n_umis));
            aux_append_float(rec.aux, "ZF", r->frac);
            aux_append_int(rec.aux, "AS", a.score);
            if (a.mapped) {
                rec.ref_id = 0;
                rec.pos = a.pos;
                rec.mapq = 60;
                rec.flag = a.reverse ? 16 : 0;
                rec.cigar = a.cigar;
                n_mapped++;
            } else {
                rec.ref_id = -1;
                rec.pos = -1;
                rec.mapq = 0;
                rec.flag = 4;
                rec.cigar.clear();
                n_score_fail++;
            }
            writer.write(rec);
            n_written++;
        }
        writer.close();
    }

    phase_ms.emplace_back("ms_write", std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::steady_clock::now() - t_phase).count());
    AlignSummary sum;
    int64_t n_molecules = 0, n_merged = 0, n_rescued = 0;
    for (size_t b = 0; b < B; ++b) {
        n_molecules += molecules[b];
        n_merged += merged_umis[b];
        n_rescued += rescued[b];
    }
    sum.reads_per_umi_hist.assign(1001, 0);
    for (auto &h : hists)
        for (size_t k = 0; k < h.size(); ++k) sum.reads_per_umi_hist[k] += h[k];
    sum.score_hist = std::move(score_hist);
    if (has_cb) {
        std::vector<std::pair<uint64_t, std::pair<int64_t, int64_t>>> cc;
        for (auto &v : cell_counts) cc.insert(cc.end(), v.begin(), v.end());
        std::sort(cc.begin(), cc.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (auto &c : cc) {
            sum.cells.push_back(unpack2bit(c.first, cb_len));
            sum.cell_reads.push_back(c.second.first);
            sum.cell_umis.push_back(c.second.second);
        }
    }
    sum.stats = {{"reads", n_reads + n_short + n_lowq},
                 {"reads_short", n_short},
                 {"reads_low_quality", n_lowq},
                 {"reads_valid", n_reads},
                 {"reads_cell_exact", has_cb ? reads_exact : n_reads},
                 {"reads_cell_corrected", reads_corrected},
                 {"reads_cell_invalid", reads_invalid},
                 {"raw_molecules", raw_molecules},
                 {"umis_merged", n_merged},
                 {"molecules", n_molecules},
                 {"reads_rescued_by_alignment", n_rescued},
                 {"unique_input_sequences", static_cast<int64_t>(n_unique_input)},
                 {"unique_aligned_sequences", static_cast<int64_t>(useqs.size())},
                 {"records_written", n_written},
                 {"records_aligned", n_mapped},
                 {"records_unaligned", n_score_fail}};
    for (auto &p : phase_ms) sum.stats.push_back(p);
    return sum;
}

}  // namespace cas
