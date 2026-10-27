#include "alleles.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "bam_io.hpp"
#include "common.hpp"
#include "parallel.hpp"
#include "sw.hpp"

namespace cas {

namespace {

enum Stat {
    kRecords = 0,
    kOnTarget,
    kFlagFiltered,
    kMapqFiltered,
    kMissingCell,
    kMissingUmi,
    kNotInCells,
    kNoIntbc,
    kUnaligned,
    kUsed,
    kUsedReads,
    kNStats
};
const char *kStatNames[] = {"records",      "records_on_target", "filtered_flag", "filtered_mapq",
                            "missing_cell", "missing_umi",       "not_in_cells",  "no_intbc",
                            "unaligned",    "records_used",      "reads_used"};

// Molecule-level key: everything is an interned id (UMIs of up to 29 ACGT
// bases are packed directly).
struct Key {
    uint32_t cell, intbc;
    uint64_t umi;
    uint32_t allele;
    bool operator==(const Key &o) const {
        return cell == o.cell && intbc == o.intbc && umi == o.umi && allele == o.allele;
    }
};
struct KeyHash {
    size_t operator()(const Key &k) const {
        uint64_t h = mix64((uint64_t(k.cell) << 32) ^ k.intbc);
        h = mix64(h ^ k.umi);
        return mix64(h ^ k.allele);
    }
};
struct KeyCount {
    Key key;
    uint32_t count;
};

constexpr uint64_t kPackedFlag = 1ULL << 63;
constexpr uint64_t kInternedFlag = 1ULL << 62;

uint64_t encode_umi(std::string_view u, ConcurrentInterner &interner) {
    uint64_t v;
    if (u.size() <= 29 && pack2bit(u, v)) return kPackedFlag | (uint64_t(u.size()) << 56) | v;
    return kInternedFlag | interner.intern(u);
}

std::string decode_umi(uint64_t k, const ConcurrentInterner &interner) {
    if (k & kPackedFlag) return unpack2bit(k & ((1ULL << 56) - 1), static_cast<int>((k >> 56) & 0x3f));
    if (k & kInternedFlag) return interner.get(static_cast<uint32_t>(k & 0xffffffffULL));
    return std::string();
}

struct ThreadState {
    ReadCall call;
    std::string seq;
    std::vector<uint32_t> cigar;
    std::string alleles;
    std::vector<std::vector<KeyCount>> buckets;  // per shard, current batch
    int64_t stats[kNStats] = {0};
    std::vector<int64_t> site_missing, site_clipped;
};

int hamming_str(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return 1 << 20;
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

void split_view(std::string_view s, char sep, std::vector<std::string_view> &out) {
    out.clear();
    size_t start = 0;
    while (true) {
        size_t p = s.find(sep, start);
        if (p == std::string_view::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

void finalize_categorical(const std::vector<std::string> &cats_in, std::vector<int32_t> &codes,
                          std::vector<std::string> &cats_out) {
    std::vector<int32_t> order(cats_in.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cats_in[a] < cats_in[b]; });
    std::vector<int32_t> remap(cats_in.size());
    cats_out.resize(cats_in.size());
    for (size_t i = 0; i < order.size(); ++i) {
        remap[order[i]] = static_cast<int32_t>(i);
        cats_out[i] = cats_in[order[i]];
    }
    for (auto &c : codes)
        if (c >= 0) c = remap[c];
}

}  // namespace

void bam_references(const std::string &path, std::vector<std::string> &names, std::vector<int32_t> &lengths) {
    BamReader reader(path, 1);
    names = reader.header().names;
    lengths = reader.header().lengths;
}

MoleculeTable call_alleles(const ConstructSpec &spec, const CallAllelesOptions &opt) {
    const int n_threads = std::max(1, opt.n_threads);
    BamReader reader(opt.path, n_threads);
    const BamHeader &hdr = reader.header();
    const size_t n_refs = hdr.names.size();
    const size_t n_sites = spec.sites.size();

    std::vector<char> selected(n_refs, opt.tids.empty() ? 1 : 0);
    for (int t : opt.tids)
        if (t >= 0 && static_cast<size_t>(t) < n_refs) selected[t] = 1;

    std::unordered_set<std::string, SvHash, SvEq> cells(opt.cells.begin(), opt.cells.end());
    const bool use_cb = !opt.cb_tag.empty();
    const bool use_umi = !opt.umi_tag.empty();
    const bool use_count = !opt.count_tag.empty();
    char cb_tag[2] = {0, 0}, umi_tag[2] = {0, 0}, count_tag[2] = {0, 0};
    if (use_cb) std::memcpy(cb_tag, opt.cb_tag.data(), 2);
    if (use_umi) std::memcpy(umi_tag, opt.umi_tag.data(), 2);
    if (use_count) std::memcpy(count_tag, opt.count_tag.data(), 2);

    ConcurrentInterner cell_ids, intbc_ids, umi_ids, allele_ids;
    const size_t n_shards = static_cast<size_t>(n_threads) * 8;
    std::vector<std::unordered_map<Key, uint64_t, KeyHash>> shards(n_shards);
    std::vector<std::unique_ptr<ThreadState>> states;
    for (int i = 0; i < n_threads; ++i) {
        auto st = std::make_unique<ThreadState>();
        st->buckets.resize(n_shards);
        st->site_missing.assign(n_sites, 0);
        st->site_clipped.assign(n_sites, 0);
        states.push_back(std::move(st));
    }
    // thread-local caches of recently seen ids (reads of a cell/intBC cluster)
    auto process = [&](const std::vector<BamRecord> &recs) {
        parallel_blocks(recs.size(), n_threads, [&](int tid, size_t b, size_t e) {
            ThreadState &st = *states[tid];
            std::string last_cb, last_intbc, last_alleles;
            uint32_t last_cb_id = 0, last_intbc_id = 0, last_allele_id = 0;
            bool have_cb = false, have_intbc = false, have_allele = false;
            for (size_t i = b; i < e; ++i) {
                const BamRecord &r = recs[i];
                st.stats[kRecords]++;
                int32_t ref_id = r.ref_id();
                if (ref_id < 0 || static_cast<size_t>(ref_id) >= n_refs || !selected[ref_id]) continue;
                st.stats[kOnTarget]++;
                uint16_t flag = r.flag();
                if (flag & (0x4 | 0x100 | 0x200 | 0x800)) {
                    st.stats[kFlagFiltered]++;
                    continue;
                }
                if (r.mapq() < opt.min_mapq) {
                    st.stats[kMapqFiltered]++;
                    continue;
                }
                std::string_view cb, umi;
                if (use_cb) {
                    cb = r.string_tag(cb_tag);
                    if (cb.empty() || cb == "-") {
                        st.stats[kMissingCell]++;
                        continue;
                    }
                    if (!cells.empty() && cells.find(cb) == cells.end()) {
                        st.stats[kNotInCells]++;
                        continue;
                    }
                }
                if (use_umi) {
                    umi = r.string_tag(umi_tag);
                    if (umi.empty() || umi == "-") {
                        st.stats[kMissingUmi]++;
                        continue;
                    }
                }
                int64_t count = use_count ? r.int_tag(count_tag, 1) : 1;
                if (count < 1) count = 1;
                r.sequence(st.seq);
                st.cigar.resize(r.n_cigar());
                for (int c = 0; c < r.n_cigar(); ++c) st.cigar[c] = r.cigar_op(c);
                const uint8_t *qual = r.l_seq() > 0 ? r.qual_ptr() : nullptr;
                if (!call_read(spec, r.pos(), st.cigar, st.seq, qual, st.call)) {
                    st.stats[kUnaligned]++;
                    continue;
                }
                std::string_view intbc;
                if (opt.intbc_from_contig)
                    intbc = hdr.names[ref_id];
                else if (spec.intbc_start >= 0)
                    intbc = st.call.intbc;
                else
                    intbc = opt.default_intbc;
                if (intbc.empty()) {
                    st.stats[kNoIntbc]++;
                    continue;
                }
                st.stats[kUsed]++;
                st.stats[kUsedReads] += count;
                for (size_t s = 0; s < n_sites; ++s) {
                    if (st.call.tokens[s].empty()) st.site_missing[s] += count;
                    if (st.call.clipped[s]) st.site_clipped[s] += count;
                }
                std::string &alleles = st.alleles;
                alleles.clear();
                for (size_t s = 0; s < n_sites; ++s) {
                    if (s) alleles.push_back('\x1f');
                    alleles.append(st.call.tokens[s]);
                }
                if (!have_cb || cb != last_cb) {
                    last_cb.assign(cb);
                    last_cb_id = cell_ids.intern(cb);
                    have_cb = true;
                }
                if (!have_intbc || intbc != last_intbc) {
                    last_intbc.assign(intbc);
                    last_intbc_id = intbc_ids.intern(intbc);
                    have_intbc = true;
                }
                if (!have_allele || alleles != last_alleles) {
                    last_alleles = alleles;
                    last_allele_id = allele_ids.intern(alleles);
                    have_allele = true;
                }
                Key k;
                k.cell = last_cb_id;
                k.intbc = last_intbc_id;
                // without UMIs (bulk data) every distinct allele is its own row
                k.umi = use_umi ? encode_umi(umi, umi_ids) : last_allele_id;
                k.allele = last_allele_id;
                size_t shard = mix64((uint64_t(k.cell) << 32) ^ k.intbc) % n_shards;
                st.buckets[shard].push_back(KeyCount{k, static_cast<uint32_t>(std::min<int64_t>(count, UINT32_MAX))});
            }
        });
        // each shard is updated by exactly one thread
        parallel_dynamic(n_shards, n_threads, [&](int, size_t sh) {
            auto &m = shards[sh];
            for (auto &st : states) {
                for (const KeyCount &kc : st->buckets[sh]) m[kc.key] += kc.count;
                st->buckets[sh].clear();
            }
        });
    };

    if (opt.use_index && !opt.tids.empty()) {
        std::vector<VirtualRange> ranges = read_index_ranges(opt.path, n_refs);
        if (ranges.empty()) {
            reader.for_each_batch(VirtualRange{hdr.first_record_voffset, kVoffsetEof}, process);
        } else {
            std::vector<int> tids = opt.tids;
            std::sort(tids.begin(), tids.end());
            std::vector<VirtualRange> todo;
            for (int t : tids) {
                if (t < 0 || static_cast<size_t>(t) >= ranges.size()) continue;
                VirtualRange vr = ranges[t];
                if (vr.end <= vr.beg) continue;
                if (!todo.empty() && vr.beg <= todo.back().end)
                    todo.back().end = std::max(todo.back().end, vr.end);
                else
                    todo.push_back(vr);
            }
            for (const auto &vr : todo) reader.for_each_batch(vr, process);
        }
    } else {
        reader.for_each_batch(VirtualRange{hdr.first_record_voffset, kVoffsetEof}, process);
    }
    for (auto &st : states) std::vector<std::vector<KeyCount>>().swap(st->buckets);

    // ---- molecules: group per shard by (cell, intBC, UMI), vote per site ----
    std::vector<ConcurrentInterner> site_ids(n_sites);
    struct Row {
        uint32_t cell, intbc;
        uint64_t umi;
        int32_t n_reads;
        float frac;
    };
    std::vector<std::vector<Row>> rows(n_shards);
    std::vector<std::vector<int32_t>> row_sites(n_shards);  // n_sites per row, -1 missing (raw site ids)
    parallel_dynamic(n_shards, n_threads, [&](int, size_t sh) {
        std::vector<KeyCount> entries;
        entries.reserve(shards[sh].size());
        for (auto &kv : shards[sh]) entries.push_back(KeyCount{kv.first, static_cast<uint32_t>(kv.second)});
        std::unordered_map<Key, uint64_t, KeyHash>().swap(shards[sh]);
        std::sort(entries.begin(), entries.end(), [](const KeyCount &a, const KeyCount &b) {
            if (a.key.cell != b.key.cell) return a.key.cell < b.key.cell;
            if (a.key.intbc != b.key.intbc) return a.key.intbc < b.key.intbc;
            return a.key.umi < b.key.umi;
        });
        std::unordered_map<uint32_t, std::vector<std::string_view>> tokens_cache;
        auto tokens_of = [&](uint32_t allele) -> const std::vector<std::string_view> & {
            auto it = tokens_cache.find(allele);
            if (it != tokens_cache.end()) return it->second;
            std::vector<std::string_view> t;
            split_view(allele_ids.get(allele), '\x1f', t);
            return tokens_cache.emplace(allele, std::move(t)).first->second;
        };
        struct Umi {
            uint64_t umi;
            uint64_t reads = 0;
            std::vector<std::pair<uint32_t, uint64_t>> alleles;
            bool merged = false;
        };
        std::vector<Umi> umis;
        std::vector<std::unordered_map<std::string_view, uint64_t, SvHash, SvEq>> votes(n_sites);
        size_t i = 0;
        auto &out = rows[sh];
        auto &out_sites = row_sites[sh];
        while (i < entries.size()) {
            size_t j = i;
            while (j < entries.size() && entries[j].key.cell == entries[i].key.cell &&
                   entries[j].key.intbc == entries[i].key.intbc)
                ++j;
            umis.clear();
            for (size_t k = i; k < j; ++k) {
                if (umis.empty() || umis.back().umi != entries[k].key.umi) umis.push_back(Umi{entries[k].key.umi});
                umis.back().reads += entries[k].count;
                umis.back().alleles.emplace_back(entries[k].key.allele, entries[k].count);
            }
            if (use_umi && opt.umi_max_dist > 0 && umis.size() > 1) {
                std::vector<std::string> strs(umis.size());
                for (size_t u = 0; u < umis.size(); ++u) strs[u] = decode_umi(umis[u].umi, umi_ids);
                std::vector<size_t> order(umis.size());
                std::iota(order.begin(), order.end(), size_t(0));
                std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    if (umis[a].reads != umis[b].reads) return umis[a].reads > umis[b].reads;
                    return strs[a] > strs[b];
                });
                std::vector<size_t> roots;
                for (size_t idx : order) {
                    size_t target = SIZE_MAX;
                    for (size_t r : roots)
                        if (hamming_str(strs[idx], strs[r]) <= opt.umi_max_dist) {
                            target = r;
                            break;
                        }
                    if (target == SIZE_MAX) {
                        roots.push_back(idx);
                    } else {
                        umis[target].reads += umis[idx].reads;
                        for (auto &a : umis[idx].alleles) umis[target].alleles.push_back(a);
                        umis[idx].alleles.clear();
                        umis[idx].reads = 0;
                        umis[idx].merged = true;
                    }
                }
            }
            for (auto &u : umis) {
                if (u.merged || u.reads == 0) continue;
                for (auto &v : votes) v.clear();
                for (auto &a : u.alleles) {
                    const auto &toks = tokens_of(a.first);
                    for (size_t s = 0; s < n_sites && s < toks.size(); ++s)
                        if (!toks[s].empty()) votes[s][toks[s]] += a.second;
                }
                Row row{entries[i].key.cell, entries[i].key.intbc, u.umi,
                        static_cast<int32_t>(std::min<uint64_t>(u.reads, INT32_MAX)), 1.0f};
                for (size_t s = 0; s < n_sites; ++s) {
                    if (votes[s].empty()) {
                        out_sites.push_back(-1);
                        continue;
                    }
                    uint64_t tot = 0, best = 0;
                    std::string_view best_tok;
                    for (auto &kv : votes[s]) {
                        tot += kv.second;
                        if (kv.second > best || (kv.second == best && kv.first < best_tok)) {
                            best = kv.second;
                            best_tok = kv.first;
                        }
                    }
                    out_sites.push_back(static_cast<int32_t>(site_ids[s].intern(best_tok)));
                    row.frac = std::min(row.frac, static_cast<float>(best) / static_cast<float>(tot));
                }
                out.push_back(row);
            }
            i = j;
        }
    });

    // ---- categoricals ----
    MoleculeTable res;
    size_t n_rows = 0;
    for (auto &r : rows) n_rows += r.size();
    std::unordered_map<uint32_t, int32_t> cell_dense, intbc_dense;
    std::vector<std::string> cell_cats = cell_ids.strings(cell_dense);
    std::vector<std::string> intbc_cats = intbc_ids.strings(intbc_dense);
    std::vector<std::unordered_map<uint32_t, int32_t>> site_dense(n_sites);
    std::vector<std::vector<std::string>> site_cats(n_sites);
    for (size_t s = 0; s < n_sites; ++s) site_cats[s] = site_ids[s].strings(site_dense[s]);
    // UMIs: dense ids over the distinct keys in the output
    std::unordered_map<uint64_t, int32_t> umi_dense;
    std::vector<std::string> umi_cats;
    std::vector<size_t> offsets(n_shards + 1, 0);
    for (size_t sh = 0; sh < n_shards; ++sh) offsets[sh + 1] = offsets[sh] + rows[sh].size();
    res.cell.codes.resize(n_rows);
    res.umi.codes.resize(n_rows);
    res.intbc.codes.resize(n_rows);
    res.sites.resize(n_sites);
    for (auto &s : res.sites) s.codes.resize(n_rows);
    res.n_reads.resize(n_rows);
    res.allele_frac.resize(n_rows);
    for (size_t sh = 0; sh < n_shards; ++sh)
        for (const Row &r : rows[sh])
            if (umi_dense.emplace(r.umi, static_cast<int32_t>(umi_dense.size())).second)
                umi_cats.push_back(use_umi ? decode_umi(r.umi, umi_ids) : std::string());
    parallel_dynamic(n_shards, n_threads, [&](int, size_t sh) {
        size_t o = offsets[sh];
        for (size_t i = 0; i < rows[sh].size(); ++i) {
            const Row &r = rows[sh][i];
            res.cell.codes[o + i] = cell_dense.at(r.cell);
            res.intbc.codes[o + i] = intbc_dense.at(r.intbc);
            res.umi.codes[o + i] = umi_dense.at(r.umi);
            for (size_t s = 0; s < n_sites; ++s) {
                int32_t v = row_sites[sh][i * n_sites + s];
                res.sites[s].codes[o + i] = v < 0 ? -1 : site_dense[s].at(static_cast<uint32_t>(v));
            }
            res.n_reads[o + i] = r.n_reads;
            res.allele_frac[o + i] = r.frac;
        }
    });
    rows.clear();
    row_sites.clear();
    finalize_categorical(cell_cats, res.cell.codes, res.cell.categories);
    finalize_categorical(umi_cats, res.umi.codes, res.umi.categories);
    finalize_categorical(intbc_cats, res.intbc.codes, res.intbc.categories);
    for (size_t s = 0; s < n_sites; ++s)
        finalize_categorical(site_cats[s], res.sites[s].codes, res.sites[s].categories);

    // Deterministic row order: cell, intBC, UMI.
    std::vector<uint32_t> order(n_rows);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (res.cell.codes[a] != res.cell.codes[b]) return res.cell.codes[a] < res.cell.codes[b];
        if (res.intbc.codes[a] != res.intbc.codes[b]) return res.intbc.codes[a] < res.intbc.codes[b];
        return res.umi.codes[a] < res.umi.codes[b];
    });
    auto permute = [&](auto &v) {
        auto copy = v;
        for (size_t i = 0; i < n_rows; ++i) v[i] = copy[order[i]];
    };
    permute(res.cell.codes);
    permute(res.umi.codes);
    permute(res.intbc.codes);
    for (auto &s : res.sites) permute(s.codes);
    permute(res.n_reads);
    permute(res.allele_frac);

    int64_t totals[kNStats] = {0};
    res.site_missing.assign(n_sites, 0);
    res.site_clipped.assign(n_sites, 0);
    for (auto &st : states) {
        for (int k = 0; k < kNStats; ++k) totals[k] += st->stats[k];
        for (size_t s = 0; s < n_sites; ++s) {
            res.site_missing[s] += st->site_missing[s];
            res.site_clipped[s] += st->site_clipped[s];
        }
    }
    for (int k = 0; k < kNStats; ++k) res.stats.emplace_back(kStatNames[k], totals[k]);
    res.stats.emplace_back("molecules", static_cast<int64_t>(n_rows));
    res.stats.emplace_back("bytes_inflated", static_cast<int64_t>(reader.bytes_inflated()));
    return res;
}

}  // namespace cas
