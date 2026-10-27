// Shared helpers for the Cassiopeia preprocessing kernels.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cas {

// 2-bit nucleotide codes; 4 marks anything else (N).
inline uint8_t base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 4;
    }
}

inline char code_base(uint8_t c) { return "ACGTN"[c < 4 ? c : 4]; }

inline char complement(char c) {
    switch (c) {
        case 'A': return 'T';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'T': return 'A';
        case 'a': return 't';
        case 'c': return 'g';
        case 'g': return 'c';
        case 't': return 'a';
        default: return 'N';
    }
}

inline std::string reverse_complement(std::string_view s) {
    std::string out(s.size(), 'N');
    for (size_t i = 0; i < s.size(); ++i) out[s.size() - 1 - i] = complement(s[i]);
    return out;
}

// Pack an ACGT string of length <= 32 into a uint64. Returns false if it
// contains a non-ACGT character.
inline bool pack2bit(std::string_view s, uint64_t &out) {
    uint64_t v = 0;
    for (char c : s) {
        uint8_t b = base_code(c);
        if (b > 3) return false;
        v = (v << 2) | b;
    }
    out = v;
    return true;
}

inline std::string unpack2bit(uint64_t v, int len) {
    std::string s(len, 'A');
    for (int i = len - 1; i >= 0; --i) {
        s[i] = code_base(v & 3);
        v >>= 2;
    }
    return s;
}

// Number of differing bases between two packed sequences of equal length.
inline int packed_hamming(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
    x = (x | (x >> 1)) & 0x5555555555555555ULL;
    return __builtin_popcountll(x);
}

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline uint64_t hash_bytes(const char *p, size_t n, uint64_t seed = 0x9e3779b97f4a7c15ULL) {
    uint64_t h = seed ^ (n * 0x9e3779b97f4a7c15ULL);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = mix64(h ^ w) + 0x9e3779b97f4a7c15ULL;
    }
    uint64_t tail = 0;
    if (i < n) std::memcpy(&tail, p + i, n - i);
    return mix64(h ^ tail ^ (n << 56));
}

inline uint64_t hash_sv(std::string_view s) { return hash_bytes(s.data(), s.size()); }

struct SvHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return hash_sv(s); }
    size_t operator()(const std::string &s) const { return hash_sv(s); }
};

struct SvEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const { return a == b; }
};

// Python-style slice s[a:b] with negative-index semantics (used to mirror
// the legacy allele strings exactly).
inline std::string py_slice(const std::string &s, long a, long b) {
    long n = static_cast<long>(s.size());
    if (a < 0) a += n;
    if (b < 0) b += n;
    a = std::clamp(a, 0L, n);
    b = std::clamp(b, 0L, n);
    if (b <= a) return std::string();
    return s.substr(a, b - a);
}

// Thread-safe string interner with striped locks. Ids are dense per stripe
// ((local << 8) | stripe); `get` is only valid once all inserts are done.
class ConcurrentInterner {
   public:
    static constexpr uint32_t kStripes = 256;
    ConcurrentInterner() {
        for (auto &s : stripes_) s = std::make_unique<Stripe>();
    }
    uint32_t intern(std::string_view s) {
        uint32_t st = static_cast<uint32_t>(hash_sv(s) & (kStripes - 1));
        Stripe &stripe = *stripes_[st];
        std::lock_guard<std::mutex> lk(stripe.m);
        auto it = stripe.index.find(s);
        if (it != stripe.index.end()) return it->second;
        uint32_t local = static_cast<uint32_t>(stripe.by_id.size());
        if (local >= (1u << 24)) throw std::runtime_error("too many distinct strings");
        uint32_t id = (local << 8) | st;
        auto res = stripe.index.emplace(std::string(s), id);
        stripe.by_id.push_back(&res.first->first);
        return id;
    }
    const std::string &get(uint32_t id) const { return *stripes_[id & (kStripes - 1)]->by_id[id >> 8]; }
    // Dense renumbering: returns all strings and fills `dense` (id -> index).
    std::vector<std::string> strings(std::unordered_map<uint32_t, int32_t> &dense) const {
        std::vector<std::string> out;
        for (uint32_t st = 0; st < kStripes; ++st)
            for (uint32_t l = 0; l < stripes_[st]->by_id.size(); ++l) {
                dense[(l << 8) | st] = static_cast<int32_t>(out.size());
                out.push_back(*stripes_[st]->by_id[l]);
            }
        return out;
    }
    size_t size() const {
        size_t n = 0;
        for (auto &s : stripes_) n += s->by_id.size();
        return n;
    }

   private:
    struct Stripe {
        std::mutex m;
        std::unordered_map<std::string, uint32_t, SvHash, SvEq> index;
        std::vector<const std::string *> by_id;
    };
    std::array<std::unique_ptr<Stripe>, kStripes> stripes_;
};

// Interns strings into dense uint32 ids. Not thread safe.
class Interner {
   public:
    uint32_t intern(std::string_view s) {
        auto it = index_.find(s);
        if (it != index_.end()) return it->second;
        uint32_t id = static_cast<uint32_t>(strings_.size());
        strings_.emplace_back(s);
        index_.emplace(strings_.back(), id);
        return id;
    }
    int64_t find(std::string_view s) const {
        auto it = index_.find(s);
        return it == index_.end() ? -1 : static_cast<int64_t>(it->second);
    }
    const std::string &get(uint32_t id) const { return strings_[id]; }
    size_t size() const { return strings_.size(); }
    const std::vector<std::string> &strings() const { return strings_; }
    void reserve(size_t n) { index_.reserve(n); }

   private:
    std::vector<std::string> strings_;
    std::unordered_map<std::string, uint32_t, SvHash, SvEq> index_;
};

}  // namespace cas
