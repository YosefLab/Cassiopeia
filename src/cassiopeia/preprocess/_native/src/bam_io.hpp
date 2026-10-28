// Self-contained BGZF/BAM reading and writing (zlib only).
//
// Reading decompresses BGZF blocks in parallel and hands out batches of raw
// BAM records. Contig-restricted reads use the BAI/CSI pseudo-bin so only the
// blocks holding that contig are touched, which keeps 100+ GB Cell Ranger
// BAMs cheap when only the target-site contigs are needed.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace cas {

constexpr uint64_t kVoffsetEof = ~uint64_t(0);

struct BamHeader {
    std::string text;
    std::vector<std::string> names;
    std::vector<int32_t> lengths;
    uint64_t first_record_voffset = 0;
};

struct VirtualRange {
    uint64_t beg = 0;
    uint64_t end = kVoffsetEof;  // exclusive
};

template <typename T>
inline T load_le(const uint8_t *p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// Lightweight view of one BAM record. `p` points at the block_size field.
struct BamRecord {
    const uint8_t *p = nullptr;

    uint32_t block_size() const { return load_le<uint32_t>(p); }
    const uint8_t *data() const { return p + 4; }
    int32_t ref_id() const { return load_le<int32_t>(p + 4); }
    int32_t pos() const { return load_le<int32_t>(p + 8); }
    uint8_t l_read_name() const { return p[12]; }
    uint8_t mapq() const { return p[13]; }
    uint16_t n_cigar() const { return load_le<uint16_t>(p + 16); }
    uint16_t flag() const { return load_le<uint16_t>(p + 18); }
    int32_t l_seq() const { return load_le<int32_t>(p + 20); }
    const char *read_name() const { return reinterpret_cast<const char *>(p + 36); }
    const uint8_t *cigar_ptr() const { return p + 36 + l_read_name(); }
    uint32_t cigar_op(int i) const { return load_le<uint32_t>(cigar_ptr() + 4 * i); }
    const uint8_t *seq_ptr() const { return cigar_ptr() + 4 * n_cigar(); }
    const uint8_t *qual_ptr() const { return seq_ptr() + (l_seq() + 1) / 2; }
    const uint8_t *aux_ptr() const { return qual_ptr() + l_seq(); }
    const uint8_t *end_ptr() const { return p + 4 + block_size(); }

    void sequence(std::string &out) const;
    // Finds an aux tag; returns pointer to its type byte or nullptr.
    const uint8_t *find_tag(const char tag[2]) const;
    // Returns a Z/H string tag value (empty view when missing).
    std::string_view string_tag(const char tag[2]) const;
    // Returns an integer tag, or `fallback` when missing / not integral.
    int64_t int_tag(const char tag[2], int64_t fallback) const;
    // Returns a float tag, or `fallback`.
    double float_tag(const char tag[2], double fallback) const;
};

class BamReader {
   public:
    BamReader(const std::string &path, int n_threads);
    ~BamReader();
    BamReader(const BamReader &) = delete;
    BamReader &operator=(const BamReader &) = delete;

    const BamHeader &header() const { return header_; }

    // Calls fn(const std::vector<BamRecord>&) for successive batches of
    // records in the virtual-offset range. Records point into an internal
    // buffer that is only valid during the callback.
    template <typename Fn>
    void for_each_batch(const VirtualRange &range, Fn &&fn, size_t chunk_bytes = 16 << 20);

    uint64_t bytes_inflated() const { return bytes_inflated_; }

   private:
    struct Block {
        uint64_t coffset;
        uint32_t bsize;  // total compressed block size
        uint32_t isize;  // uncompressed size
        size_t buf_offset;  // offset of compressed block in raw buffer
    };
    void read_header();
    size_t read_raw(uint64_t offset, size_t n, std::vector<uint8_t> &buf);
    // Reads blocks starting at coffset into `blocks`/`raw` (at most
    // chunk_bytes of compressed data, whole blocks only). Returns the
    // compressed offset of the next unread block (or file size at EOF).
    uint64_t load_blocks(uint64_t coffset, size_t chunk_bytes, std::vector<Block> &blocks,
                         std::vector<uint8_t> &raw);
    void inflate_blocks(const std::vector<Block> &blocks, const std::vector<uint8_t> &raw,
                        std::vector<uint8_t> &out, std::vector<size_t> &out_offsets);

    std::string path_;
    FILE *fp_ = nullptr;
    uint64_t file_size_ = 0;
    int n_threads_ = 1;
    BamHeader header_;
    uint64_t bytes_inflated_ = 0;
};

// Per-reference virtual offset ranges from a .bai or .csi index. Returns an
// empty vector if no index is found. Each entry is {beg, end}; beg == end
// means the reference has no reads.
std::vector<VirtualRange> read_index_ranges(const std::string &bam_path, size_t n_refs,
                                            std::string *index_path_out = nullptr);

// Inflate one BGZF block (raw points at block start). Returns false on error.
bool inflate_bgzf_block(const uint8_t *raw, uint32_t bsize, uint8_t *out, uint32_t isize);

// ------------------------------------------------------------- writing ---

struct BamOutRecord {
    int32_t ref_id = -1;
    int32_t pos = -1;
    uint8_t mapq = 255;
    uint16_t flag = 4;
    std::string name;
    std::vector<uint32_t> cigar;  // (len << 4) | op
    std::string seq;
    std::string qual;  // phred values (not +33); empty = missing
    std::string aux;   // pre-encoded aux bytes
};

void aux_append_string(std::string &aux, const char tag[2], std::string_view value);
void aux_append_int(std::string &aux, const char tag[2], int32_t value);
void aux_append_float(std::string &aux, const char tag[2], float value);

class BamWriter {
   public:
    BamWriter(const std::string &path, const BamHeader &header, int n_threads, int level = 6);
    ~BamWriter();
    void write(const BamOutRecord &rec);
    void close();

   private:
    void flush_blocks(bool final);
    void encode(const BamOutRecord &rec);
    FILE *fp_ = nullptr;
    int n_threads_;
    int level_;
    std::string pending_;  // uncompressed bytes not yet compressed
    bool closed_ = false;
};

int32_t reg2bin(int32_t beg, int32_t end);

}  // namespace cas

#include "bam_io_impl.hpp"
