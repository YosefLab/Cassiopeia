#include "bam_io.hpp"

#include <zlib.h>

#include <algorithm>
#include <stdexcept>

#include "parallel.hpp"

namespace cas {

namespace {

const char kSeqNt16[] = "=ACMGRSVTWYHKDBN";

uint8_t nt16_code(char c) {
    switch (c) {
        case '=': return 0;
        case 'A': case 'a': return 1;
        case 'C': case 'c': return 2;
        case 'G': case 'g': return 4;
        case 'T': case 't': return 8;
        default: return 15;
    }
}

size_t aux_value_size(const uint8_t *t, const uint8_t *end) {
    switch (*t) {
        case 'A': case 'c': case 'C': return 1;
        case 's': case 'S': return 2;
        case 'i': case 'I': case 'f': return 4;
        case 'Z': case 'H': {
            const uint8_t *q = t + 1;
            while (q < end && *q) ++q;
            return static_cast<size_t>(q - (t + 1)) + 1;
        }
        case 'B': {
            if (t + 6 > end) return static_cast<size_t>(end - t);
            char sub = static_cast<char>(t[1]);
            uint32_t n = load_le<uint32_t>(t + 2);
            size_t es = (sub == 'c' || sub == 'C') ? 1 : (sub == 's' || sub == 'S') ? 2 : 4;
            return 5 + es * n;
        }
        default: return static_cast<size_t>(end - t);  // unknown: stop parsing
    }
}

std::vector<uint8_t> read_file(const std::string &path) {
    std::vector<uint8_t> data;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return data;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) {
        data.resize(static_cast<size_t>(n));
        size_t got = std::fread(data.data(), 1, data.size(), f);
        data.resize(got);
    }
    std::fclose(f);
    return data;
}

bool file_exists(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Decompress a (possibly multi-member) gzip buffer.
std::vector<uint8_t> gunzip_all(const std::vector<uint8_t> &in) {
    std::vector<uint8_t> out;
    z_stream zs{};
    if (inflateInit2(&zs, 15 + 32) != Z_OK) throw std::runtime_error("inflateInit2 failed");
    zs.next_in = const_cast<Bytef *>(in.data());
    zs.avail_in = static_cast<uInt>(in.size());
    std::vector<uint8_t> chunk(1 << 16);
    while (true) {
        zs.next_out = chunk.data();
        zs.avail_out = static_cast<uInt>(chunk.size());
        int ret = inflate(&zs, Z_NO_FLUSH);
        out.insert(out.end(), chunk.data(), chunk.data() + (chunk.size() - zs.avail_out));
        if (ret == Z_STREAM_END) {
            if (zs.avail_in == 0) break;
            inflateReset(&zs);  // next gzip member
            continue;
        }
        if (ret != Z_OK) {
            inflateEnd(&zs);
            throw std::runtime_error("failed to decompress index");
        }
        if (zs.avail_in == 0 && zs.avail_out != 0) break;
    }
    inflateEnd(&zs);
    return out;
}

struct ThreadInflater {
    z_stream zs{};
    bool ok = false;
    ThreadInflater() { ok = inflateInit2(&zs, -15) == Z_OK; }
    ~ThreadInflater() {
        if (ok) inflateEnd(&zs);
    }
};

struct ThreadDeflater {
    z_stream zs{};
    int level = -1;
    ~ThreadDeflater() {
        if (level >= 0) deflateEnd(&zs);
    }
    void init(int lvl) {
        if (level == lvl) {
            deflateReset(&zs);
            return;
        }
        if (level >= 0) deflateEnd(&zs);
        zs = z_stream{};
        if (deflateInit2(&zs, lvl, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("deflateInit2 failed");
        level = lvl;
    }
};

}  // namespace

// ----------------------------------------------------------- BamRecord ---

void BamRecord::sequence(std::string &out) const {
    int32_t n = l_seq();
    out.resize(n);
    const uint8_t *s = seq_ptr();
    for (int32_t i = 0; i < n; ++i) {
        uint8_t code = (i & 1) ? (s[i >> 1] & 0xf) : (s[i >> 1] >> 4);
        out[i] = kSeqNt16[code];
    }
}

const uint8_t *BamRecord::find_tag(const char tag[2]) const {
    const uint8_t *q = aux_ptr();
    const uint8_t *end = end_ptr();
    while (q + 3 <= end) {
        const uint8_t *t = q + 2;
        size_t sz = aux_value_size(t, end);
        if (q[0] == static_cast<uint8_t>(tag[0]) && q[1] == static_cast<uint8_t>(tag[1])) return t;
        q = t + 1 + sz;
    }
    return nullptr;
}

std::string_view BamRecord::string_tag(const char tag[2]) const {
    const uint8_t *t = find_tag(tag);
    if (!t || (*t != 'Z' && *t != 'H')) return {};
    const char *s = reinterpret_cast<const char *>(t + 1);
    const char *end = reinterpret_cast<const char *>(end_ptr());
    const char *e = s;
    while (e < end && *e) ++e;
    return std::string_view(s, static_cast<size_t>(e - s));
}

int64_t BamRecord::int_tag(const char tag[2], int64_t fallback) const {
    const uint8_t *t = find_tag(tag);
    if (!t) return fallback;
    switch (*t) {
        case 'c': return static_cast<int8_t>(t[1]);
        case 'C': return t[1];
        case 's': return load_le<int16_t>(t + 1);
        case 'S': return load_le<uint16_t>(t + 1);
        case 'i': return load_le<int32_t>(t + 1);
        case 'I': return load_le<uint32_t>(t + 1);
        default: return fallback;
    }
}

double BamRecord::float_tag(const char tag[2], double fallback) const {
    const uint8_t *t = find_tag(tag);
    if (!t) return fallback;
    if (*t == 'f') return load_le<float>(t + 1);
    int64_t v = int_tag(tag, INT64_MIN);
    return v == INT64_MIN ? fallback : static_cast<double>(v);
}

// ----------------------------------------------------------- BamReader ---

bool inflate_bgzf_block(const uint8_t *raw, uint32_t bsize, uint8_t *out, uint32_t isize) {
    thread_local ThreadInflater inf;
    if (!inf.ok) return false;
    uint16_t xlen = load_le<uint16_t>(raw + 10);
    if (bsize < static_cast<uint32_t>(xlen) + 20) return false;
    uint32_t clen = bsize - xlen - 20;
    if (isize == 0) return true;
    inflateReset(&inf.zs);
    inf.zs.next_in = const_cast<Bytef *>(raw + 12 + xlen);
    inf.zs.avail_in = clen;
    inf.zs.next_out = out;
    inf.zs.avail_out = isize;
    int ret = inflate(&inf.zs, Z_FINISH);
    return ret == Z_STREAM_END && inf.zs.avail_out == 0;
}

BamReader::BamReader(const std::string &path, int n_threads) : path_(path), n_threads_(n_threads) {
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) throw std::runtime_error("cannot open BAM: " + path);
    fseeko(fp_, 0, SEEK_END);
    file_size_ = static_cast<uint64_t>(ftello(fp_));
    fseeko(fp_, 0, SEEK_SET);
    read_header();
}

BamReader::~BamReader() {
    if (fp_) std::fclose(fp_);
}

size_t BamReader::read_raw(uint64_t offset, size_t n, std::vector<uint8_t> &buf) {
    buf.resize(n);
    if (fseeko(fp_, static_cast<off_t>(offset), SEEK_SET) != 0) return 0;
    size_t got = std::fread(buf.data(), 1, n, fp_);
    buf.resize(got);
    return got;
}

uint64_t BamReader::load_blocks(uint64_t coffset, size_t chunk_bytes, std::vector<Block> &blocks,
                                std::vector<uint8_t> &raw) {
    blocks.clear();
    size_t want = std::max<size_t>(chunk_bytes, 1 << 17);
    if (coffset + want > file_size_) want = static_cast<size_t>(file_size_ - coffset);
    read_raw(coffset, want, raw);
    size_t i = 0;
    while (i + 18 <= raw.size()) {
        const uint8_t *b = raw.data() + i;
        if (b[0] != 31 || b[1] != 139 || b[2] != 8 || !(b[3] & 4))
            throw std::runtime_error("not a BGZF file (bad block header) in " + path_);
        uint16_t xlen = load_le<uint16_t>(b + 10);
        if (i + 12 + xlen > raw.size()) break;
        uint32_t bsize = 0;
        for (size_t x = 0; x + 4 <= xlen;) {
            const uint8_t *sf = b + 12 + x;
            uint16_t slen = load_le<uint16_t>(sf + 2);
            if (sf[0] == 'B' && sf[1] == 'C' && slen == 2) bsize = load_le<uint16_t>(sf + 4) + 1u;
            x += 4 + slen;
        }
        if (bsize == 0) throw std::runtime_error("BGZF block without BC subfield in " + path_);
        if (i + bsize > raw.size()) break;
        uint32_t isize = load_le<uint32_t>(b + bsize - 4);
        blocks.push_back(Block{coffset + i, bsize, isize, i});
        i += bsize;
    }
    return coffset + i;
}

void BamReader::inflate_blocks(const std::vector<Block> &blocks, const std::vector<uint8_t> &raw,
                               std::vector<uint8_t> &out, std::vector<size_t> &out_offsets) {
    out_offsets.assign(blocks.size() + 1, 0);
    for (size_t i = 0; i < blocks.size(); ++i) out_offsets[i + 1] = out_offsets[i] + blocks[i].isize;
    out.resize(out_offsets.back());
    bool failed = false;
    parallel_dynamic(
        blocks.size(), n_threads_,
        [&](int, size_t i) {
            const Block &bl = blocks[i];
            if (!inflate_bgzf_block(raw.data() + bl.buf_offset, bl.bsize, out.data() + out_offsets[i],
                                    bl.isize))
                failed = true;
        },
        8);
    if (failed) throw std::runtime_error("BGZF decompression failed in " + path_);
    bytes_inflated_ += out.size();
}

void BamReader::read_header() {
    // Inflate blocks from the start until the full header is available.
    std::vector<Block> blocks;
    std::vector<uint8_t> raw, out, data;
    std::vector<size_t> offs;
    std::vector<std::pair<uint64_t, size_t>> block_starts;  // (coffset, uncompressed start)
    uint64_t coff = 0;
    size_t need = 12;
    auto have_header = [&](size_t &end_pos) -> bool {
        if (data.size() < 12) return false;
        if (std::memcmp(data.data(), "BAM\1", 4) != 0) throw std::runtime_error("not a BAM file: " + path_);
        uint32_t l_text = load_le<uint32_t>(data.data() + 4);
        size_t pos = 8 + l_text;
        if (data.size() < pos + 4) return false;
        uint32_t n_ref = load_le<uint32_t>(data.data() + pos);
        pos += 4;
        for (uint32_t r = 0; r < n_ref; ++r) {
            if (data.size() < pos + 4) return false;
            uint32_t l_name = load_le<uint32_t>(data.data() + pos);
            pos += 4 + l_name + 4;
            if (data.size() < pos) return false;
        }
        end_pos = pos;
        return true;
    };
    size_t end_pos = 0;
    while (coff < file_size_) {
        uint64_t next = load_blocks(coff, 1 << 20, blocks, raw);
        if (blocks.empty()) break;
        inflate_blocks(blocks, raw, out, offs);
        for (size_t i = 0; i < blocks.size(); ++i) block_starts.emplace_back(blocks[i].coffset, data.size() + offs[i]);
        data.insert(data.end(), out.begin(), out.end());
        coff = next;
        if (data.size() >= need && have_header(end_pos)) break;
    }
    if (!have_header(end_pos)) throw std::runtime_error("truncated BAM header: " + path_);
    uint32_t l_text = load_le<uint32_t>(data.data() + 4);
    header_.text.assign(reinterpret_cast<const char *>(data.data() + 8), l_text);
    size_t text_nul = header_.text.find('\0');
    if (text_nul != std::string::npos) header_.text.resize(text_nul);
    size_t pos = 8 + l_text;
    uint32_t n_ref = load_le<uint32_t>(data.data() + pos);
    pos += 4;
    for (uint32_t r = 0; r < n_ref; ++r) {
        uint32_t l_name = load_le<uint32_t>(data.data() + pos);
        pos += 4;
        std::string name(reinterpret_cast<const char *>(data.data() + pos), l_name ? l_name - 1 : 0);
        pos += l_name;
        int32_t l_ref = load_le<int32_t>(data.data() + pos);
        pos += 4;
        header_.names.push_back(std::move(name));
        header_.lengths.push_back(l_ref);
    }
    // Virtual offset of the first record.
    uint64_t voff = 0;
    bool found = false;
    for (size_t i = 0; i < block_starts.size(); ++i) {
        size_t bstart = block_starts[i].second;
        size_t bend = (i + 1 < block_starts.size()) ? block_starts[i + 1].second : data.size();
        if (end_pos >= bstart && end_pos < bend) {
            voff = (block_starts[i].first << 16) | (end_pos - bstart);
            found = true;
            break;
        }
    }
    if (!found) voff = coff << 16;  // header ends exactly at a block boundary
    header_.first_record_voffset = voff;
}

// --------------------------------------------------------------- index ---

std::vector<VirtualRange> read_index_ranges(const std::string &bam_path, size_t n_refs,
                                            std::string *index_path_out) {
    std::vector<std::string> candidates = {bam_path + ".bai", bam_path + ".csi"};
    if (bam_path.size() > 4 && bam_path.substr(bam_path.size() - 4) == ".bam")
        candidates.push_back(bam_path.substr(0, bam_path.size() - 4) + ".bai");
    std::string idx;
    for (const auto &c : candidates)
        if (file_exists(c)) {
            idx = c;
            break;
        }
    std::vector<VirtualRange> ranges;
    if (idx.empty()) return ranges;
    if (index_path_out) *index_path_out = idx;
    std::vector<uint8_t> data = read_file(idx);
    bool csi = false;
    if (data.size() >= 2 && data[0] == 31 && data[1] == 139) data = gunzip_all(data);
    if (data.size() < 8) throw std::runtime_error("index too short: " + idx);
    if (std::memcmp(data.data(), "BAI\1", 4) == 0)
        csi = false;
    else if (std::memcmp(data.data(), "CSI\1", 4) == 0)
        csi = true;
    else
        throw std::runtime_error("unrecognized index format: " + idx);

    size_t p = 4;
    uint32_t pseudo_bin = 37450;
    if (csi) {
        int32_t min_shift = load_le<int32_t>(data.data() + p);
        int32_t depth = load_le<int32_t>(data.data() + p + 4);
        int32_t l_aux = load_le<int32_t>(data.data() + p + 8);
        (void)min_shift;
        p += 12 + l_aux;
        pseudo_bin = static_cast<uint32_t>(((1u << ((depth + 1) * 3)) - 1) / 7 + 1);
    }
    int32_t n_ref = load_le<int32_t>(data.data() + p);
    p += 4;
    ranges.assign(n_refs, VirtualRange{0, 0});
    for (int32_t r = 0; r < n_ref; ++r) {
        int32_t n_bin = load_le<int32_t>(data.data() + p);
        p += 4;
        uint64_t mn = kVoffsetEof, mx = 0;
        bool have_pseudo = false;
        VirtualRange pseudo{0, 0};
        for (int32_t b = 0; b < n_bin; ++b) {
            uint32_t bin = load_le<uint32_t>(data.data() + p);
            p += 4;
            if (csi) p += 8;  // loffset
            int32_t n_chunk = load_le<int32_t>(data.data() + p);
            p += 4;
            for (int32_t c = 0; c < n_chunk; ++c) {
                uint64_t beg = load_le<uint64_t>(data.data() + p);
                uint64_t end = load_le<uint64_t>(data.data() + p + 8);
                p += 16;
                if (bin == pseudo_bin) {
                    if (c == 0) {
                        pseudo = {beg, end};
                        have_pseudo = true;
                    }
                } else {
                    mn = std::min(mn, beg);
                    mx = std::max(mx, end);
                }
            }
        }
        if (!csi) {
            int32_t n_intv = load_le<int32_t>(data.data() + p);
            p += 4 + 8 * static_cast<size_t>(n_intv);
        }
        if (static_cast<size_t>(r) < n_refs) {
            if (have_pseudo)
                ranges[r] = pseudo;
            else if (mn != kVoffsetEof)
                ranges[r] = {mn, mx};
        }
    }
    return ranges;
}

// ----------------------------------------------------------- BamWriter ---

int32_t reg2bin(int32_t beg, int32_t end) {
    --end;
    if (beg >> 14 == end >> 14) return ((1 << 15) - 1) / 7 + (beg >> 14);
    if (beg >> 17 == end >> 17) return ((1 << 12) - 1) / 7 + (beg >> 17);
    if (beg >> 20 == end >> 20) return ((1 << 9) - 1) / 7 + (beg >> 20);
    if (beg >> 23 == end >> 23) return ((1 << 6) - 1) / 7 + (beg >> 23);
    if (beg >> 26 == end >> 26) return ((1 << 3) - 1) / 7 + (beg >> 26);
    return 0;
}

void aux_append_string(std::string &aux, const char tag[2], std::string_view value) {
    aux.push_back(tag[0]);
    aux.push_back(tag[1]);
    aux.push_back('Z');
    aux.append(value.data(), value.size());
    aux.push_back('\0');
}

void aux_append_int(std::string &aux, const char tag[2], int32_t value) {
    aux.push_back(tag[0]);
    aux.push_back(tag[1]);
    aux.push_back('i');
    char b[4];
    std::memcpy(b, &value, 4);
    aux.append(b, 4);
}

void aux_append_float(std::string &aux, const char tag[2], float value) {
    aux.push_back(tag[0]);
    aux.push_back(tag[1]);
    aux.push_back('f');
    char b[4];
    std::memcpy(b, &value, 4);
    aux.append(b, 4);
}

namespace {
template <typename T>
void put(std::string &s, T v) {
    char b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T));
    s.append(b, sizeof(T));
}
constexpr size_t kBlockPayload = 0xff00;
}  // namespace

BamWriter::BamWriter(const std::string &path, const BamHeader &header, int n_threads, int level)
    : n_threads_(n_threads), level_(level) {
    fp_ = std::fopen(path.c_str(), "wb");
    if (!fp_) throw std::runtime_error("cannot open for writing: " + path);
    std::string h;
    h.append("BAM\1", 4);
    put<int32_t>(h, static_cast<int32_t>(header.text.size()));
    h.append(header.text);
    put<int32_t>(h, static_cast<int32_t>(header.names.size()));
    for (size_t i = 0; i < header.names.size(); ++i) {
        put<int32_t>(h, static_cast<int32_t>(header.names[i].size() + 1));
        h.append(header.names[i]);
        h.push_back('\0');
        put<int32_t>(h, header.lengths[i]);
    }
    pending_ = std::move(h);
    // Keep the header in its own blocks.
    flush_blocks(true);
}

BamWriter::~BamWriter() {
    try {
        close();
    } catch (...) {
    }
}

void BamWriter::encode(const BamOutRecord &rec) {
    int32_t l_seq = static_cast<int32_t>(rec.seq.size());
    int32_t ref_len = 0;
    for (uint32_t op : rec.cigar) {
        uint32_t o = op & 0xf;
        if (o == 0 || o == 2 || o == 3 || o == 7 || o == 8) ref_len += static_cast<int32_t>(op >> 4);
    }
    int32_t bin = rec.pos >= 0 ? reg2bin(rec.pos, rec.pos + std::max(ref_len, 1)) : 4680;
    std::string &s = pending_;
    size_t start = s.size();
    put<int32_t>(s, 0);  // block_size placeholder
    put<int32_t>(s, rec.ref_id);
    put<int32_t>(s, rec.pos);
    s.push_back(static_cast<char>(rec.name.size() + 1));
    s.push_back(static_cast<char>(rec.mapq));
    put<uint16_t>(s, static_cast<uint16_t>(bin));
    put<uint16_t>(s, static_cast<uint16_t>(rec.cigar.size()));
    put<uint16_t>(s, rec.flag);
    put<int32_t>(s, l_seq);
    put<int32_t>(s, -1);
    put<int32_t>(s, -1);
    put<int32_t>(s, 0);
    s.append(rec.name);
    s.push_back('\0');
    for (uint32_t op : rec.cigar) put<uint32_t>(s, op);
    for (int32_t i = 0; i < l_seq; i += 2) {
        uint8_t hi = nt16_code(rec.seq[i]);
        uint8_t lo = (i + 1 < l_seq) ? nt16_code(rec.seq[i + 1]) : 0;
        s.push_back(static_cast<char>((hi << 4) | lo));
    }
    if (rec.qual.empty())
        s.append(static_cast<size_t>(l_seq), static_cast<char>(0xff));
    else
        s.append(rec.qual);
    s.append(rec.aux);
    int32_t block_size = static_cast<int32_t>(s.size() - start - 4);
    std::memcpy(&s[start], &block_size, 4);
}

void BamWriter::write(const BamOutRecord &rec) {
    encode(rec);
    if (pending_.size() >= 64 * kBlockPayload) flush_blocks(false);
}

void BamWriter::flush_blocks(bool final) {
    size_t n_full = pending_.size() / kBlockPayload;
    size_t n_blocks = n_full + ((final && pending_.size() % kBlockPayload) ? 1 : 0);
    if (n_blocks == 0) return;
    std::vector<std::string> compressed(n_blocks);
    bool failed = false;
    parallel_dynamic(n_blocks, n_threads_, [&](int, size_t i) {
        thread_local ThreadDeflater def;
        size_t off = i * kBlockPayload;
        size_t len = std::min(kBlockPayload, pending_.size() - off);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(pending_.data()) + off;
        std::string &out = compressed[i];
        out.assign(65536, '\0');
        int lvl = level_;
        for (int attempt = 0; attempt < 2; ++attempt) {
            def.init(lvl);
            def.zs.next_in = const_cast<Bytef *>(src);
            def.zs.avail_in = static_cast<uInt>(len);
            def.zs.next_out = reinterpret_cast<Bytef *>(&out[18]);
            def.zs.avail_out = static_cast<uInt>(65536 - 18 - 8);
            int ret = deflate(&def.zs, Z_FINISH);
            if (ret == Z_STREAM_END) break;
            if (attempt == 1) {
                failed = true;
                return;
            }
            lvl = 0;
        }
        size_t clen = (65536 - 18 - 8) - def.zs.avail_out;
        size_t total = 18 + clen + 8;
        uint8_t hdr[18] = {31, 139, 8, 4, 0, 0, 0, 0, 0, 255, 6, 0, 'B', 'C', 2, 0, 0, 0};
        uint16_t bsize = static_cast<uint16_t>(total - 1);
        std::memcpy(hdr + 16, &bsize, 2);
        std::memcpy(&out[0], hdr, 18);
        uint32_t crc = static_cast<uint32_t>(crc32(0L, src, static_cast<uInt>(len)));
        uint32_t isize = static_cast<uint32_t>(len);
        std::memcpy(&out[18 + clen], &crc, 4);
        std::memcpy(&out[18 + clen + 4], &isize, 4);
        out.resize(total);
    });
    if (failed) throw std::runtime_error("BGZF compression failed");
    for (const auto &c : compressed) std::fwrite(c.data(), 1, c.size(), fp_);
    size_t consumed = std::min(pending_.size(), n_blocks * kBlockPayload);
    pending_.erase(0, consumed);
}

void BamWriter::close() {
    if (closed_ || !fp_) return;
    flush_blocks(true);
    static const uint8_t kEof[28] = {31, 139, 8, 4, 0, 0, 0, 0, 0, 255, 6, 0, 66, 67,
                                     2, 0, 27, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::fwrite(kEof, 1, 28, fp_);
    std::fclose(fp_);
    fp_ = nullptr;
    closed_ = true;
}

}  // namespace cas
