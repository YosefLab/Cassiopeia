// Template implementation for BamReader::for_each_batch.
#pragma once

namespace cas {

template <typename Fn>
void BamReader::for_each_batch(const VirtualRange &range, Fn &&fn, size_t chunk_bytes) {
    if (range.end != kVoffsetEof && range.end <= range.beg) return;
    uint64_t coff = range.beg >> 16;
    size_t skip = range.beg & 0xffff;
    const bool to_eof = range.end == kVoffsetEof;
    const uint64_t end_coff = to_eof ? ~uint64_t(0) : (range.end >> 16);
    const size_t end_uoff = to_eof ? 0 : (range.end & 0xffff);

    std::vector<Block> blocks;
    std::vector<uint8_t> raw, out, buf;
    std::vector<size_t> offs;
    std::vector<BamRecord> recs;
    std::vector<uint8_t> carry;
    bool done = false;
    while (!done && coff < file_size_) {
        uint64_t next = load_blocks(coff, chunk_bytes, blocks, raw);
        if (blocks.empty()) break;
        bool reached_end = false;
        size_t keep = blocks.size();
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].coffset > end_coff) {
                keep = i;
                reached_end = true;
                break;
            }
            if (blocks[i].coffset == end_coff) {
                keep = i + 1;
                reached_end = true;
                break;
            }
        }
        blocks.resize(keep);
        if (blocks.empty()) break;
        inflate_blocks(blocks, raw, out, offs);
        size_t start = std::min(skip, offs.back());
        skip = 0;
        size_t stop = offs.back();
        if (reached_end) {
            if (blocks.back().coffset == end_coff)
                stop = std::min(offs.back(), offs[blocks.size() - 1] + end_uoff);
            done = true;
        }
        if (next >= file_size_) done = true;
        coff = next;

        // Assemble carry-over bytes plus this chunk.
        buf.clear();
        buf.reserve(carry.size() + (stop > start ? stop - start : 0));
        buf.insert(buf.end(), carry.begin(), carry.end());
        if (stop > start) buf.insert(buf.end(), out.begin() + start, out.begin() + stop);
        carry.clear();

        recs.clear();
        size_t pos = 0;
        while (pos + 4 <= buf.size()) {
            uint32_t bs = load_le<uint32_t>(buf.data() + pos);
            if (pos + 4 + bs > buf.size()) break;
            recs.push_back(BamRecord{buf.data() + pos});
            pos += 4 + bs;
        }
        if (!recs.empty()) fn(recs);
        carry.assign(buf.begin() + pos, buf.end());
    }
}

}  // namespace cas
