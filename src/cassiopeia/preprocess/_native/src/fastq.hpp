// Buffered (gzipped) FASTQ reading.
#pragma once

#include <zlib.h>

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cas {

// A batch of FASTQ records stored contiguously.
struct FastqBatch {
    std::string data;
    std::vector<uint32_t> seq_off, qual_off, len;
    size_t size() const { return len.size(); }
    std::string_view seq(size_t i) const { return std::string_view(data.data() + seq_off[i], len[i]); }
    std::string_view qual(size_t i) const { return std::string_view(data.data() + qual_off[i], len[i]); }
    void clear() {
        data.clear();
        seq_off.clear();
        qual_off.clear();
        len.clear();
    }
};

class FastqReader {
   public:
    explicit FastqReader(const std::string &path) : path_(path) {
        gz_ = gzopen(path.c_str(), "rb");
        if (!gz_) throw std::runtime_error("cannot open FASTQ: " + path);
        gzbuffer(gz_, 1 << 20);
        buf_.resize(1 << 22);
    }
    ~FastqReader() {
        if (gz_) gzclose(gz_);
    }
    FastqReader(const FastqReader &) = delete;
    FastqReader &operator=(const FastqReader &) = delete;

    // Appends up to n records to the batch; returns number appended.
    size_t read_batch(FastqBatch &b, size_t n) {
        size_t got = 0;
        size_t starts[4], lens[4];
        while (got < n) {
            int k = locate_record(starts, lens);
            if (k == 0) break;
            if (k < 4) throw std::runtime_error("truncated FASTQ record in " + path_);
            const char *base = buf_.data();
            if (lens[0] == 0 || base[starts[0]] != '@')
                throw std::runtime_error("malformed FASTQ (expected '@') in " + path_);
            if (lens[1] != lens[3]) throw std::runtime_error("sequence and quality lengths differ in " + path_);
            b.seq_off.push_back(static_cast<uint32_t>(b.data.size()));
            b.data.append(base + starts[1], lens[1]);
            b.qual_off.push_back(static_cast<uint32_t>(b.data.size()));
            b.data.append(base + starts[3], lens[3]);
            b.len.push_back(static_cast<uint32_t>(lens[1]));
            ++got;
        }
        return got;
    }

   private:
    // Finds the next 4 lines starting at pos_, refilling the buffer as
    // needed (only compacting at record boundaries). Returns the number of
    // lines found (0 at clean EOF) and advances pos_ past them.
    int locate_record(size_t starts[4], size_t lens[4]) {
        while (true) {
            // skip blank lines between records
            while (pos_ < end_ && (buf_[pos_] == '\n' || buf_[pos_] == '\r')) ++pos_;
            size_t p = pos_;
            int k = 0;
            while (k < 4) {
                const void *nl = std::memchr(buf_.data() + p, '\n', end_ - p);
                if (!nl) break;
                size_t e = static_cast<const char *>(nl) - buf_.data();
                size_t l = e - p;
                if (l && buf_[p + l - 1] == '\r') --l;
                starts[k] = p;
                lens[k] = l;
                ++k;
                p = e + 1;
            }
            if (k == 4) {
                pos_ = p;
                return 4;
            }
            if (eof_) {
                if (pos_ >= end_) return 0;
                // last line without trailing newline
                if (k == 3 && p < end_) {
                    size_t l = end_ - p;
                    if (l && buf_[p + l - 1] == '\r') --l;
                    starts[3] = p;
                    lens[3] = l;
                    pos_ = end_;
                    return 4;
                }
                pos_ = end_;
                return k == 0 ? 0 : k;
            }
            refill();
        }
    }
    void refill() {
        size_t rem = end_ - pos_;
        if (pos_ > 0) {
            if (rem) std::memmove(buf_.data(), buf_.data() + pos_, rem);
            pos_ = 0;
            end_ = rem;
        }
        if (end_ == buf_.size()) buf_.resize(buf_.size() * 2);
        int got = gzread(gz_, buf_.data() + end_, static_cast<unsigned>(buf_.size() - end_));
        if (got < 0) {
            int err;
            const char *msg = gzerror(gz_, &err);
            throw std::runtime_error(std::string("error reading ") + path_ + ": " + msg);
        }
        if (got == 0) eof_ = true;
        end_ += static_cast<size_t>(got);
    }

    std::string path_;
    gzFile gz_ = nullptr;
    std::vector<char> buf_;
    size_t pos_ = 0, end_ = 0;
    bool eof_ = false;
};

template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
    }
    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }
    // Producer signals completion; consumers drain remaining items.
    void finish() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

   private:
    size_t cap_;
    std::deque<T> q_;
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    bool closed_ = false;
};

}  // namespace cas
