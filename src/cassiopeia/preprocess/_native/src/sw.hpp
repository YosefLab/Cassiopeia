// Affine-gap Smith-Waterman / Needleman-Wunsch that reproduces the
// behaviour (including tie-breaking) of seq-align / pyseq_align, which the
// legacy Cassiopeia pipeline used. This keeps legacy allele calls
// reproducible while running ~100x faster.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cas {

struct Scoring {
    int match = 5;
    int mismatch = -4;
    int n_score = 0;         // score for any pairing involving N
    int gap_open = 20;       // penalty for the first base of a gap
    int gap_extend = 1;      // penalty for each additional base
};

// CIGAR op codes (BAM numbering).
enum CigarOp : uint32_t { kM = 0, kI = 1, kD = 2, kN = 3, kS = 4, kH = 5, kP = 6, kEq = 7, kX = 8 };

struct AlignResult {
    int score = 0;
    int ref_start = 0;    // 0-based first aligned reference base
    int query_start = 0;  // 0-based first aligned query base
    int ref_end = 0;      // exclusive
    int query_end = 0;    // exclusive
    std::vector<uint32_t> ops;  // (len << 4) | op over the aligned region, M/I/D only
    bool ok = false;
};

class Aligner {
   public:
    explicit Aligner(const Scoring &s);
    // Local alignment (Smith-Waterman), best hit only.
    AlignResult local(std::string_view ref, std::string_view query);
    // Global alignment with free end gaps (seq-align NW, no_end_gap_penalty).
    AlignResult global(std::string_view ref, std::string_view query);

   private:
    void fill(const uint8_t *a, size_t la, const uint8_t *b, size_t lb, bool sw);
    int sub(uint8_t x, uint8_t y) const { return table_[x * 5 + y]; }
    Scoring s_;
    int table_[25];
    int open_;   // seq-align gap_open (negative, = -gap_open + 1)
    int ext_;    // seq-align gap_extend (negative)
    std::vector<uint8_t> ptr_;              // traceback pointers, 2 bits per matrix
    std::vector<int32_t> rows_;             // rolling score rows
    std::vector<int32_t> prof_;             // query-base score profile
    std::vector<int32_t> last_m_, last_a_, last_b_;  // final row (global traceback)
    int32_t best_ = 0;
    size_t best_i_ = 0, best_j_ = 0;
    std::vector<uint8_t> a_, b_;
    size_t w_ = 0, h_ = 0;
};

// Convert ops (M/I/D) over [query_start, query_end) of a query of length
// query_len into a full BAM CIGAR with soft clips; leading/trailing D are
// trimmed (shifting the reference start) and terminal I become S.
std::vector<uint32_t> to_bam_cigar(const AlignResult &r, int query_len, int &ref_pos);

}  // namespace cas
