// Per-read allele calling from an alignment to the target-site reference.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cas {

enum class EditModel : int { Indel = 0, InsertionLookup = 1, SequentialInsertion = 2, Substitution = 3 };

struct SiteSpec {
    std::string name;
    EditModel model = EditModel::Indel;
    int position = 0;      // 0-based reference coordinate of the cut / insertion point
    int window_left = 0;   // window = [position - window_left, position + window_right]
    int window_right = 0;
    // indel model
    bool context = true;
    int context_size = 5;
    // insertion_lookup model: inserted sequence -> state label
    std::unordered_map<std::string, std::string> lookup;
    // substitution model: [start, end) reference window
    int start = 0;
    int end = 0;
    char from_base = 'A';
    char to_base = 'G';
    int min_base_qual = 0;
    // sequential_insertion model: slot index within the array
    int slot = 0;
};

// Tandem array of identical units that are edited sequentially (DNA
// Typewriter TAPE). Insertions of `insert_len` bases precede each unit.
struct ArraySpec {
    bool enabled = false;
    int start = -1;          // reference coordinate where the first slot begins
    std::string unit;        // repeated unit sequence (as it appears unedited)
    int insert_len = 0;
    int n_units = 0;         // maximum number of slots
    std::string end_flank;   // sequence following the last unit
    int max_mismatch = 1;
    // Reference coordinate where sequence parsing starts (<= start). The
    // bases [anchor, start) are matched from the read before the first slot,
    // so parsing does not depend on where an aligner places indels inside
    // the repeat. Defaults to start.
    int anchor = -1;
};

struct ConstructSpec {
    std::string reference;
    int intbc_start = -1;    // [start, end) of the intBC in the reference; -1 if none
    int intbc_end = -1;
    bool clip_missing = true;
    bool normalize_indels = true;
    std::vector<SiteSpec> sites;
    ArraySpec array;
};

inline void set_site_model(SiteSpec &s, int model) { s.model = static_cast<EditModel>(model); }

struct ReadCall {
    std::string intbc;                // empty when it could not be read
    std::vector<std::string> tokens;  // per site; empty string = missing
    std::vector<uint8_t> clipped;     // per site; 1 if missing due to clipping / read end
};

// Calls the intBC and per-site tokens for one read. `cigar` is the full BAM
// CIGAR (may contain S/H/N/=/X). `qual` may be null. Returns false if the
// alignment is unusable (no aligned bases).
bool call_read(const ConstructSpec &spec, int ref_pos, const std::vector<uint32_t> &cigar,
               const std::string &seq, const uint8_t *qual, ReadCall &out);

// Left-align indels in a CIGAR (in place). Only shifts across matching
// bases so no new mismatches are introduced.
void left_align_indels(const std::string &ref, int ref_pos, const std::string &seq,
                       std::vector<uint32_t> &cigar);

}  // namespace cas
