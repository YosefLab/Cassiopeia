// intBC error correction.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cas {

// Levenshtein distance (bit-parallel when the shorter string is <= 64bp).
int edit_distance(const std::string &a, const std::string &b);

// Map each intBC to a corrected intBC index (or -1 when it cannot be placed).
//  * whitelist non-empty: map to the unique closest whitelist entry within
//    max_dist (index into whitelist); ties or no match -> -1.
//  * otherwise: merge each intBC into the closest more abundant intBC within
//    max_dist whose count is >= min_ratio times its own (index into seqs).
//    Unmerged intBCs map to themselves.
std::vector<int32_t> correct_intbcs(const std::vector<std::string> &seqs, const std::vector<int64_t> &counts,
                                    const std::vector<std::string> &whitelist, int max_dist, double min_ratio,
                                    int n_threads);

}  // namespace cas
