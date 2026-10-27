// FASTQ -> deduplicated, aligned, CB/UB-tagged BAM.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sw.hpp"

namespace cas {

struct Segment {
    int read = 0;
    int start = 0;
    int end = -1;  // exclusive; -1 = to end of read
};

struct LayoutSpec {
    int n_reads = 2;
    std::vector<Segment> cb, umi;
    Segment cdna{1, 0, -1};
};

struct AlignOptions {
    std::vector<std::vector<std::string>> lanes;  // each lane lists n_reads FASTQ paths
    LayoutSpec layout;
    std::string output_bam;
    std::string whitelist;       // optional; plain or gzipped, one barcode per line
    int min_bc_qual = 10;
    int min_base_qual = 20;      // target-site bases below this are low quality for the UMI consensus
    int umi_max_dist = 1;
    bool dedup_umi = true;       // false = one record per unique sequence
    int max_mismatches = 3;      // clustering radius within a UMI (dedup_umi)
    int max_indels = 2;          // gap bases allowed when merging a UMI's clusters (-1: no merge)
    bool global = false;         // false = local (Smith-Waterman)
    Scoring scoring;
    int min_score = 1;
    int strand = 0;              // 0 forward, 1 reverse complement, 2 both
    int n_threads = 1;
    std::string ref_name;
    std::string ref_seq;
    double cb_confidence = 0.975;
    std::string header_extra;    // extra SAM header lines (each ending in \n)
    size_t batch_size = 200000;
};

struct AlignSummary {
    std::vector<std::pair<std::string, int64_t>> stats;
    std::vector<std::string> cells;
    std::vector<int64_t> cell_reads, cell_umis;
    std::vector<int64_t> reads_per_umi_hist;  // index = reads per molecule (last bin = overflow)
    std::vector<int64_t> score_hist;          // index = alignment score (clipped at 0)
};

AlignSummary align_fastqs(const AlignOptions &opt);

// Legacy test for merging two clusters of a UMI's reads (lowercase bases are
// low quality); exposed for parity tests.
bool merge_check(const std::string &a, const std::string &b, int max_indels, int max_mismatches);

// Align a list of sequences to a reference (exposed for testing / parity).
struct SimpleAlignment {
    int score;
    int ref_start;
    int query_start;
    std::string cigar;  // M/I/D over the aligned region (legacy style)
};
std::vector<SimpleAlignment> align_sequences(const std::string &ref, const std::vector<std::string> &queries,
                                             const Scoring &scoring, bool global, int n_threads);

}  // namespace cas
