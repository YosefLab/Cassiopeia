// BAM -> molecule table.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "caller.hpp"

namespace cas {

struct CallAllelesOptions {
    std::string path;
    std::vector<int> tids;     // reference ids to process; empty = all
    bool use_index = true;     // restrict reading to `tids` via the index
    std::string cb_tag = "CB";
    std::string umi_tag = "UB";
    std::string count_tag;     // e.g. "ZR"; empty = each record counts once
    int min_mapq = 0;
    int n_threads = 1;
    std::vector<std::string> cells;  // allowed cell barcodes; empty = all
    int umi_max_dist = 0;      // UMI correction within (cell, intBC)
    int64_t min_reads = 0;     // drop UMIs with fewer reads (with UMIs only)
    bool intbc_from_contig = false;
    std::string default_intbc = "0";  // intBC when the construct has none
};

struct Categorical {
    std::vector<std::string> categories;
    std::vector<int32_t> codes;  // -1 = missing
};

struct MoleculeTable {
    Categorical cell, umi, intbc;
    std::vector<Categorical> sites;
    std::vector<int32_t> n_reads;
    std::vector<std::pair<std::string, int64_t>> stats;
    std::vector<int64_t> site_missing;  // reads with a missing call per site
    std::vector<int64_t> site_clipped;  // reads missing because of clipping / read end
    std::vector<int64_t> hist_reads, hist_umis;  // UMIs per read count, before `min_reads`
};

MoleculeTable call_alleles(const ConstructSpec &spec, const CallAllelesOptions &opt);

// Reads only the header (reference names / lengths) of a BAM.
void bam_references(const std::string &path, std::vector<std::string> &names, std::vector<int32_t> &lengths);

}  // namespace cas
