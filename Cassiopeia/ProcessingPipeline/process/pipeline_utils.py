import sys
import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
from pathlib import Path
import bokeh.palettes
import time
import matplotlib.pyplot as plt
import subprocess

import Cassiopeia as sclt

SCLT_PATH = Path(sclt.__path__[0])

from . import collapse

SAM_HEADER_PCT48 = "@HD	VN:1.3\n@SQ	SN:PCT48.ref	LN:750" 
CELL_BC_TAG = 'CB'
UMI_TAG = 'UR'
NUM_READS_TAG = 'ZR'
CLUSTER_ID_TAG = 'ZC'
LOC_TAG = "BC"
CO_TAG = "CO"

HIGH_Q = 31
LOW_Q = 10
N_Q = 2

def collapseUMIs(base_dir, fn, max_hq_mismatches = 3, max_indels = 2, max_UMI_distance = 2, n_threads = 1, show_progress=True, force_sort=True):
    """
    Collapses UMIs together from a bam file. On a basic level, it aggregates together identical reads to count how many times a UMI was read.
    Also, it performs basic error correction, allowing UMIs to be collapsed together which differ by at most a certain number of high quality 
    mismatches and indels in the sequence read itself. Writes out a dataframe of the collapsed UMIs table.

    :param base_dir:
        Base directory in which to look for a bam file. This will also be where the collapsed matrix is written to.
    :param fn:
        File name of the bam file -- just the name, not the entire file path.
    :param max_hq_mismatches:
        Maximum number of high quality mismatches allowed between two seqeunces to be collapsed.
    :param max_indels:
        Maximum number of indels allowed between two sequences to be collapsed.
    :param n_threads:
        Number of threads used. Currently only supports single threaded use.
    :param show_progress:
        Allow progress bar to be shown.
    :param force_sort:
        Sort the initial bam directory. 
    :return:
        None; output table is written to file.
    """


    base_dir = Path(base_dir)
    sorted_fn = Path('.'.join(str(base_dir / fn).split(".")[:-1]) + "_sorted.bam")

    sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))
    filter_func = lambda al: al.has_tag(CELL_BC_TAG)

    print(str(sorted_fn))
    if force_sort or not sorted_fn.exists():
        sorted_fn = Path('.'.join(fn.split(".")[:-1]) + "_sorted.bam")
        collapse.sort_cellranger_bam(fn, str(sorted_fn), sort_key, filter_func, show_progress=show_progress)

    collapsed_fn = sorted_fn.with_suffix(".collapsed.bam")
    print(str(collapsed_fn))
    if not collapsed_fn.exists():
        collapse.form_collapsed_clusters(str(sorted_fn),
                            max_hq_mismatches,
                            max_indels,
                            max_UMI_distance,
                            show_progress=show_progress
                           )
    
    collapsed_df_fn = sorted_fn.with_suffix(".collapsed.txt")
    collapseBam2DF(str(collapsed_fn), str(collapsed_df_fn)) 


def errorCorrectUMIs(input_fn, _id, log_file, max_UMI_distance=2, show_progress=True):
    """
    Error correct UMIs together within equivalence classes, as defined as the same cellBC-intBC pair. UMIs whose identifier 
    is within the maximum UMI distance are corrected towards whichever UMI is more abundant. 

    :param input_fn:
        Input file name.
    :param _id:
        Identification of sample.
    :param log_file:
        Filepath for logging error correction information.
    :param max_UMI_distance:
        Maximum UMI distance allowed for error correction.
    :param show_progress:
        Allow a progress bar to be shown.
    :return:
        None; a table of error corrected UMIs is written to file.

    """

    sort_key = lambda al: (al.get_tag(LOC_TAG), -1*int(al.query_name.split("_")[-1]))
    
    name = Path(input_fn)
    sorted_fn = name.with_name(name.stem + "_sorted.bam")

    filter_func = lambda al: al.has_tag(LOC_TAG) or al.has_tag(CELL_BC_TAG)

    collapse.sort_cellranger_bam(input_fn, sorted_fn, sort_key, filter_func, show_progress = show_progress)

    collapse.error_correct_allUMIs(sorted_fn, 
                              max_UMI_distance,
                              _id, 
                              log_fh = log_file,
                            show_progress = show_progress)

    ec_fh = sorted_fn.with_name(sorted_fn.stem + "_ec.bam")
    mt_fh = Path(str(ec_fh).split(".")[0] + ".moleculeTable.txt")

    convert_bam_to_moleculeTable(ec_fh, mt_fh)


def collapseBam2DF(data_fp, out_fp):
    """
    Collapses a BAM file to a Dataframe. 

    :param data_fp:
        BAM file input.
    :param out_fp:
        Filepath for output DataFrame.
    :return:
        None.
    """

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'collapseBam2dataFrame.pl')

    cmd = "perl " + str(perl_script) + " " + data_fp + " " + out_fp 
    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)


def collapseDF2Fastq(data_fp, out_fp):
    """
    Collapses a DataFrame to a FASTQ file.

    :param data_fp:
        Dataframe input file.
    :param out_fp:
        Filepath for output FASTQ file.
    :return:
        None.
    """

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'collapseDF2fastq.pl')
    cmd = "perl " + str(perl_script) + " " + data_fp + " " + out_fp

    p = subprocess.check_output(cmd, shell=True)

def align_sequences(ref, queries, outfile, gapopen=20, gapextend=1, ref_format="fasta", query_format="fastq", out_format="sam"):
    """
    Aligns many queries to a single reference sequence using EMBOSS water. By default, we assume the reference is in FASTA format, the queries 
    are all in FASTQ format, and that the output format will be a SAM file. The output file is automatically written.

    :param ref:
        File path to the reference sequence.
    :param queries:
        Queries, provided as a dataframe output from the `pickSeq` function. This will automatically be converted to a FASTQ file.
    :param gapopen:
        Gap open penalty.
    :param gapextend:
        Gap extension penalty.
    :param ref_format:
        Format of reference sequence.
    :param query_format:
        Format of query seqeunces.
    :param out_format:
        Output file format.
    :return:
        None. 
    """

    queries_fastq = str(Path(queries).with_suffix(".fastq"))
    collapseDF2Fastq(queries, queries_fastq)  

    cmd = "water -asequence " + ref + " -sformat1 " + ref_format  + " -bsequence " + queries_fastq + " -sformat2 " + query_format + " -gapopen " + str(gapopen) + " -gapextend " + str(gapextend) + " -outfile " + outfile + " -aformat3 " + out_format

    cmd = cmd.split(" ")

    subprocess.check_output(cmd)

    with open(outfile, "r+") as f:
        content = f.read()
        f.seek(0,0)
        f.write(SAM_HEADER_PCT48 + "\n" + content)

def call_indels(alignments, ref, output, context=True):
    """
    Given many alignments, we extract the indels by comparing the CIGAR strings in the alignments to the reference sequence. 

    :param alignments:
        Alignments provided in SAM or BAM format.
    :param ref:
        File path to the reference seqeunce, assumed to be a FASTA.
    :param output:
        Output file path.
    :param context:
        Include sequence context around indels.
    :return:
        None
    """

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'callAlleles-PCT48.pl')
    cmd = "perl " + str(perl_script) + " " + alignments + " " + ref + " " + output
    if context:
        cmd += " --context"

    cmd += " > _log.stdout 2> _log.stderr"

    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)

    bam_file = str(Path(output).with_suffix(".bam"))

    convert_sam_to_bam(output, bam_file) 

def append_sample_id(data_fp, sampleID):
    """
    Append sample IDs to the cellBCs of a given molecule table. 

    :param data_fp:
        DataFrame file path.
    :param sampleID:
        Sample ID to be appended to the cellBCs.
    :return:
        New molecule table Dataframe with modified cellBCs.
    """
    
    data = pd.read_csv(data_fp, sep='\t')

    data["cellBC"] = data.apply(lambda x: sampleID + "." + x.cellBC, axis=1)

    return data

def convert_sam_to_bam(sam_input, bam_output):
    """
    Converts a SAM file to BAM file.

    :param sam_input:
        Sam input file 
    :param bam_output:
        File path to write the bam output.
    :return:
        None.
    """

    cmd = "samtools view -S -b " + sam_input + " > " + bam_output

    os.system(cmd)

def convert_bam_to_moleculeTable(bam_input, mt_out):
    """
    Converts a BAM file to a molecule table Dataframe.

    :param bam_input:
        BAM input file 
    :param mt_out:
        File path to write the molecule table.
    :return:
        None.
    """

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'processBam2MT.pl')
    cmd = "perl " + str(perl_script) + " " + str(bam_input) + " " + str(mt_out) 

    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)

def filter_molecule_table(mt, out_fp, outputdir, cell_umi_thresh = 10, umi_read_thresh = None, intbc_prop_thresh=0.5, intbc_umi_thresh=10, 
                intbc_dist_thresh=1, verbose=False, ec_intbc = False, detect_intra_doublets=True, doublet_threshold=0.35):
    """
    Wrapper function for interacting with the `Filter Molecule Table` module. Takes in a molecule table file path and performs cellBC filtering, 
    UMI filtering, intBC filtering, intBC error correction, and intra doublet detection.

    :param mt:
        File path to molecule table to be filtered.
    :param out_fp:
        Output file name. This will be written to `outputdir`.
    :param outputdir:
        Directory to output all logs and files.
    :param cell_umi_thresh:
        Cell UMI Threshold for filtering.
    :param umi_read_thresh:
        UMI read threshold for filtering.
    :param intbc_prop_thresh:
        A minimum proportion of reads required for the more abundant intBC during intBC error correction. If the proportion is not met, then 
        error correction is not performed.
    :param intbc_umi_thresh:
        A minimum number of UMIs that need to be observed for the m ore abundant intBC during intBC error correction.
    :param intbc_dist_thresh:
        A hamming distance threshold for considering intBCs to be error corrected.
    :param verbose:
        Verbose output, consisting of output to log files.
    :param ec_intbc:
        Error correct integration barcodes.
    :param detect_intra_doublets:
        Perform intra doublet detection.
    :param doublet_threshold:
        Threshold to be used during intra doublet detection.
    :return:
        None. A filtered molecule table is written to file.
    """


    args = ["filter-molecule-table", mt, out_fp, outputdir,  "--cell_umi_thresh", str(cell_umi_thresh), "--intbc_prop_thresh", str(intbc_prop_thresh), "--intbc_umi_thresh", str(intbc_umi_thresh), "--intbc_dist_thresh", str(intbc_dist_thresh)]

    if umi_read_thresh:
        args.append("--umi_read_thresh " + str(umi_read_thresh))
    if verbose:
        args.append("--verbose")
    if ec_intbc:
        args.append("--ec_intbc")
    if detect_intra_doublets:
        args.append("--detect_doublets_intra")
        args.append("--doublet_threshold")
        args.append(str(doublet_threshold))

    subprocess.check_output(args)

def call_lineage_groups(mt, out_fp, outputdir, cell_umi_filter=10, min_cluster_prop=0.005, min_intbc_thresh=0.05, detect_doublets_inter=True, doublet_threshold=0.35, 
            no_filter_intbcs=False, filter_intbc_thresh = 0.001, verbose=False):
    """
    Wrapper function for interacting with the `Lineage Group` module. Takes in a filtered molecule table and preforms lineage group calling, 
    inter doublet detection, intBC filtering, and a final round of cellBC filtering. 

    :param mt:
        File path to the filtered molecule table.
    :param out_fp:
        Output file name. This will be written to `outputdir`.
    :param outputdir:
        Directory to output all logs and files.
    :param cell_umi_thresh:
        Cell UMI Threshold for filtering.
    :param min_cluster_prop:
        Lower bound of lineage group size, as defined as a proportion of the total number of cells.
    :param min_intbc_thresh:
        Filtering criteria for intBC at the lineage group level -- the minimum proportion of cells that must have an intBC for the intBC to be
        considered legitimate.
    :param detect_doublets_inter:
        Perform inter doublet detection.
    :param doublet_threshold:
        Threshold to be used during inter doublet detection.
    :param no_filter_intbcs:
        Do not filter intBCS before calling lineage groups.
    :param filter_intbc_thresh:
        Filtering criteria for intBC before calling lineage groups -- the minimum number of cells overall that contain that intBC in order
        for it to be used during lineage group calling.
    :param verobse:
        Allow output to log files.
    :return:
        None. An alleletable is written to file.
    """

    args = ["call-lineages", mt, out_fp, outputdir, "--min_cluster_prop", str(min_cluster_prop), "--min_intbc_thresh", str(min_intbc_thresh), "--doublet_threshold", str(doublet_threshold), "--cell_umi_filter", str(cell_umi_filter), "--filter_intbc_thresh", str(filter_intbc_thresh)]

    if no_filter_intbcs:
        args.append("--no_filter_intbcs")

    if verbose:
        args.append("--verbose")

    if detect_doublets_inter:
        args.append("--detect_doublets_inter")

    print(" ".join(args))
    subprocess.check_output(args)



def filterCellBCs(data, umi_per_cellBC_thresh, avg_reads_per_UMI_thresh):
    """
    Filter out cell barcodes that have too few UMIs or too few reads/UMI

    :param data:
        Molecule table
    :param umi_per_cellBC_thresh:
        Minimum number of UMIs per cellBC allowed.
    :param avg_reads_per_UMI_thresh:
        Minimum average number of reads per UMI allowed. 
    :return:
        A filtered dataset and a stat dictionary storing how many cells/UMIs were filtered/kept.
    """

    tooFewUMI_UMI = []
    cellBC2nM = {}

    # Create a cell-filter dictionary for hash lookup later on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(data.groupby(["cellBC"])):
        umi_per_cellBC_n = group.shape[0]
        reads_per_cellBC_n = group.agg({"readCount":'sum'}).readCount
        avg_reads_per_UMI_n = float(reads_per_cellBC_n)/float(umi_per_cellBC_n)
        if (umi_per_cellBC_n <= umi_per_cellBC_thresh) or (avg_reads_per_UMI_n <= avg_reads_per_UMI_thresh):
            cell_filter[n] = "bad"
            tooFewUMI_UMI.append(group.shape[0])
        else:
            cell_filter[n] = "good"
            cellBC2nM[n] = group.shape[0]

    # apply the filter using the hash table created above
    data["status"] = data["cellBC"].map(cell_filter)

    # count how many cells/umi's passed the filter for logging purposes
    status = cell_filter.values()
    tooFewUMI_cellBC = len(status) - len(np.where(status == "good")[0])
    tooFewUMI_UMI = np.sum(tooFewUMI_UMI)
    goodumis = data[(data["status"] == "good")].shape[0]

    # return filtered data table
    n_data = data[(data["status"] == "good")]

    stat_dict = {"cells_kept": len(n_data["cellBC"].unique()), "num_umi_kept": goodumis, "cells_removed": tooFewUMI_cellBC, "num_umi_removed": tooFewUMI_UMI}

    return n_data, stat_dict

def resolveSequences(moleculetable, outputdir):
    """
    Pick most abundant sequence from a set of equivalent reads (i.e. same cellBC and UMI)
    Output a single sequence per unique cellBC-UMI pair

    :param moleculetable:
        Moleculetable relating cellBCs to UMIs and reads.
    :param outputdir:
        Output directory to store plots and logs.
    :return:
        A molecule table with unique mappings between cellBC-UMI pairs and sequences.
    """

    mt_filter = {}
    total_numReads = {}
    top_reads = {}
    second_reads = {}
    first_reads = {}

    for n, group in tqdm(moleculetable.groupby(["cellBC", "UMI"])):
        if group.shape[0] == 1:
            good_readName = group["readName"].iloc[0]
            mt_filter[good_readName] = "good"
            total_numReads[good_readName] = group["readCount"]
            top_reads[good_readName] = group["readCount"]
            # second_reads[good_readName] = group["readCount"]
        else:
            group_sort = group.sort_values("readCount", ascending=False).reset_index()
            good_readName = group_sort["readName"].iloc[0]
            mt_filter[good_readName] = "good" # add the first entry (highest readCount) to "good"
        
            total_numReads[good_readName] = group_sort["readCount"].sum()
            top_reads[good_readName] = group_sort["readCount"].iloc[0]
            second_reads[good_readName] = group_sort["readCount"].iloc[1]
            first_reads[good_readName] = group_sort["readCount"].iloc[0]

            for i in range(1,group.shape[0]):
                bad_readName = group_sort["readName"].iloc[i]
                mt_filter[bad_readName] = "bad" # add the rest of the entry(ies) (lowest readCount(s)) to "bad"

    # apply the filter using the hash table created above
    moleculetable["status"] = moleculetable["readName"].map(mt_filter)

    # filter based on status & reindex
    n_moleculetable = moleculetable[(moleculetable["status"] == "good")]

    h = plt.figure(figsize=(14, 10))
    plt.plot(top_reads.values(), total_numReads.values(), "r.")
    plt.ylabel("Total Reads")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Total vs. Top Reads for Picked Sequence")
    plt.savefig(outputdir + "/total_vs_top_reads_pickSeq.png")
    plt.close()
    
    h = plt.figure(figsize=(14, 10))
    plt.plot(first_reads.values(), second_reads.values(), "r.")
    plt.ylabel("Number Reads for Second Best Sequence")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Second Best vs. Top Reads for Picked Sequence")
    plt.savefig(outputdir + "/second_vs_top_reads_pickSeq.png")
    plt.close()

    return n_moleculetable

def pickSeq(moltable, out_fp, outputdir, cell_umi_thresh = 10, avg_reads_per_UMI_thresh = 2.0, save_output=False): 
    """
    Procedure for filtering out UMIs and cellBCs and then resolving sequences with the `resolveSequences` function.

    :param moltable:
        Moleculetable relating cellBCs to UMIs and reads.
    :param out_fp:
        Output file name.
    :param outputdir:
        Output directory for logs, new molecule tables, and plots.
    :param avg_reads_per_UMI_thresh:
        Minimum average number of reads per UMI allowed. 
    :param cell_umi_thresh:
        Minimum number of UMIs per cell allowed.
    :param save_output:
        Write new molecule table to file.
    :return:
        A molecule table with unique mappings between cellBC-UMI pairs and sequences if save_output is false. Else None.
    """

    # Log time
    t0 = time.time()

    print(">>> READING DATA IN...")
    # read in allele table and apply desired transformmtions
    mt = pd.read_csv(moltable, sep='\t')

    print(">>> PLOTTING EQUIVALENCE CLASS READS...")
    equivClass_group = mt.groupby(["cellBC", "UMI"]).agg({"grpFlag": 'count'}).sort_values('grpFlag', ascending=False).reset_index()

    h = plt.figure(figsize=(8,5))
    ax = plt.hist(equivClass_group["grpFlag"], bins=range(1,equivClass_group["grpFlag"].max()))
    t = plt.title("Unique Seqs per cellBC+UMI")
    yax = plt.yscale('log', basey=10)
    plt.xlabel("Number of Unique Seqs")
    plt.ylabel("Count (Log)")
    plt.savefig(outputdir + "/" + "seqs_per_equivClass.png")

    print(">>> RESOLVING CONFLICTING SEQUENCES...")
    mt = resolveSequences(mt, outputdir)

    print(">>> FILTERING CELL BARCODES...")
    mt = filterCellBCs(mt, cell_umi_thresh, avg_reads_per_UMI_thresh)[0]

    if save_output:
        mt.to_csv(out_fp, sep='\t')
        return

    return mt
