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

import SingleCellLineageTracing as sclt

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

def collapseUMIs(base_dir, fn, max_hq_mismatches = 3, max_indels = 2, max_UMI_distance = 2, n_threads = 1, show_progress=True, force_sort=False):

        base_dir = Path(base_dir)
        sorted_fn = (base_dir / fn).with_suffix('.sorted.bam')

        sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))
        filter_func = lambda al: al.has_tag(CELL_BC_TAG)

        if force_sort or not sorted_fn.exists():
            sorted_fn = '.'.join(fn.split(".")[:-1]) + "_sorted.bam"
            collapse.sort_cellranger_bam(fn, sorted_fn, sort_key, filter_func, show_progress=show_progress)

        collapsed_fn = (base_dir / fn).with_suffix(".collapsed.bam")
        if not collapsed_fn.exists():
            collapse.form_collapsed_clusters(sorted_fn,
                                max_hq_mismatches,
                                max_indels,
                                max_UMI_distance,
                                show_progress=show_progress
                               )
        
        collapsed_df_fn = (base_dir / fn).with_suffix(".collapsed.txt")
        collapseBam2DF(str(collapsed_fn), str(collapsed_df_fn)) 


def errorCorrectUMIs(input_fn, _id, log_file, max_hq_mismatches = 3, max_indels=2, max_UMI_distance=2, show_progress=True):

    sort_key = lambda al: (al.get_tag(LOC_TAG), -1*int(al.query_name.split("_")[-1]))
    
    name = Path(input_fn)
    sorted_fn = name.with_name(name.stem + "_sorted.bam")

    filter_func = lambda al: al.has_tag(LOC_TAG) or al.has_tag(CELL_BC_TAG)

    collapse.sort_cellranger_bam(input_fn, sorted_fn, sort_key, filter_func, show_progress = show_progress)

    collapse.error_correct_allUMIs(sorted_fn, 
                              max_hq_mismatches,
                              max_indels,
                              max_UMI_distance,
                              _id, 
                              log_fh = log_file,
                            show_progress = show_progress)

    ec_fh = sorted_fn.with_name(sorted_fn.stem + "_ec.bam")
    mt_fh = ec_fh.with_suffix(".moleculeTable.txt")

    convert_bam_to_moleculeTable(ec_fh, mt_fh)


def collapseBam2DF(data_fp, out_fp):

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'collapseBam2dataFrame.pl')

    cmd = "perl " + str(perl_script) + " " + data_fp + " " + out_fp 
    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)


def collapseDF2Fastq(data_fp, out_fp):

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'collapseDF2fastq.pl')
    cmd = "perl " + str(perl_script) + " " + data_fp + " " + out_fp

    p = subprocess.check_output(cmd, shell=True)

def align_sequences(ref, queries, outfile, gapopen=20, gapextend=1, ref_format="fasta", query_format="fastq", out_format="sam"):

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

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'callAlleles-PCT48.pl')
    cmd = "perl " + str(perl_script) + " " + alignments + " " + ref + " " + output
    if context:
        cmd += " --context"

    cmd += " > _log.stdout 2> _log.stderr"

    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)

    bam_file = str(Path(output).with_suffix(".bam"))

    convert_sam_to_bam(output, bam_file) 

def append_sample_id(data_fp, out_fp, sampleID):
    
    data = pd.read_csv(data_fp, sep='\t')

    data["cellBC"] = data.apply(lambda x: sampleID + "." + x.cellBC, axis=1)

    return data

def convert_sam_to_bam(sam_input, bam_input):

    cmd = "samtools view -S -b " + sam_input + " > " + bam_input

    os.system(cmd)

def convert_bam_to_moleculeTable(bam_input, mt_out):

    perl_script = (SCLT_PATH / 'ProcessingPipeline' / 'process' / 'processBam2MT.pl')
    cmd = "perl " + str(perl_script) + " " + str(bam_input) + " " + str(mt_out) 

    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)

def filter_molecule_table(mt, out_fp, outputdir, cell_umi_thresh = 10, umi_read_thresh = None, intbc_prop_thresh=0.5, intbc_umi_thresh=10, intbc_dist_thresh=1, verbose=False, ec_intbc = False, detect_intra_doublets=True, doublet_threshold=0.35):

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

def call_lineage_groups(mt, out_fp, outputdir, min_cluster_prop=0.005, min_intbc_thresh=0.05, detect_doublets_inter=True, doublet_threshold=0.35, no_filter_intbcs=False, verbose=False, cell_umi_filter=10, filter_intbc_thresh = 0.001):

    args = ["call-lineages", mt, out_fp, outputdir, "--min_cluster_prop", str(min_cluster_prop), "--min_intbc_thresh", str(min_intbc_thresh), "--doublet_threshold", str(doublet_threshold), "--cell_umi_filter", str(cell_umi_filter), "--filter_intbc_thresh", str(filter_intbc_thresh)]

    if no_filter_intbcs:
        args.append("--no_filter_intbcs")

    if verbose:
        args.append("--verbose")

    if detect_doublets_inter:
        args.append("--detect_doublets_inter")

    print(" ".join(args))
    subprocess.check_output(args)



def filterCellBCs(data, outputdir, umi_per_cellBC_thresh, avg_reads_per_UMI_thresh, verbose=True):
    """
    Filter out cell barcodes that have too few UMIs or too few reads/UMI
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

def resolveSequences(moleculetable, outputdir, verbose=True):
    """
    Pick most abundant sequence from a set of equivalent reads (i.e. same cellBC and UMI)
    Output a single sequence per unique cellBC-UMI pair
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

def changeCellBCID(alleleTableIN, sampleID, alleleTableOUT):

    data = pd.read_csv(alleleTableIN, sep='\t')

    data['cellBC'] = data.apply(lambda x: x.cellBC + "-" + sampleID, axis=1)

    data.to_csv(alleleTableOUT, sep='\t')

def pickSeq(moltable, out_fp, outputdir, cell_umi_thresh = 10, avg_reads_per_UMI_thresh = 2.0, verbose=True, save_output=False): 

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
    mt = filterCellBCs(mt, outputdir, cell_umi_thresh, avg_reads_per_UMI_thresh)[0]

    if save_output:
        mt.to_csv(out_fp, sep='\t')
        return

    return mt
